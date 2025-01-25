export const wgslFunctions = {
  commonDefinitions: /* wgsl */ `
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lid: u32 /* 1D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }`,
  workgroupReduce: (
    env,
    bindings = { inputBuffer: "inputBuffer", temp: "temp" }
  ) => {
    return /* wgsl */ `
    /**
     * Approach: Envision threads in a 2D array within workgroup, where
     *     the "width" of that array is one subgroup. First reduce
     *     left to right (within a subgroup), then reduce top to bottom
     *     (across subgroups). Use shared memory to store per-subgroup
     *     reductions.
     * (1) Read one item per thread. If we don't have enough workgroups
     *     to do that, read more. Reduce all those items per-thread.
     *     (The usual case will be "one item per thread".)
     * (2) Reduce within the subgroup. One thread in subgroup n writes
     *     that reduction to temp[n] in shared memory.
     * (3) Subgroup 0 reads that temp array and reduces it. Output is
     *     returned from thread 0.
     *
     * This approach requires an associative reduction operator (this
     * is necessary to take advantage of parallelism). If we read more than
     * one item per thread, it also requires a commutative operator.
     */
  fn workgroupReduce(
    input: ptr<storage, array<${env.datatype}>, read>,
    temp: ptr<workgroup, array<${env.datatype}, 32> >,
    builtins: Builtins
  ) -> ${env.datatype} {
    /* TODO: fix 'assume id.y == 0 always' */
    /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
    var acc: ${env.datatype} = ${env.binop.identity};
    var numSubgroups = roundUpDivU32(${env.workgroupSize}, builtins.sgsz);
    /* note: this access pattern is not particularly TLB-friendly */
    for (var i = builtins.gid.x;
      i < arrayLength(input);
      i += builtins.nwg.x * ${env.workgroupSize}) {
        /* on every iteration, grab wkgpsz items */
        acc = binop(acc, input[i]);
    }
    /* acc contains a partial sum for every thread */
    workgroupBarrier();
    /* now we need to reduce acc within our workgroup */
    /* switch to local IDs only. write into wg memory */
    acc = ${env.binop.subgroupReduceOp}(acc);
    var mySubgroupID = builtins.lid / builtins.sgsz;
    if (subgroupElect()) {
      /* I'm the first element in my subgroup */
      ${bindings.temp}[mySubgroupID] = acc;
    }
    workgroupBarrier(); /* completely populate wg memory */
    if (builtins.lid < builtins.sgsz) { /* only activate 0th subgroup */
      /* read sums of all other subgroups into acc, in parallel across the subgroup */
      /* acc is only valid for lid < numSubgroups, so ... */
      /* select(f, t, cond) */
      acc = select(${env.binop.identity}, ${bindings.temp}[builtins.lid], builtins.lid < numSubgroups);
    }
    /* acc is called here for everyone, but it only matters for thread 0 */
    acc = ${env.binop.subgroupReduceOp}(acc);
    return acc;
        };`;
  },
  workgroupScan: (
    env,
    bindings = { inputBuffer: "inputBuffer", temp: "temp" }
  ) => {
    const scanType = env.type;
    const scanTypeCap = scanType.charAt(0).toUpperCase() + scanType.slice(1);
    const subgroupScanOp = env.binop[`subgroup${scanTypeCap}ScanOp`];
    return /* wgsl */ `
    fn workgroup${scanTypeCap}Scan(builtins: Builtins) -> ${env.datatype} {
        /* TODO: fix 'assume id.y == 0 always' */
    /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
    var numSubgroups = roundUpDivU32(${env.workgroupSize}, builtins.sgsz);
    var i = builtins.gid.x;
    var in = select(${env.binop.identity}, ${bindings.inputBuffer}[i], i < arrayLength(&${bindings.inputBuffer}));
    workgroupBarrier();
    /* "in" now contains the block of data to scan, padded with the identity */
    /* (1) reduce "in" within our workgroup */
    /* switch to local IDs only. write into wg memory */
    var sgReduction = ${env.binop.subgroupReduceOp}(in);
    var mySubgroupID = builtins.lid / builtins.sgsz;
    if (subgroupElect()) {
      /* I'm the first element in my subgroup */
      ${bindings.temp}[mySubgroupID] = sgReduction;
    }
    workgroupBarrier(); /* completely populate wg memory */
    /* Now temp[i] contains reduction of subgroup i */
    /* (2) read sums of all other subgroups into acc, in parallel across the subgroup */
    /** acc is only valid for lid < numSubgroups, but we need uniform control flow
     * for the subgroupScanOp. So the select and subgroup scan are wasted work for
     * all but subgroup == 0. */
    var spineScanInput = select(${env.binop.identity},
                                ${bindings.temp}[builtins.lid],
                                builtins.lid < numSubgroups);
    /* no matter what type of scan we have, we use exclusiveScan here */
    var spineScanOutput = ${env.binop.subgroupExclusiveScanOp}(spineScanInput);
    if (builtins.lid < builtins.sgsz) { /* only activate 0th subgroup */
      ${bindings.temp}[builtins.lid] = spineScanOutput;
    }
    workgroupBarrier();
    /** Now go add that spineScan value back to my local scan. Here's where
     * we differentiate between exclusive/inclusive. */
    var subgroupScan = ${subgroupScanOp}(in);
    return binop(${bindings.temp}[mySubgroupID], subgroupScan);
  };`;
  },
};
