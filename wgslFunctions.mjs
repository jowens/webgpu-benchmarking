/**
 * Order of arguments to functions:
 * 1. builtins
 * 2. outputs, ordered from most global (storage) to most local (workgroup),
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 * 3. inputs, ordered from most global (storage) to most local (workgroup)
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 */

export const wgslFunctions = {
  commonDefinitions: /* wgsl */ `
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lid: u32 /* 1D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsNonuniform {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(local_invocation_index) lid: u32 /* 1D thread index within workgroup */,
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsUniform {
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(subgroup_size) sgsz: u32 /* 32 on Apple GPUs */
  }`,
  roundUpDivU32: /* wgsl */ `fn roundUpDivU32(a : u32, b : u32) -> u32 {
    return (a + b - 1) / b;
  }`,
  workgroupReduce: (env) => {
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
    wgTemp: ptr<workgroup, array<${env.datatype}, 32> >,
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
      wgTemp[mySubgroupID] = acc;
    }
    workgroupBarrier(); /* completely populate wg memory */
    if (builtins.lid < builtins.sgsz) { /* only activate 0th subgroup */
      /* read sums of all other subgroups into acc, in parallel across the subgroup */
      /* acc is only valid for lid < numSubgroups, so ... */
      /* select(f, t, cond) */
      acc = select(${env.binop.identity}, wgTemp[builtins.lid], builtins.lid < numSubgroups);
    }
    /* acc is called here for everyone, but it only matters for thread 0 */
    acc = ${env.binop.subgroupReduceOp}(acc);
    return acc;
        };`;
  },
  workgroupScan: (env) => {
    /**
     * Supports both inclusive and exclusive scan.
     * Arguments:
     * - output: Output array in global storage memory
     * - input: Input array in read-only global storage memory
     * - partials: Input array, one element per workgroup, to be added to workgroup
     * - wgTemp: workgroup temporary memory
     * Operation: Scans workgroup. Adds result of that scan to corresponding
     *   element of "partials" input array.
     * Requires declarations of:
     * - "type" (exclusive or inclusive)
     * - "binop" that, in turn, declares a subgroup{Type}ScanOp
     */
    const scanType = env.type;
    const scanTypeCap = scanType.charAt(0).toUpperCase() + scanType.slice(1);
    const subgroupScanOp = env.binop[`subgroup${scanTypeCap}ScanOp`];
    return /* wgsl */ `
    fn workgroup${scanTypeCap}Scan(builtins: Builtins,
      output: ptr<storage, array<${env.datatype}>, read_write>,
      input: ptr<storage, array<${env.datatype}>, read>,
      partials: ptr<storage, array<${env.datatype}>, read>,
      wgTemp: ptr<workgroup, array<${env.datatype}, 32> >
    ) -> ${env.datatype} {
      /* TODO: fix 'assume wgid.y == 0 always' */
      /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
      var numSubgroups = roundUpDivU32(${env.workgroupSize}, builtins.sgsz);
      var i = builtins.gid.x;
      var in = select(${env.binop.identity}, input[i], i < arrayLength(input));
      workgroupBarrier();
      /* "in" now contains the block of data to scan, padded with the identity */
      /* (1) reduce "in" within our workgroup */
      /* switch to local IDs only. write into wg memory */
      var sgReduction = ${env.binop.subgroupReduceOp}(in);
      var mySubgroupID = builtins.lid / builtins.sgsz;
      if (subgroupElect()) {
        /* I'm the first element in my subgroup */
        wgTemp[mySubgroupID] = sgReduction;
      }
      workgroupBarrier(); /* completely populate wg memory */
      /* Now temp[i] contains reduction of subgroup i */
      /* (2) read sums of all other subgroups into acc, in parallel across the subgroup */
      /** acc is only valid for lid < numSubgroups, but we need uniform control flow
       * for the subgroupScanOp. So the select and subgroup scan are wasted work for
       * all but subgroup == 0. */
      var spineScanInput = select(${env.binop.identity},
                                  wgTemp[builtins.lid],
                                  builtins.lid < numSubgroups);
      /* no matter what type of scan we have, we use exclusiveScan here */
      var spineScanOutput = ${env.binop.subgroupExclusiveScanOp}(spineScanInput);
      /** add reduction of previous workgroups, computed in previous kernel */
      if (builtins.lid < builtins.sgsz) { /* only activate 0th subgroup */
        wgTemp[builtins.lid] = binop(partials[builtins.wgid.x], spineScanOutput);
      }
      workgroupBarrier();
      /** Now go add that spineScan value back to my local scan. Here's where
       * we differentiate between exclusive/inclusive. */
      var subgroupScan = ${subgroupScanOp}(in);
      return binop(wgTemp[mySubgroupID], subgroupScan);
    };`;
  },
  oneWorkgroupExclusiveScan: (env) => {
    /**
     * Arguments:
     * - inputoutput: Input/output array in global storage memory (in place)
     * Returns:
     *   Nothing
     * Operation: Scans workgroup, writes back, one subgroup at a time, serially
     *   Ignores any threads that aren't part of the 0th workgroup
     *   Not efficent
     * Requires declaration of:
     * - "binop" that, in turn, declares a subgroupExclusiveScanOp
     */
    return /* wgsl */ `
    fn oneWorkgroupExclusiveScan(builtinsUniform: BuiltinsUniform,
      builtinsNonuniform: BuiltinsNonuniform,
      inputoutput: ptr<storage, array<${env.datatype}>, read_write>,
    ) {
      var acc : ${env.datatype} = ${env.binop.identity};
      /* making this work under uniform control flow is tricky */
      /* big idea: convert any control dependence to data dependence (i) */
      var ibase : u32 = 0;
      var sg0 = builtinsNonuniform.lid < builtinsUniform.sgsz;
      while (ibase < arrayLength(inputoutput)) {
        /* work still left to be done */
        var i = ibase + builtinsNonuniform.lid;
        var in = select(${env.binop.identity},
                        inputoutput[i],
                        (i < arrayLength(inputoutput)) && sg0);

        var sgEScan = ${env.binop.subgroupExclusiveScanOp}(in);
        var sgReduction = ${env.binop.subgroupReduceOp}(in);
        if (sg0) {
          inputoutput[i] = binop(acc, sgEScan);
          acc = binop(acc, sgReduction);
        }
        var eadd = subgroupExclusiveAdd(in);
        ibase += builtinsUniform.sgsz;
      }
      return;
    };`;
  },
};
