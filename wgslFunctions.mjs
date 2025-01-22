export const wgslFunctions = {
  workgroupReduce: (
    env,
    bindings = { inputBuffer: "inputBuffer", temp: "temp" }
  ) => {
    return /* wgsl */ `
  fn workgroupReduce(id: vec3u,
                     nwg: vec3u,
                     lid: u32,
                     sgsz: u32) -> ${env.datatype} {
    /* TODO: fix 'assume id.y == 0 always' */
    /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
    var acc: ${env.datatype} = ${env.binop.identity};
    var numSubgroups = roundUpDivU32(${env.workgroupSize}, sgsz);
    /* note: this access pattern is not particularly TLB-friendly */
    for (var i = id.x;
      i < arrayLength(&${bindings.inputBuffer});
      i += nwg.x * ${env.workgroupSize}) {
        /* on every iteration, grab wkgpsz items */
        acc = binop(acc, ${bindings.inputBuffer}[i]);
    }
    /* acc contains a partial sum for every thread */
    workgroupBarrier();
    /* now we need to reduce acc within our workgroup */
    /* switch to local IDs only. write into wg memory */
    acc = ${env.binop.subgroupReduceOp}(acc);
    var mySubgroupID = lid / sgsz;
    if (subgroupElect()) {
      /* I'm the first element in my subgroup */
      ${bindings.temp}[mySubgroupID] = acc;
    }
    workgroupBarrier(); /* completely populate wg memory */
    if (lid < sgsz) { /* only activate 0th subgroup */
      /* read sums of all other subgroups into acc, in parallel across the subgroup */
      /* acc is only valid for lid < numSubgroups, so ... */
      /* select(f, t, cond) */
      acc = select(${env.binop.identity}, ${bindings.temp}[lid], lid < numSubgroups);
    }
    /* acc is called here for everyone, but it only matters for thread 0 */
    acc = ${env.binop.subgroupReduceOp}(acc);
    return acc;
        };`;
  },
  workgroupExclusiveScan: (
    env,
    bindings = { inputBuffer: "inputBuffer", temp: "temp" }
  ) => {
    return /* wgsl */ `
    fn workgroupExclusiveScan(id: vec3u,
      nwg: vec3u,
      lid: u32,
      sgsz: u32) -> ${env.datatype} {
        /* TODO: fix 'assume id.y == 0 always' */
    /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
    var numSubgroups = roundUpDivU32(${env.workgroupSize}, sgsz);
    var i = id.x;
    var in = select(${env.binop.identity}, ${bindings.inputBuffer}[i], i < arrayLength(&${bindings.inputBuffer}));
    workgroupBarrier();
    /* "in" now contains the block of data to scan, padded with the identity */
    /* (1) reduce "in" within our workgroup */
    /* switch to local IDs only. write into wg memory */
    var sgReduction = ${env.binop.subgroupReduceOp}(in);
    var mySubgroupID = lid / sgsz;
    if (subgroupElect()) {
      /* I'm the first element in my subgroup */
      ${bindings.temp}[mySubgroupID] = sgReduction;
    }
    workgroupBarrier(); /* completely populate wg memory */
    /* Now temp[i] contains reduction of subgroup i */
    /* (2) read sums of all other subgroups into acc, in parallel across the subgroup */
    /** acc is only valid for lid < numSubgroups, but we need uniform control flow
     * for the subgroupScanOp */
    var spineScanInput = select(${env.binop.identity}, ${bindings.temp}[lid], lid < numSubgroups);
    var spineScanOutput = ${env.binop.subgroupExclusiveScanOp}(spineScanInput);
    if (lid < sgsz) { /* only activate 0th subgroup */
      ${bindings.temp}[lid] = spineScanOutput;
    }
    workgroupBarrier();
    /* now go add that spineScan value back to my local scan */
    var subgroupExclusiveScan = ${env.binop.subgroupExclusiveScanOp}(in);
    return binop(${bindings.temp}[mySubgroupID], subgroupExclusiveScan);
    // return ${bindings.temp}[mySubgroupID];
  };`;
  },
};
