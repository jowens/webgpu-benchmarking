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

/** wgslFunctions requires subgroups
 *  wgslFunctionsWithoutSubgroupSupport does not
 *
 * For writers of kernels who are not writing their own functions:
 * - Declare subgroups with ${this.fnDeclarations.enableSubgroupsIfAppropriate}
 * - The Primitive class picks the right library to include
 *
 * For writers of functions:
 * - If a function doesn't require subgroups, put it in wgslFunctions
 * - If it does, write the subgroup version in wgslFunctions
 *   and the non-subgroup version in wgslFunctionsWithoutSubgroupSupport;
 *   the latter will override the former (this is done in the Primitive class)
 *
 * General philosophy is to design kernels around an expectation that
 * subgroups are present and optimize kernel organization strategies for
 * that case; a non-subgroup-enabled function that provides fallback must
 * be correct but may not be optimized.
 */

import { BinOpAdd } from "./binop.mjs";

export class wgslFunctions {
  constructor(env) {
    this.env = env;
  }
  get commonDefinitions() {
    return /* wgsl */ `
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsNonuniform {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsUniform {
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(subgroup_size) sgsz: u32 /* 32 on Apple GPUs */
  }`;
  }
  get initializeSubgroupVars() {
    return "let sgid = builtinsNonuniform.sgid;\nlet sgsz = builtinsUniform.sgsz;\n";
  }
  get enableSubgroupsIfAppropriate() {
    return "enable subgroups;\n";
  }
  get wgMemoryForSubgroupEmulation() {
    return "";
  }
  get roundUpDivU32() {
    return /* wgsl */ `fn roundUpDivU32(a : u32, b : u32) -> u32 {
    return (a + b - 1) / b;
  }`;
  }
  get computeLinearizedGridParameters() {
    return /* wgsl */ `
    /* wgid is a linearized (1d) unique ID per wg;
     * gid is a linearized (1d) unique ID per thread */
    var wgid = builtins.wgid.z * builtins.nwg.y * builtins.nwg.x +
               builtins.wgid.y * builtins.nwg.x +
               builtins.wgid.x;
    var numThreadsPerWorkgroup: u32 = ${
      this.env.numThreadsPerWorkgroup ?? this.env.workgroupSize
    };
    var gid: u32 = wgid * numThreadsPerWorkgroup + builtins.lidx;
    var workgroupCount = builtins.nwg.z * builtins.nwg.y * builtins.nwg.x;
    var totalThreadCount = workgroupCount * numThreadsPerWorkgroup;`;
  }
  get vec4InclusiveScan() {
    return /* wgsl */ `
    fn vec4InclusiveScan(in: vec4<${this.env.datatype}>) ->
      vec4<${this.env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${this.env.datatype}> = in;
      out.y = binop(in.x,  in.y);
      out.z = binop(out.y, in.z);
      out.w = binop(out.z, in.w);
      return out;
    }`;
  }
  get vec4ExclusiveScan() {
    return /* wgsl */ `
    fn vec4ExclusiveScan(in: vec4<${this.env.datatype}>) ->
      vec4<${this.env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${this.env.datatype}>;
      out.x = ${this.env.binop.identity};
      out.y = in.x;
      out.z = binop(in.x,  in.y);
      out.w = binop(out.z, in.z);
      return out;
    }`;
  }
  get vec4InclusiveToExclusive() {
    return /* wgsl */ `
    fn vec4InclusiveToExclusive(in: vec4<${this.env.datatype}>) ->
      vec4<${this.env.datatype}> {
      var out: vec4<${this.env.datatype}>;
      out.w = in.z;
      out.z = in.y;
      out.y = in.x;
      out.x = ${this.env.binop.identity};
      return out;
    }`;
  }
  get vec4Reduce() {
    // TODO: Don't special-case this. Worried about polyfilling dot with
    // int arguments, that it'll potentially do four multiplies
    if (this.env.binop instanceof BinOpAdd) {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${this.env.datatype}>) -> ${this.env.datatype} {
        return dot(in, vec4<${this.env.datatype}>(1, 1, 1, 1));
      }
      `;
    } else {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${this.env.datatype}>) -> ${this.env.datatype} {
        return binop(binop(binop(in.x, in.y), in.z), in.w);
      }`;
    }
  }
  get vec4ScalarBinopV4() {
    /* if binop is +, this still seems just as efficient, unless there's a vec4 +, I guess? */
    // TODO: "WGSL has mixed vector-scalar arithmetic operators, so it's probably best to use those if you can."
    return /* wgsl */ `
    fn vec4ScalarBinopV4(scalar: ${this.env.datatype}, vector: vec4<${this.env.datatype}>) ->
    vec4<${this.env.datatype}> {
      var out: vec4<${this.env.datatype}>;
      out.x = binop(scalar, vector.x);
      out.y = binop(scalar, vector.y);
      out.z = binop(scalar, vector.z);
      out.w = binop(scalar, vector.w);
      return out;
    }
    `;
  }
  get subgroupZero() {
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return lidx < sgsz;
    }`;
  }
  get subgroupShuffle() {
    /* keep builtin */
    return /* wgsl */ `
    @diagnostic(off, subgroup_uniformity)
    fn subgroupShuffleWrapper(x: ${this.env.datatype}, source: u32, sgid: u32) -> ${this.env.datatype} {
      return subgroupShuffle(x, source);
    }`;
  }
  get subgroupBallot() {
    /* keep builtin */
    return /* wgsl */ `
    @diagnostic(off, subgroup_uniformity)
    fn subgroupBallotWrapper(pred: bool, sgid: u32) -> vec4<u32> {
      return subgroupBallot(pred);
    }`;
  }
  get subgroupMax() {
    /* keep builtin */
    return "";
  }
  get subgroupInclusiveOpScan() {
    /* helpful reference from Thomas Smith:
     *   https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingCUDA/Utils.cuh
     */
    if (this.env.binop.subgroupInclusiveScanOp) {
      /* use the builtin subgroupInclusiveScanOp */
      return /* wgsl */ `
      fn subgroupInclusiveOpScan(in: ${this.env.datatype}, laneID: u32, laneCount: u32) ->
        ${this.env.datatype} {
        return ${this.env.binop.subgroupInclusiveScanOp}(in);
      }
      `;
    } else {
      /* emulate subgroupInclusiveScanOp with subgroupShuffleUp */
      /* for (int i = 1; i <= 16; i <<= 1) { // 16 = LANE_COUNT >> 1
       *   const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
       *   if (getLaneId() >= i) val += t;
       * }
       * return val;
       */
      return /* wgsl */ `
      fn subgroupInclusiveOpScan(in: ${this.env.datatype}, laneID: u32, laneCount: u32) ->
        ${this.env.datatype} {
        var i: u32;
        var val = in;
        for (i = 1; i <= (laneCount >> 1); i <<= 1) {
          let t: ${this.env.datatype} = subgroupShuffleUp(val, i);
          val = binop(select(${this.env.binop.identity}, t, laneID >= i), val);
        }
        return val;
      }
    `;
    }
  }
  get subgroupReduce() {
    /* this will fail if subgroupReduceOp isn't defined; TODO is write it */
    return /* wgsl */ `
    fn subgroupReduceWrapper(in: ${this.env.datatype}, sgid: u32) -> ${this.env.datatype} {
      return ${this.env.binop.subgroupReduceOp}(in);
    }
    `;
  }
  get workgroupReduce() {
    return /* wgsl */ `
    /**
     * Function: Each workgroup's thread 0 returns the reduction of
     *     numThreadsPerWorkgroup items from the input array
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
    input: ptr<storage, array<${this.env.datatype}>, read>,
    wgTemp: ptr<workgroup, array<${this.env.datatype}, 32> >,
    builtins: Builtins
  ) -> ${this.env.datatype} {
    ${this.env.fnDeclarations.computeLinearizedGridParameters}
    /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
    var acc: ${this.env.datatype} = ${this.env.binop.identity};
    var numSubgroups = roundUpDivU32(${this.env.workgroupSize}, builtins.sgsz);
    /* note: this access pattern is not particularly TLB-friendly */
    for (var i = gid;
      i < arrayLength(input);
      i += totalThreadCount) {
        /* on every iteration, grab wkgpsz items */
        acc = binop(acc, input[i]);
    }
    /* acc contains a partial sum for every thread */
    workgroupBarrier();
    /* now we need to reduce acc within our workgroup */
    /* switch to local IDs only. write into wg memory */
    acc = ${this.env.binop.subgroupReduceOp}(acc);
    var mySubgroupID = builtins.lidx / builtins.sgsz;
    if (subgroupElect()) {
      /* I'm the first element in my subgroup */
      wgTemp[mySubgroupID] = acc;
    }
    workgroupBarrier(); /* completely populate wg memory */
    if (builtins.lidx < builtins.sgsz) { /* only activate 0th subgroup */
      /* read sums of all other subgroups into acc, in parallel across the subgroup */
      /* acc is only valid for lid < numSubgroups, so ... */
      /* select(f, t, cond) */
      acc = select(${this.env.binop.identity}, wgTemp[builtins.lidx], builtins.lidx < numSubgroups);
    }
    /* acc is called here for everyone, but it only matters for thread 0 */
    acc = ${this.env.binop.subgroupReduceOp}(acc);
    return acc;
        };`;
  }
  get workgroupScan() {
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
    const scanType = this.env.type;
    const scanTypeCap = scanType.charAt(0).toUpperCase() + scanType.slice(1);
    const subgroupScanOp = this.env.binop[`subgroup${scanTypeCap}ScanOp`];
    return /* wgsl */ `
    fn workgroup${scanTypeCap}Scan(builtins: Builtins,
      output: ptr<storage, array<${this.env.datatype}>, read_write>,
      input: ptr<storage, array<${this.env.datatype}>, read>,
      partials: ptr<storage, array<${this.env.datatype}>, read>,
      wgTemp: ptr<workgroup, array<${this.env.datatype}, 32> >
    ) -> ${this.env.datatype} {
      /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
      ${this.env.fnDeclarations.computeLinearizedGridParameters}
      var numSubgroups = roundUpDivU32(${this.env.workgroupSize}, builtins.sgsz);
      var i = gid;
      var in = select(${this.env.binop.identity}, input[i], i < arrayLength(input));
      workgroupBarrier();
      /* "in" now contains the block of data to scan, padded with the identity */
      /* (1) reduce "in" within our workgroup */
      /* switch to local IDs only. write into wg memory */
      var sgReduction = ${this.env.binop.subgroupReduceOp}(in);
      var mySubgroupID = builtins.lidx / builtins.sgsz;
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
      var spineScanInput = select(${this.env.binop.identity},
                                  wgTemp[builtins.lidx],
                                  builtins.lidx < numSubgroups);
      /* no matter what type of scan we have, we use exclusiveScan here */
      var spineScanOutput = ${this.env.binop.subgroupExclusiveScanOp}(spineScanInput);
      /** add reduction of previous workgroups, computed in previous kernel */
      if (builtins.lidx < builtins.sgsz) { /* only activate 0th subgroup */
        wgTemp[builtins.lidx] = binop(partials[wgid], spineScanOutput);
      }
      workgroupBarrier();
      /** Now go add that spineScan value back to my local scan. Here's where
       * we differentiate between exclusive/inclusive. */
      var subgroupScan = ${subgroupScanOp}(in);
      return binop(wgTemp[mySubgroupID], subgroupScan);
    };`;
  }
  get oneWorkgroupExclusiveScan() {
    /**
     * Arguments:
     * - inputoutput: Input/output array in global storage memory (in place)
     * Returns:
     *   Nothing
     * Restrictions:
     *   Call this with one workgroup.
     * Operation: Scans workgroup, writes back, one subgroup at a time, serially
     *   Ignores any threads that aren't part of the 0th workgroup
     *   Not efficent. Not even close to being efficient.
     * Requires declaration of:
     * - "binop" that, in turn, declares a subgroupExclusiveScanOp
     */
    return /* wgsl */ `
    fn oneWorkgroupExclusiveScan(builtinsUniform: BuiltinsUniform,
      builtinsNonuniform: BuiltinsNonuniform,
      inputoutput: ptr<storage, array<${this.env.datatype}>, read_write>,
    ) {
      var acc : ${this.env.datatype} = ${this.env.binop.identity};
      /* making this work under uniform control flow is tricky */
      /* big idea: convert any control dependence to data dependence (i) */
      var ibase : u32 = 0;
      var sg0 = builtinsNonuniform.lidx < builtinsUniform.sgsz;
      while (ibase < arrayLength(inputoutput)) {
        /* work still left to be done */
        var i = ibase + builtinsNonuniform.lidx;
        var in = select(${this.env.binop.identity},
                        inputoutput[i],
                        (i < arrayLength(inputoutput)) && sg0);

        var sgEScan = ${this.env.binop.subgroupExclusiveScanOp}(in);
        var sgReduction = ${this.env.binop.subgroupReduceOp}(in);
        if (sg0) {
          inputoutput[i] = binop(acc, sgEScan);
          acc = binop(acc, sgReduction);
        }
        var eadd = subgroupExclusiveAdd(in);
        ibase += builtinsUniform.sgsz;
      }
      return;
    };`;
  }
}

/** Philosophy of subgroup fallbacks:
 * - Emulate subgroup operations via workgroup memory
 * - This requires workgroup barriers since we can make no assumptions
 *   about SIMD width / lockstep execution
 * - Given that we HAVE to have workgroup barriers, might as well support
 *   large workgroups
 * - So assume that the emulated subgroup size == workgroup size (1 subgroup
 *   per workgroup)
 * - Implementations are generally O(n log n) (Hillis-Steele / Kogge-Stone)
 */

export class wgslFunctionsWithoutSubgroupSupport extends wgslFunctions {
  constructor(env) {
    super(env);
  }
  get commonDefinitions() {
    return /* wgsl */ `
    struct Builtins {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
      sgsz: u32, /* 32 on Apple GPUs */
      sgid: u32 /* 1D thread index within subgroup */
    }
    struct BuiltinsNonuniform {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */
    }
    struct BuiltinsUniform {
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */
    }`;
  }
  get initializeSubgroupVars() {
    return /* wgsl */ `
    let sgsz: u32 = ${this.env.workgroupSize};
    let sgid: u32 = builtinsNonuniform.lidx;`;
  }
  get enableSubgroupsIfAppropriate() {
    /* don't enable subgroups */
    return "";
  }
  get wgMemoryForSubgroupEmulation() {
    return /* wgsl */ `var<workgroup> wg_sw_subgroups: array<${this.env.datatype}, ${this.env.workgroupSize}>;\n`;
  }
  get subgroupZero() {
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return true;
    }`;
  }
  get subgroupShuffle() {
    return /* wgsl */ `fn subgroupShuffleWrapper(x: ${this.env.datatype}, source: u32, sgid: u32) -> ${this.env.datatype} {
  /* subgroup emulation must pass through wg_sw_subgroups */
  /* write my value to workgroup memory */
  wg_sw_subgroups[sgid] = x;
  workgroupBarrier();
  return wg_sw_subgroups[source];
}`;
  }
  get subgroupBallot() {
    return /* wgsl */ `fn subgroupBallotWrapper(pred: bool, sgid: u32) -> vec4<u32> {
  /* this is simple but will have significant bank conflicts,
   * and that's probably not easily avoidable
   * we could pad but then we'd have to grow wg_sw_subgroups */
  /* hardwired to 32b shuffles, because output is in 32b words */
  /* should work if workgroup size % 32 != 0 */
  /* trying to not do any work for threads >= 128 */
  const bitsPerOutput = 32u;
  var acc: u32 = select(0u, 1u, pred) << (sgid & (bitsPerOutput - 1));
  if (sgid < 128) {
    wg_sw_subgroups[sgid] = acc;
  }
  workgroupBarrier();
  for (var i: u32 = 1; i < bitsPerOutput; i <<= 1) {
    /* get and integrate my neighbor */
    /* if we're out-of-bounds, just fetch my own value instead */
    if (sgid < 128) {
      var sourceID: u32 = select(sgid, sgid ^ i, (sgid ^ i) < ${this.env.workgroupSize});
      acc |= wg_sw_subgroups[sourceID];
    }
    workgroupBarrier();
  }
  if (sgid < 128) {
    wg_sw_subgroups[sgid] = acc;
  }
  workgroupBarrier();
  var out: vec4u = vec4u(0);
  /* uniformity analysis requires the next loads be uniform ones */
  out[0] = workgroupUniformLoad(&wg_sw_subgroups[0]);
  /** possible bank-conflict avoidance: instead of reading from [0,32,64,96],
   *  we could read from [3,34,65,96], but then we'd have to check for overflow
   *  on the first three reads, not sure that's a win
   */
  for (var i: u32 = 32; i < min(${this.env.workgroupSize}, 128); i += 32) {
    /* write out[i], i in [1,2,3], if the workgroup is big enough */
    out[i >> 5] = workgroupUniformLoad(&wg_sw_subgroups[i]);
  }
  return out;
}`;
  }
  get subgroupReduce() {
    return /* wgsl */ `
fn subgroupReduceWrapper(in: ${this.env.datatype}, sgid: u32) -> ${this.env.datatype} {
  wg_sw_subgroups[sgid] = in;
  var red: ${this.env.datatype} = in;
  for (var i: u32 = 1; i < ${this.env.workgroupSize}; i <<= 1) {
    workgroupBarrier();
    var neighbor: u32 = sgid ^ i;
    var neighborVal: ${this.env.datatype} =
      select(${this.env.binop.identity},
             wg_sw_subgroups[neighbor],
             neighbor < ${this.env.workgroupSize});
    red = binop(red, neighborVal);
    wg_sw_subgroups[sgid] = red;
  }
  return workgroupUniformLoad(&wg_sw_subgroups[0]);
}`;
  }
  get subgroupInclusiveOpScan() {
    return /* wgsl */ `
fn subgroupInclusiveOpScan(in: ${this.env.datatype}, sgid: u32, laneCount: u32) ->
  ${this.env.datatype} {
  /* emulate subgroupInclusiveScanOp with subgroupShuffleUp */
  /* for (int i = 1; i <= 16; i <<= 1) { // 16 = LANE_COUNT >> 1
   *   const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
   *   if (getLaneId() >= i) val += t;
   * }
   * return val;
   */
  var red: ${this.env.datatype} = in;
  var t: ${this.env.datatype};
  for (var i: u32 = 1; i < ${this.env.workgroupSize}; i <<= 1) {
    wg_sw_subgroups[sgid] = red;
    workgroupBarrier();
    var neighborIdx: u32 = sgid - i;
    if (neighborIdx >= 0) {
      t = wg_sw_subgroups[neighborIdx];
      red = binop(t, red);
    }
  }
  return red;
}`;
  }
}
