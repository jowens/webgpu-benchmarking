/**
 * wgslFunctions provides a set of prewritten functions for use in compute
 * kernels. The intent is that the same functions are available on both
 * subgroup-supporting and subgroup-nonsupporting hardware.
 *
 * Usage: The Primitive class automatically incorporates these functions
 * by instantiating an instance of the wgslFunctions or
 * wgslFunctionsWithoutSubgroupSupport class in its constructor:
 *
 * `this.fnDeclarations = new wgslFunctions(this);`
 *
 * If subgroups are not supported, instead:
 *
 * `this.fnDeclarations = new wgslFunctionsWithoutSubgroupSupport(this);`
 *
 * Then, within a kernel, the kernel programmer can include individual
 * functions, such as:
 *
 * `${this.fnDeclarations.vec4InclusiveScan}`
 *
 * which declares the function `vec4InclusiveScan`.
 *
 * More detailed usage:
 *
 * 1. At the top of the kernel, place the following:
 *    `${this.fnDeclarations.enableSubgroupsIfAppropriate}`
 *    WGSL requires declaring subgroups must be done at the top of the kernel.
 *    If subgroups are not enabled, this declaration will do nothing.
 * 2. Next, if the kernel is using any subgroup calls, place the following:
 *    `${this.fnDeclarations.subgroupEmulation}`
 *    If we're making subgroup calls and we don't have subgroup hardware,
 *    this sets up necessary declarations (workgroup memory and subgroup
 *    variables). If subgroups are supported, this declaration does nothing.
 * 3. If the primitive is parameterized by a datatype and a binary operation,
 *    it may be useful to define a monoid that combines the two. If "monoid"
 *    means nothing to you, instead consider: it may be useful to have a WGSL
 *    operation called "binop" that is a binary operation that combines two
 *    inputs of a specific datatype. (For instance, + on f32 values.)
 *    Declare that binop with:
 *    `${this.binop.wgslfn}`
 *    (This definition is in binop.mjs, not here.)
 * 4. In this file are a large number of functions to call, each of which
 *    outputs a WGSL function definition. These functions output WGSL code
 *    that may be parameterized by objects in the Primitive's `this`.
 *    Override any parameter by passing it in the input object to the
 *    function call.
 *    - Example: vec4InclusiveScan is parameterized by `datatype`. If
 *      Primitive's this.datatype is "f32", and we call
 *      `${this.fnDeclarations.vec4InclusiveScan()}` in our kernel,
 *      the signature of the declared function will be
 *      `fn vec4InclusiveScan(in: vec4<f32>) -> vec4<f32>`.
 *      But if we want to override it to u32 when we instantiate the
 *      function, we can instead call
 *      `${this.fnDeclarations.vec4InclusiveScan({datatype: "u32"})}`.
 *    The names of the parameters that are necessary for various function
 *    declarations are:
 *    - binop (the monoid containing a binary operator and datatype, see
 *      binop.mjs)
 *    - datatype (a string, one of {"f32", "u32", "i32"})
 *    - workgroupSize (a number)
 *
 * Notes for authors of functions:
 * - If a function doesn't require subgroups, put it in wgslFunctions
 * - If it does, write the subgroup version in wgslFunctions
 *   and the non-subgroup version in wgslFunctionsWithoutSubgroupSupport;
 *   the latter will override the former (this is done in the Primitive class)
 * Your functions can be parameterized by any element in the Primitive object
 *   and by any element in the argument to the function. The recommended usage
 *   places both of these in an `env` object, where any element of args
 *   overrides any element in the Primitive object, and then the WGSL function
 *   definition can use elements in this `env` object. (For instance,
 *   `env.datatype`.)
 *   - Implementation note. Originally these functions were getters and did
 *     not have to be called as functions. However, during the development
 *     of more complex functions, it became evident that some functions would
 *     definitely require arguments, so for consistency, all of them are
 *     defined as functions not getters.
 *
 * Convention for order of arguments to WGSL functions:
 * 1. builtins
 * 2. outputs, ordered from most global (storage) to most local (workgroup),
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 * 3. inputs, ordered from most global (storage) to most local (workgroup)
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 *
 * General philosophy is to design kernels around an expectation that
 * subgroups are present and optimize kernel organization strategies for
 * that case; a non-subgroup-enabled function that provides fallback must
 * be correct but may not be optimized.
 */

import { BinOpAdd } from "./binop.mjs";

export class wgslFunctions {
  constructor(args) {
    this.env = args;
  }
  commonDefinitions(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(local_invocation_id) lid: vec3u /* 3D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsNonuniform {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(local_invocation_id) lid: vec3u /* 3D thread index within workgroup */,
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsUniform {
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(subgroup_size) sgsz: u32 /* 32 on Apple GPUs */
  }`;
  }
  initializeSubgroupVars(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return "let sgsz: u32 = builtinsUniform.sgsz;\nlet sgid: u32 = builtinsNonuniform.sgid;";
  }
  enableSubgroupsIfAppropriate(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return "enable subgroups;";
  }
  subgroupEmulation(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return "";
  }
  roundUpDivU32(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `fn roundUpDivU32(a : u32, b : u32) -> u32 {
    return (a + b - 1) / b;
  }`;
  }
  computeLinearizedGridParameters(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    /* wgid is a linearized (1d) unique ID per wg;
     * gid is a linearized (1d) unique ID per thread */
    var wgid = builtins.wgid.z * builtins.nwg.y * builtins.nwg.x +
               builtins.wgid.y * builtins.nwg.x +
               builtins.wgid.x;
    var numThreadsPerWorkgroup: u32 = ${
      env.numThreadsPerWorkgroup ?? env.workgroupSize
    };
    var gid: u32 = wgid * numThreadsPerWorkgroup + builtins.lidx;
    var workgroupCount = builtins.nwg.z * builtins.nwg.y * builtins.nwg.x;
    var totalThreadCount = workgroupCount * numThreadsPerWorkgroup;`;
  }
  computeLinearizedGridParametersSplit(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* "split" meaning "split builtin uniforms vs. nonuniforms" */
    return /* wgsl */ `
    /* wgid is a linearized (1d) unique ID per wg;
     * gid is a linearized (1d) unique ID per thread */
    var wgid = builtinsUniform.wgid.z * builtinsUniform.nwg.y * builtinsUniform.nwg.x +
               builtinsUniform.wgid.y * builtinsUniform.nwg.x +
               builtinsUniform.wgid.x;
    var numThreadsPerWorkgroup: u32 = ${
      env.numThreadsPerWorkgroup ?? env.workgroupSize
    };
    var gid: u32 = wgid * numThreadsPerWorkgroup + builtinsNonuniform.lidx;
    var workgroupCount = builtinsUniform.nwg.z * builtinsUniform.nwg.y * builtinsUniform.nwg.x;
    var totalThreadCount = workgroupCount * numThreadsPerWorkgroup;`;
  }
  vec4InclusiveScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4InclusiveScan(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${env.datatype}> = in;
      out.y = binop(in.x,  in.y);
      out.z = binop(out.y, in.z);
      out.w = binop(out.z, in.w);
      return out;
    }`;
  }
  vec4ExclusiveScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4ExclusiveScan(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${env.datatype}>;
      out.x = ${env.binop.identity};
      out.y = in.x;
      out.z = binop(in.x,  in.y);
      out.w = binop(out.z, in.z);
      return out;
    }`;
  }
  vec4InclusiveToExclusive(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4InclusiveToExclusive(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      var out: vec4<${env.datatype}>;
      out.w = in.z;
      out.z = in.y;
      out.y = in.x;
      out.x = ${env.binop.identity};
      return out;
    }`;
  }
  vec4Reduce(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    // TODO: Don't special-case this. Worried about polyfilling dot with
    // int arguments, that it'll potentially do four multiplies
    if (env.binop instanceof BinOpAdd) {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${env.datatype}>) -> ${env.datatype} {
        return dot(in, vec4<${env.datatype}>(1, 1, 1, 1));
      }
      `;
    } else {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${env.datatype}>) -> ${env.datatype} {
        return binop(binop(binop(in.x, in.y), in.z), in.w);
      }`;
    }
  }
  vec4ScalarBinopV4(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* if binop is +, this still seems just as efficient, unless there's a vec4 +, I guess? */
    // TODO: "WGSL has mixed vector-scalar arithmetic operators, so it's probably best to use those if you can."
    return /* wgsl */ `
    fn vec4ScalarBinopV4(scalar: ${env.datatype}, vector: vec4<${env.datatype}>) ->
    vec4<${env.datatype}> {
      var out: vec4<${env.datatype}>;
      out.x = binop(scalar, vector.x);
      out.y = binop(scalar, vector.y);
      out.z = binop(scalar, vector.z);
      out.w = binop(scalar, vector.w);
      return out;
    }
    `;
  }
  subgroupZero(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return lidx < sgsz;
    }`;
  }
  subgroupShuffle(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupBallot(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupMax(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupInclusiveOpScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* helpful reference from Thomas Smith:
     *   https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingCUDA/Utils.cuh
     */
    /** Would prefer to not make sgsz/sgid an argument here, but we need it for
     *  the subgroup-hardware-capable, non-hardware-supported-scan-op case
     *  An alternative would be putting sgsz/sgid in workgroup memory. */
    if (env.binop.subgroupInclusiveScanOp) {
      /* use the builtin subgroupInclusiveScanOp */
      return /* wgsl */ `
      fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
        ${env.datatype} {
        return ${env.binop.subgroupInclusiveScanOp}(in);
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
      fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
        ${env.datatype} {
        var i: u32;
        var val = in;
        for (i = 1; i <= (sgsz >> 1); i <<= 1) {
          let t: ${env.datatype} = subgroupShuffleUp(val, i);
          val = binop(select(${env.binop.identity}, t, sgid >= i), val);
        }
        return val;
      }
    `;
    }
  }
  subgroupReduce(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* this will fail if subgroupReduceOp isn't defined; TODO is write it */
    return /* wgsl */ `
    fn subgroupReduce(in: ${env.datatype}) -> ${env.datatype} {
      return ${env.binop.subgroupReduceOp}(in);
    }
    `;
  }
  wgReduce(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env

    /** The normal case would be that we only need one function of each type
     * thus we can use the shortFnName for declaration and call, and can do
     * everything within a template string.
     * But if we need more flexibility (here, more than one reduce call in
     * a module), we should do it outside the template string. Until that is
     * necessary, this capability has not been used and is untested.
     *
     * Primitive-specific args are:
     * - wgTempIsArgument: if true, pass in a temp array for temporary use
     * - useLongFunctionName: use config-specific name, otherwise wgReduce
     * Default for all of these is "false".
     */
    const shortFnName = "wgReduce";
    /* every entry in params below needs to be a member of env */
    const params = ["binop", "datatype", "workgroupSize", "SUBGROUP_MIN_SIZE"];
    for (const necessary of params) {
      if (!(necessary in env)) {
        console.warn(`wgReduce: field '${necessary}' must be set in env`);
      }
    }
    const config = params.map((param) => env[param]).join("_");
    const wgTemp = env.wgTempIsArgument ? "wgTemp" : `wg_temp_${config}`;
    const declareAndUseLocalWgTemp = !env.wgTempIsArgument;
    const longFnName = `${shortFnName}_${config}`;
    const fnName = env?.useLongFunctionName ? longFnName : shortFnName;
    const fn = /* wgsl */ `
${
  declareAndUseLocalWgTemp
    ? `const TEMP_${longFnName}_MEM_SIZE = 2 * ${env.workgroupSize} / ${env.SUBGROUP_MIN_SIZE};
var<workgroup> ${wgTemp}: array<${env.datatype}, TEMP_${longFnName}_MEM_SIZE>;`
    : ""
}

fn ${fnName}(// in: ptr<storage, array<${env.datatype}>, read>,
             in: ${env.datatype},
             ${
               declareAndUseLocalWgTemp
                 ? ""
                 : `wgTemp: ptr<workgroup, array<${env.datatype}, MAX_PARTIALS_SIZE>>,`
             }
             builtinsUniform: BuiltinsUniform,
             builtinsNonuniform: BuiltinsNonuniform) -> ${env.datatype} {
  let lidx = builtinsNonuniform.lidx;
  let sgsz = builtinsUniform.sgsz;
  let sgid = builtinsNonuniform.sgid;
  let BLOCK_DIM: u32 = ${env.workgroupSize};
  let sid = lidx / sgsz;
  let lane_log = u32(countTrailingZeros(sgsz)); /* log_2(sgsz) */
  /* workgroup size / subgroup size; how many partial reductions in this tile? */
  let local_spine: u32 = BLOCK_DIM >> lane_log;
  let aligned_size_base = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
  /* fix for aligned_size_base == 1 (needed when subgroup_size == BLOCK_DIM) */
  let aligned_size = select(aligned_size_base, BLOCK_DIM, aligned_size_base == 1);

  let t_red = in;
  let s_red = ${env.binop.subgroupReduceOp}(t_red);
  if (sgid == 0u) {
    ${wgTemp}[sid] = s_red;
  }
  workgroupBarrier();
  var f_red: ${env.datatype} = ${env.binop.identity};

  var offset = 0u;
  var top_offset = 0u;
  let lane_pred = sgid == sgsz - 1u;
  if (sgsz > aligned_size) {
    /* don't enter the loop */
    f_red = ${wgTemp}[lidx + top_offset];
  } else {
    for (var j = sgsz; j <= aligned_size; j <<= lane_log) {
      let step = local_spine >> offset;
      let pred = lidx < step;
      f_red = ${env.binop.subgroupReduceOp}(
        select(${env.binop.identity},
        ${wgTemp}[lidx + top_offset],
        pred));
      if (pred && lane_pred) {
        ${wgTemp}[sid + step + top_offset] = f_red;
      }
      workgroupBarrier();
      top_offset += step;
      offset += lane_log;
    }
  }
  return f_red;
}`;
    return fn;
  }
  workgroupScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
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
      /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
      ${env.fnDeclarations.computeLinearizedGridParameters}
      var numSubgroups = roundUpDivU32(${env.workgroupSize}, builtins.sgsz);
      var i = gid;
      var in = select(${env.binop.identity}, input[i], i < arrayLength(input));
      workgroupBarrier();
      /* "in" now contains the block of data to scan, padded with the identity */
      /* (1) reduce "in" within our workgroup */
      /* switch to local IDs only. write into wg memory */
      var sgReduction = ${env.binop.subgroupReduceOp}(in);
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
      var spineScanInput = select(${env.binop.identity},
                                  wgTemp[builtins.lidx],
                                  builtins.lidx < numSubgroups);
      /* no matter what type of scan we have, we use exclusiveScan here */
      var spineScanOutput = ${env.binop.subgroupExclusiveScanOp}(spineScanInput);
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
  oneWorkgroupExclusiveScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
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
      inputoutput: ptr<storage, array<${env.datatype}>, read_write>,
    ) {
      var acc : ${env.datatype} = ${env.binop.identity};
      /* making this work under uniform control flow is tricky */
      /* big idea: convert any control dependence to data dependence (i) */
      var ibase : u32 = 0;
      var sg0 = builtinsNonuniform.lidx < builtinsUniform.sgsz;
      while (ibase < arrayLength(inputoutput)) {
        /* work still left to be done */
        var i = ibase + builtinsNonuniform.lidx;
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
  commonDefinitions(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    struct Builtins {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    }
    struct BuiltinsNonuniform {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    }
    struct BuiltinsUniform {
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    }`;
  }
  initializeSubgroupVars(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    sgid = builtinsNonuniform.lidx;`;
  }
  enableSubgroupsIfAppropriate(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return "/* don't enable subgroups */\n";
  }
  /* if this declaration works for you, put it at the top of your kernel file at module scope */
  subgroupEmulation(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `var<workgroup> wg_sw_subgroups: array<${env.datatype}, ${env.workgroupSize}>;
    const sgsz: u32 = ${env.workgroupSize};
    var<private> sgid: u32;`;
  }
  subgroupZero(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return true;
    }`;
  }
  subgroupShuffle(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupShuffle(x: u32, source: u32) -> u32 {
  /* subgroup emulation must pass through wg_sw_subgroups */
  /* write my value to workgroup memory */
  wg_sw_subgroups[sgid] = bitcast<${env.datatype}>(x);
  workgroupBarrier();
  var shuffled: u32 = bitcast<u32>(wg_sw_subgroups[source]);
  workgroupBarrier();
  return shuffled;
}`;
  }
  subgroupBallot(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupBallot(pred: bool) -> vec4<u32> {
  /* this is simple but will have significant bank conflicts,
   * and that's probably not easily avoidable
   * we could pad but then we'd have to grow wg_sw_subgroups */
  /* hardwired to 32b shuffles, because output is in 32b words */
  /* should work if workgroup size % 32 != 0 */
  /* trying to not do any work for threads >= 128 */
  /** note acc is always u32 but wg_sw_subgroups might be another datatype,
   * so we have to bitcast. Every write to wg_sw_subgroups must be cast
   * to its datatype; every read from it must be cast to u32. */
  const bitsPerOutput = 32u;
  var acc: u32 = select(0u, 1u, pred) << (sgid & (bitsPerOutput - 1));
  if (sgid < 128) {
    wg_sw_subgroups[sgid] = bitcast<${env.datatype}>(acc);
  }
  workgroupBarrier();
  for (var i: u32 = 1; i < bitsPerOutput; i <<= 1) {
    /* and integrate my neighbor, write it back */
    /* if we're out-of-bounds, just fetch my own value instead */
    if (sgid < 128) {
      var sourceID: u32 = select(sgid, sgid ^ i, (sgid ^ i) < ${env.workgroupSize});
      acc |= bitcast<u32>(wg_sw_subgroups[sourceID]);
    }
    workgroupBarrier();
    if (sgid < 128) {
      wg_sw_subgroups[sgid] = bitcast<${env.datatype}>(acc);
    }
  }
  workgroupBarrier();
  var out: vec4u = vec4u(0);
  /* uniformity analysis requires the next loads be uniform ones */
  out[0] = bitcast<u32>(workgroupUniformLoad(&wg_sw_subgroups[0]));
  /** possible bank-conflict avoidance: instead of reading from [0,32,64,96],
   *  we could read from [3,34,65,96], but then we'd have to check for overflow
   *  on the first three reads, not sure that's a win
   */
  for (var i: u32 = 32; i < min(${env.workgroupSize}, 128); i += 32) {
    /* write out[i], i in [1,2,3], if the workgroup is big enough */
    out[i / bitsPerOutput] = bitcast<u32>(workgroupUniformLoad(&wg_sw_subgroups[i]));
  }
  return out;
}`;
  }
  subgroupReduce(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `
fn subgroupReduce(in: ${env.datatype}) -> ${env.datatype} {
  wg_sw_subgroups[sgid] = in;
  var red: ${env.datatype} = in;
  for (var i: u32 = 1; i < ${env.workgroupSize}; i <<= 1) {
    workgroupBarrier();
    var neighbor: u32 = sgid ^ i;
    var neighborVal: ${env.datatype} =
      select(${env.binop.identity},
             wg_sw_subgroups[neighbor],
             neighbor < ${env.workgroupSize});
    red = binop(red, neighborVal);
    workgroupBarrier();
    wg_sw_subgroups[sgid] = red;
  }
  workgroupBarrier(); // possibly not necessary given the next line?
  return workgroupUniformLoad(&wg_sw_subgroups[0]);
}`;
  }
  subgroupBinopIsU32Add(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `fn binop(a : u32, b : u32) -> u32 {return a+b;}`;
  }
  subgroupAdd(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupAdd(in: ${env.datatype}) -> ${env.datatype} {
  return subgroupReduce(in);
}`;
  }
  subgroupInclusiveOpScan(args = {}) {
    const env = { ...this.env, ...args }; // properties in args overwrite this.env
    /* this is almost certainly faster if we double-buffered */
    return /* wgsl */ `
fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
  ${env.datatype} {
  /* sgsz is not used, see above */
  var red: ${env.datatype} = in;
  var t: ${env.datatype};
  /* TODO: Handle case where input size is not a multiple of workgroup size */
  for (var delta: u32 = 1; delta < ${env.workgroupSize}; delta <<= 1) {
    /** On pass 0, element i - 1 (if in range) is added into element i, in parallel.
     * On pass 1, element i - 2 is added into element i.
     * On pass 2, element i - 4 is added into element i, and so on. */
    /* delta == how many threads I'm reaching back */
    wg_sw_subgroups[sgid] = red;
    workgroupBarrier();
    var neighborIdx: i32 = i32(sgid) - i32(delta);
    if (neighborIdx >= 0) {
      t = wg_sw_subgroups[neighborIdx];
      red = binop(t, red);
    }
    workgroupBarrier();
  }
  return red;
}`;
  }
}
