import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  // InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpMaxU32 } from "./binop.mjs";
import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";

// exports: TestSuites, Primitives

class BaseReduce extends BasePrimitive {
  constructor(args) {
    super(args);

    for (const required of ["datatype"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
    }

    this.knownBuffers = ["inputBuffer", "outputBuffer"];

    for (const knownBuffer of this.knownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
        delete this[knownBuffer]; // let's make sure it's in one place only
      }
    }

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }

  get bytesTransferred() {
    return (
      this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
    );
  }

  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
    const memdest =
      args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    /* "reduction" is a one-element array, initialized to identity */
    const reduction = new (datatypeToTypedArray(this.datatype))([
      this.binop.identity,
    ]);
    for (let i = 0; i < memsrc.length; i++) {
      reduction[0] = this.binop.op(reduction[0], memsrc[i]);
    }
    console.log(
      this.label,
      "should validate to",
      reduction[0],
      "and actually validates to",
      memdest[0],
      "\n",
      this.binop.constructor.name,
      this.binop.datatype,
      "identity is",
      this.binop.identity,
      "length is",
      memsrc.length,
      "memsrc[",
      memsrc.length - 1,
      "] is",
      memsrc[memsrc.length - 1]
    );
    function validates(args) {
      switch (args.datatype) {
        case "f32":
          return Math.abs((args.cpu - args.gpu) / args.cpu) < 0.001;
        default:
          return args.cpu == args.gpu;
      }
    }
    if (
      validates({ cpu: reduction[0], gpu: memdest[0], datatype: this.datatype })
    ) {
      return "";
    } else {
      return `Element ${0}: expected ${reduction[0]}, instead saw ${
        memdest[0]
      }.`;
    }
  };
}

const ReduceWGSizePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupSize" },
  fx: { field: "timing" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

const ReduceWGCountPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupCount" },
  fx: { field: "timing" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup count",
};

/* needs to be a function if we do string interpolation */
// eslint-disable-next-line no-unused-vars
function ReduceWGCountFnPlot() {
  return {
    x: { field: "inputBytes", label: "Input array size (B)" },
    /* y.field is just showing off that we can have an embedded function */
    /* { field: "bandwidth", ...} would do the same */
    y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "workgroupCount" },
    text_br: (d) => `${d.gpuinfo.description}`,
    caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
  };
}

// eslint-disable-next-line no-unused-vars
const ReduceWGSizeBinOpPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fy: { field: "binop" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export class NoAtomicPKReduce extends BaseReduce {
  /* persistent kernel, no atomics */
  constructor(args) {
    super(args);
    /* Check for required parameters for this kernel first */
    for (const required of ["binop", "datatype"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
    }
  }

  finalizeRuntimeParameters() {
    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;
    this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.min(
      Math.ceil(this.getBuffer("inputBuffer").length / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    this.partialsLength = this.workgroupCount;
  }
  reductionKernel = () => {
    /** this needs to be an arrow function so "this" is the Primitive
     *  that declares it
     */

    /** this definition could be inline when the kernel is specified,
     * but since we call it twice, we move it here
     */
    return /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> outputBuffer: array<${this.datatype}>;
      /* input */
      @group(0) @binding(1) var<storage, read> inputBuffer: array<${this.datatype}>;

      /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
      var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

      ${this.fnDeclarations.commonDefinitions}
      ${this.fnDeclarations.roundUpDivU32}
      ${this.binop.wgslfn}
      ${this.fnDeclarations.workgroupReduce}

      @compute @workgroup_size(${this.workgroupSize})
      fn noAtomicPKReduceIntoPartials(builtins : Builtins) {
        var reduction: ${this.datatype} = workgroupReduce(&inputBuffer, &wgTemp, builtins);
        if (builtins.lid == 0) {
          outputBuffer[builtins.wgid.x] = reduction;
        }
      }`;
  };
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "partials",
        size: this.partialsLength * datatypeToBytes(this.datatype),
      }),
      /* first kernel: per-workgroup persistent-kernel reduce */
      new Kernel({
        kernel: this.reductionKernel,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["partials", "inputBuffer"]],
        label: "noAtomicPKReduce workgroup reduce -> partials",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [this.workgroupCount];
        },
      }),
      /* second kernel: reduce partials into final output */
      new Kernel({
        kernel: this.reductionKernel,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["outputBuffer", "partials"]],
        label: "noAtomicPKReduce partials->final",
        getDispatchGeometry: () => {
          /* This reduce is defined to do its final step with one workgroup */
          return [1];
        },
        enable: true,
        logKernelCodeToConsole: false,
      }),
    ];
  }
}

const PKReduceParams = {
  inputLength: range(8, 26).map((i) => 2 ** i) /* slowest */,
  maxGSLWorkgroupCount: range(2, 8).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

// eslint-disable-next-line no-unused-vars
const PKReduceParamsSingleton = {
  inputLength: [2 ** 20],
  maxGSLWorkgroupCount: [2 ** 5],
  workgroupSize: [2 ** 7],
};

export const NoAtomicPKReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "no-atomic persistent-kernel u32 sum reduction",
  trials: 10,
  params: PKReduceParams,
  uniqueRuns: ["inputLength", "workgroupCount", "workgroupSize"],
  primitive: NoAtomicPKReduce,
  primitiveArgs: {
    datatype: "u32",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
  plots: [
    { ...ReduceWGSizePlot, ...{ fy: { field: "workgroupCount" } } },
    { ...ReduceWGCountPlot, ...{ fy: { field: "workgroupSize" } } },
  ],
});

export class ReduceDLDF extends BaseReduce {
  /* includes decoupled lookback, decoupled fallback */
  constructor(args) {
    super(args);
    /* Check for required parameters for this kernel first */
    for (const required of ["binop", "datatype"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
    }
  }

  finalizeRuntimeParameters() {
    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;
    this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.min(
      Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    this.partialsLength = this.workgroupCount;
  }
  reductionKernelDefinition = () => {
    /** this needs to be an arrow function so "this" is the Primitive
     *  that declares it
     */

    /** this definition could be inline when the kernel is specified,
     * but since we call it twice, we move it here
     */
    return /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> outBuffer: array<${this.datatype}>;
      /* input */
      @group(0) @binding(1) var<storage, read> inBuffer: array<${this.datatype}>;

      /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
      var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

      ${this.binop.wgslfn}

      fn roundUpDivU32(a : u32, b : u32) -> u32 {
        return (a + b - 1) / b;
      }

      @compute @workgroup_size(${this.workgroupSize}) fn noAtomicPKReduceIntoPartials(
        @builtin(global_invocation_id) id: vec3u /* 3D thread id in compute shader grid */,
        @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
        @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
        @builtin(local_invocation_index) lid: u32 /* 1D thread index within workgroup */,
        @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
        @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */) {
          /* TODO: fix 'assume id.y == 0 always' */
          /* TODO: what if there are more threads than subgroup_size * subgroup_size? */
          var acc: ${this.datatype} = ${this.binop.identity};
          var numSubgroups = roundUpDivU32(${this.workgroupSize}, sgsz);
          for (var i = id.x;
            i < arrayLength(&inBuffer);
            i += nwg.x * ${this.workgroupSize}) {
              /* on every iteration, grab wkgpsz items */
              acc = binop(acc, inBuffer[i]);
          }
          /* acc contains a partial sum for every thread */
          workgroupBarrier();
          /* now we need to reduce acc within our workgroup */
          /* switch to local IDs only. write into wg memory */
          acc = ${this.binop.subgroupOp}(acc);
          var mySubgroupID = lid / sgsz;
          if (subgroupElect()) {
            /* I'm the first element in my subgroup */
            wgTemp[mySubgroupID] = acc;
          }
          workgroupBarrier(); /* completely populate wg memory */
          if (lid < sgsz) { /* only activate 0th subgroup */
            /* read sums of all other subgroups into acc, in parallel across the subgroup */
            /* acc is only valid for lid < numSubgroups, so ... */
            /* select(f, t, cond) */
            acc = select(${this.binop.identity}, wgTemp[lid], lid < numSubgroups);
          }
          /* acc is called here for everyone, but it only matters for thread 0 */
          acc = ${this.binop.subgroupOp}(acc);
          if (lid == 0) {
            outBuffer[wgid.x] = acc;
          }

          /** Now we're done with our own local reduction. Turn to the global phase.
           * We have an global array, size of number of workgroups, that can store per-workgroup:
           * - No useful information (flag: EMPTY)
           * - The workgroup's reduction (only) (flag: LOCAL_REDUCTION)
           * - The reduction of its workgroup + all previous workgroups (flag: GLOBAL_REDUCTION)
           * Each workgroup now activates only its thread 0 and does the following:
           *
           * partials[id] = { LOCAL_REDUCTION, local reduction }
           * lookback_id = id - 1
           * lookbackreduction = identity
           * loop {
           *   { flag, val } = partials[lookback_id]
           *   switch(flag) {
           *     GLOBAL_REDUCTION:
           *       partials[id] = { GLOBAL_REDUCTION, val + local_reduction }
           *       we're done
           *     LOCAL_REDUCTION:
           *       local_reduction += val
           *       lookback_id--
           *     EMPTY:
           *       nothing
           * }
           *
           * But then we have a problem: what if we spin forever? We have no
           *   forward progress guarantees; we might be waiting on a previous EMPTY
           *   entry that never moves forward
           *
           * First, we guard all of the following with a "done" flag that is initialized to FALSE
           * If this flag is true, then our global work is done, i.e.,
           *   our result in the global array is flagged with GLOBAL_REDUCTION
           * We loop until that flag is set to true
           *
           * Phase 1. Optimistically hope previous workgroups complete, spinning a fixed
           *   number of times to check.
           *   All of this is done with thread 0 only
           * partials[id] = { LOCAL_REDUCTION, local reduction }
           * lookback_id = id - 1
           * lookbackreduction = identity
           * loop (!done) {
           *   loop (still spinning) { // this is different than above
           *     { flag, val } = partials[lookback_id]
           *     switch(flag) {
           *       GLOBAL_REDUCTION:
           *         partials[id] = { GLOBAL_REDUCTION, val + local_reduction }
           *         we're done, return
           *       LOCAL_REDUCTION:
           *         lookback_id--
           *         local_reduction += val
           *         spin_count = 0
           *       EMPTY
           *         spin_count++
           *     }
           *   }
           *   // fall out of spin loop, we've waited too long on an EMPTY
           *   broadcast lookback_id associated with empty to entire workgroup
           * }
           *
           * Phase 2. There's a prior block (lookback_id) that's EMPTY.
           *   - Set fallback_id = lookback_id
           *   - Using the whole workgroup, compute f_red = reduction of block fallback_id.
           *   - Set reduction[fallback_id] to { LOCAL_REDUCTION, f_red }.
           * This is complicated, though, because reduction[fallback_id] might have been
           *   subsequently updated, and we'd (possibly) like to know what was there
           *   rather than blindly write over it.
           * So here's the cases:
           *   - reduction[fallback_id] is EMPTY. Want to write
           *
           *
           * Phase 3. Fallback to computing reduction of block lookback_id,
           *   which we now call fallback_id. fallback_id is constant in this block.
           *   Goal is to reach backwards and update block fallback_id from EMPTY to LOCAL REDUCTION
           * As an entire block, load and reduce block # fallback_id
           * For thread 0 only:
           *   fetch partials[fallback_id]
           *   if it is { EMPTY, 0 }:
           *     nothing has been stored there; store { LOCAL_REDUCTION, f_red }.
           *   if it is { LOCAL_REDUCTION, val }:
           *     because f_red == val, storing has no effect; can store or not
           *   if it is { GLOBAL_REDUCTION, val }:
           *     want to keep that value and NOT store
           */
        }`;
  };
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "partials",
        size: this.partialsLength * datatypeToBytes(this.datatype),
      }),
      /* first kernel: per-workgroup persistent-kernel reduce */
      new Kernel({
        kernel: this.reductionKernelDefinition,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["partials", "inputBuffer"]],
        label: "noAtomicPKReduce workgroup reduce -> partials",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [this.workgroupCount];
        },
      }),
      /* second kernel: reduce partials into final output */
      new Kernel({
        kernel: this.reductionKernelDefinition,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["outputBuffer", "partials"]],
        label: "noAtomicPKReduce partials->final",
        getDispatchGeometry: () => {
          /* This reduce is defined to do its final step with one workgroup */
          return [1];
        },
        enable: true,
        logKernelCodeToConsole: false,
      }),
    ];
  }
}
