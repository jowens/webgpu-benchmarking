import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  // InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpMaxU32 } from "./binop.mjs";
import { datatypeToTypedArray } from "./util.mjs";

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

  bytesTransferred() {
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
    function validates(cpu, gpu, datatype) {
      switch (datatype) {
        case "f32":
          return Math.abs(cpu - gpu) / cpu < 0.001;
        default:
          return cpu == gpu;
      }
    }
    if (validates(reduction[0], memdest[0], this.datatype)) {
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
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

const ReduceWGCountPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupCount" },
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
      Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    this.numPartials = this.workgroupCount;
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

      var<workgroup> temp: array<${this.datatype}, 32>; // zero initialized

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
            temp[mySubgroupID] = acc;
          }
          workgroupBarrier(); /* completely populate wg memory */
          if (lid < sgsz) { /* only activate 0th subgroup */
            /* read sums of all other subgroups into acc, in parallel across the subgroup */
            /* acc is only valid for lid < numSubgroups, so ... */
            /* select(f, t, cond) */
            acc = select(${this.binop.identity}, temp[lid], lid < numSubgroups);
          }
          /* acc is called here for everyone, but it only matters for thread 0 */
          acc = ${this.binop.subgroupOp}(acc);
          if (lid == 0) {
            outBuffer[wgid.x] = acc;
          }
        }`;
  };
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "partials",
        size: this.numPartials * 4,
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
        debugPrintKernel: false,
      }),
    ];
  }
}

const PKReduceParams = {
  inputSize: range(8, 26).map((i) => 2 ** i) /* slowest */,
  maxGSLWorkgroupCount: range(2, 8).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

export const NoAtomicPKReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "no-atomic persistent-kernel u32 sum reduction",
  trials: 10,
  params: PKReduceParams,
  uniqueRuns: ["inputSize", "workgroupCount", "workgroupSize"],
  primitive: NoAtomicPKReduce,
  primitiveConfig: {
    datatype: "u32",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
  plots: [
    { ...ReduceWGSizePlot, ...{ fy: { field: "workgroupCount" } } },
    { ...ReduceWGCountPlot, ...{ fy: { field: "workgroupSize" } } },
  ],
});
