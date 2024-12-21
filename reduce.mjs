import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpMinU32, BinOpMaxU32 } from "./binop.mjs";
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

    // every reduce test sets the following
    this.inputBytes = this.buffers[1].size;
    this.outputBytes = this.buffers[0].size;
    this.bytesTransferred = this.inputBytes + this.outputBytes;

    /* initialize buffer structures for all reduces */
    /* SHOULD BE read-only for input, but that screws up calling kernel twice */
    // this.bufferDescription = { 0: "storage", 1: "read-only-storage" };
    this.bufferDescription = { 0: "storage", 1: "storage" };

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }
  validate = (buffersArg) => {
    const buffers = this.bindingsToTypedArrays(buffersArg);
    const memsrc = buffers["in"][0];
    const memdest = buffers["out"][0];
    /* "reduction" is a one-element array, initialized to identity */
    const reduction = new (datatypeToTypedArray(this.datatype))([
      this.binop.identity,
    ]);
    for (let i = 0; i < buffers["in"][0].length; i++) {
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

const ReduceParams = {
  workgroupSize: range(2, 8).map((i) => 2 ** i),
  workgroupCount: range(0, 20).map((i) => 2 ** i),
};

const ReduceAndBinOpParams = {
  workgroupSize: range(2, 8).map((i) => 2 ** i),
  workgroupCount: range(0, 20).map((i) => 2 ** i),
  binop: [BinOpAddU32, BinOpMinU32, BinOpMaxU32],
};

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

const ReduceWGSizeBinOpPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fy: { field: "binop" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export class AtomicGlobalU32Reduce extends BaseReduce {
  constructor(args) {
    this.datatype = "u32";
    super(args);
  }
  compute() {
    return [
      new InitializeMemoryBlock({
        buffer: this.outputs[0],
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      new Kernel(
        () => /* wgsl */ `
        /* output */
        @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
        /* input */
        @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

        ${this.binop.wgslfn}

        @compute @workgroup_size(${this.workgroupSize}) fn globalU32ReduceKernel(
          @builtin(global_invocation_id) id: vec3u,
          @builtin(num_workgroups) nwg: vec3u) {
            let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
            ${this.binop.wgslatomic}(&memDest, memSrc[i]);
        }`
      ),
    ];
  }
}

export const AtomicGlobalU32ReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "atomic 1 element per thread global-atomic u32 sum reduction",
  // datatype: "u32",
  trials: 100,
  params: ReduceParams,
  primitive: AtomicGlobalU32Reduce,
  primitiveConfig: {
    binop: BinOpAddU32,
    gputimestamps: true,
  },
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

export const AtomicGlobalU32ReduceBinOpsTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "atomic 1 element per thread global-atomic u32 sum reduction",
  // datatype: "u32",
  trials: 100,
  params: ReduceAndBinOpParams,
  primitive: AtomicGlobalU32Reduce,
  primitiveConfig: {
    gputimestamps: true,
  },
  plots: [ReduceWGSizePlot, ReduceWGCountPlot, ReduceWGSizeBinOpPlot],
});

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

    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;
    this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

    /* this sets all the rest of the necessary parameters */
    this.updateSettings();
  }
  updateSettings() {
    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.min(
      Math.ceil(this.buffers[1].size / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    this.numPartials = this.workgroupCount;
  }
  reductionKernelDefinition = ({ inBinding, outBinding }) => {
    /** this needs to be an arrow function so "this" is the Primitive
     *  that declares it
     */

    /** this definition could be inline when the kernel is specified,
     * but since we call it twice, we move it here
     */
    return /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(${outBinding}) var<storage, read_write> outBuffer: array<${this.datatype}>;
      /* input */
      /* ideally this is read, not read_write, but then we can't call the kernel twice with partials */
      @group(0) @binding(${inBinding}) var<storage, read_write> inBuffer: array<${this.datatype}>;

      var<workgroup> temp: array<${this.datatype}, 32>; // zero initialized

      ${this.binop.wgslfn}

      @compute @workgroup_size(${this.workgroupSize}) fn noAtomicPKReduceIntoPartials(
        @builtin(global_invocation_id) id: vec3u /* 3D thread id in compute shader grid */,
        @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
        @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
        @builtin(local_invocation_index) lid: u32 /* 1D thread index within workgroup */,
        @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
        @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */) {
          /* TODO: fix 'assume id.y == 0 always' */
          var acc: ${this.datatype} = ${this.binop.identity};
          var numSubgroups = ${this.workgroupSize} / sgsz;
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
    return [
      new AllocateBuffer({ label: "partials", size: this.numPartials * 4 }),
      new InitializeMemoryBlock({
        buffer: "partials",
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      new InitializeMemoryBlock({
        buffer: this.buffers[0],
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      /* first kernel: per-workgroup persistent-kernel reduce */
      new Kernel({
        kernel: this.reductionKernelDefinition,
        kernelArgs: { inBinding: 1, outBinding: 2 },
        label: "noAtomicPKReduce workgroup reduce -> partials",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [this.workgroupCount];
        },
      }),
      /* second kernel: reduce partials into final output */
      new Kernel({
        kernel: this.reductionKernelDefinition,
        kernelArgs: { inBinding: 2, outBinding: 0 },
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

const PKReduceParamsSingleton = {
  inputSize: [2 ** 20],
  maxGSLWorkgroupCount: [2 ** 5],
  workgroupSize: [2 ** 6],
};

export const NoAtomicPKReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "no-atomic persistent-kernel u32 sum reduction",
  trials: 100,
  params: PKReduceParamsSingleton,
  uniqueRuns: ["inputSize", "workgroupCount", "workgroupSize"],
  primitive: NoAtomicPKReduce,
  primitiveConfig: {
    datatype: "u32",
    binop: BinOpAddU32,
    gputimestamps: true,
  },
  plots: [
    { ...ReduceWGSizePlot, ...{ fy: { field: "workgroupCount" } } },
    { ...ReduceWGCountPlot, ...{ fy: { field: "workgroupSize" } } },
  ],
});

class AtomicGlobalU32SGReduce extends BaseReduce {
  constructor(args) {
    this.datatype = "u32";
    super(args);
    this.compute = [
      new InitializeMemoryBlock({
        buffer: this.outputs[0],
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      new Kernel(
        () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

      @compute @workgroup_size(${this.workgroupSize}) fn globalU32SGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          let sgSum = subgroupAdd(memSrc[i]);
          if (sgid == 0) {
            atomicAdd(&memDest, sgSum);
          }
      }`
      ),
    ];
  }
}

export const AtomicGlobalU32SGReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testsuite: "atomic 1 element per thread per-subgroup u32 sum reduction",
  datatype: "u32",
  params: ReduceParams,
  primitive: AtomicGlobalU32Reduce,
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

class AtomicGlobalU32WGReduce extends BaseReduce {
  constructor(args) {
    this.datatype = "u32";
    super(args);
    this.kernel = () => /* wgsl */ `
      enable subgroups;

      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<u32>;
      var<workgroup> wgAcc: atomic<u32>; // auto-initialized to 0

      @compute @workgroup_size(${this.workgroupSize}) fn globalU32WGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          atomicAdd(&wgAcc, memSrc[i]);
          workgroupBarrier();
          if (lid == 0) {
            let wgSum = atomicLoad(&wgAcc);
            atomicAdd(&memDest, wgSum);
          }
      }`;
  }
}
export const AtomicGlobalU32WGReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testsuite: "Atomic per-workgroup u32 sum reduction",
  datatype: "u32",
  params: ReduceParams,
  primitive: AtomicGlobalU32WGReduce,
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

class AtomicGlobalF32WGReduce extends BaseReduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
    this.datatype = "f32";
    super(args);
    this.kernel = () => /* wgsl */ `
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;
      var<workgroup> wgAcc: atomic<u32>; // auto-initialized to 0

      alias u32WGCell = ptr<workgroup, atomic<u32>>;
      alias u32GlobalCell = ptr<storage, atomic<u32>, read_write>;

      // Adds f32 'value' to the value in 'sumCell', atomically.
      // Perform atomic compare-exchange with u32 type, and bitcast in and out of f32.
      fn atomicAddWGF32(sumCell: u32WGCell, value: f32) -> f32 {
        // Initializing to 0 forces second iteration in almost all cases.
        var old = 0u; //  alternately, atomicLoad(sumCell);
        loop {
          let new_value = value + bitcast<f32>(old);
          let exchange_result = atomicCompareExchangeWeak(sumCell, old, bitcast<u32>(new_value));
          if exchange_result.exchanged {
            return new_value;
          }
          old = exchange_result.old_value;
        }
      }

      fn atomicAddGlobalF32(sumCell: u32GlobalCell, value: f32) -> f32 {
        // Initializing to 0 forces second iteration in almost all cases.
        var old = 0u; //  alternately, atomicLoad(sumCell);
        loop {
          let new_value = value + bitcast<f32>(old);
          let exchange_result = atomicCompareExchangeWeak(sumCell, old, bitcast<u32>(new_value));
          if exchange_result.exchanged {
            return new_value;
          }
          old = exchange_result.old_value;
        }
      }

      @compute @workgroup_size(${this.workgroupSize}) fn globalF32WGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          atomicAddWGF32(&wgAcc, memSrc[i]);
          workgroupBarrier();
          if (lid == 0) {
            let wgSum = bitcast<f32>(atomicLoad(&wgAcc));
            atomicAddGlobalF32(&memDest, wgSum);
          }
      }`;
  }
}

export const AtomicGlobalF32WGReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testsuite: "Atomic per-workgroup f32 sum reduction",
  datatype: "f32",
  params: ReduceParams,
  primitive: AtomicGlobalF32WGReduce,
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

class AtomicGlobalNonAtomicWGF32Reduce extends BaseReduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
    this.datatype = "f32";
    super(args);
    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      var<workgroup> temp: array<f32, 32>; // zero initialized
                                           // 32 to ensure all warps are covered

      alias u32GlobalCell = ptr<storage, atomic<u32>, read_write>;

      fn atomicAddGlobalF32(sumCell: u32GlobalCell, value: f32) -> f32 {
        // Initializing to 0 forces second iteration in almost all cases.
        var old = 0u; //  alternately, atomicLoad(sumCell);
        loop {
          let new_value = value + bitcast<f32>(old);
          let exchange_result = atomicCompareExchangeWeak(sumCell, old, bitcast<u32>(new_value));
          if exchange_result.exchanged {
            return new_value;
          }
          old = exchange_result.old_value;
        }
      }

      @compute @workgroup_size(${this.workgroupSize}) fn globalF32WGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          // first, sum up within a workgroup
          let sgsum: f32 = subgroupAdd(memSrc[i]);
          if (subgroupElect()) {
            temp[lid / sgsz] = sgsum;
          }
          workgroupBarrier();
          var sgsum32: f32 = 0;
          if (lid < sgsz) {
            sgsum32 = temp[lid];
          }
          let wgSum: f32 = subgroupAdd(sgsum32);
          workgroupBarrier();
          if (lid == 0) {
            atomicAddGlobalF32(&memDest, wgSum);
          }
      }`;
  }
}

export const AtomicGlobalNonAtomicWGF32ReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testsuite: "Non-atomic per-workgroup, atomic-f32 sum reduction",
  datatype: "f32",
  params: ReduceParams,
  primitive: AtomicGlobalNonAtomicWGF32Reduce,
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

class AtomicGlobalPrimedNonAtomicWGF32Reduce extends BaseReduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
    this.datatype = "f32";
    super(args);
    this.testsuite =
      "Non-atomic per-workgroup, atomic-f32 sum reduction, with primed atomic";

    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      var<workgroup> temp: array<f32, 32>; // zero initialized
                                           // 32 to ensure all warps are covered

      alias u32GlobalCell = ptr<storage, atomic<u32>, read_write>;

      fn atomicAddGlobalF32(sumCell: u32GlobalCell, value: f32) -> f32 {
        // Initializing to 0 forces second iteration in almost all cases.
        var old = atomicLoad(sumCell);
        loop {
          let new_value = value + bitcast<f32>(old);
          let exchange_result = atomicCompareExchangeWeak(sumCell, old, bitcast<u32>(new_value));
          if exchange_result.exchanged {
            return new_value;
          }
          old = exchange_result.old_value;
        }
      }

      @compute @workgroup_size(${this.workgroupSize}) fn globalF32WGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          // first, sum up within a workgroup
          let sgsum: f32 = subgroupAdd(memSrc[i]);
          if (subgroupElect()) {
            temp[lid / sgsz] = sgsum;
          }
          workgroupBarrier();
          var sgsum32: f32 = 0;
          if (lid < sgsz) {
            sgsum32 = temp[lid];
          }
          let wgSum: f32 = subgroupAdd(sgsum32);
          workgroupBarrier();
          if (lid == 0) {
            atomicAddGlobalF32(&memDest, wgSum);
          }
      }`;
  }
}

export const AtomicGlobalPrimedNonAtomicWGF32ReduceTestSuite =
  new BaseTestSuite({
    category: "reduce",
    testsuite: "Non-atomic per-workgroup, primed-atomic-f32 sum reduction",
    datatype: "f32",
    params: ReduceParams,
    primitive: AtomicGlobalPrimedNonAtomicWGF32Reduce,
    plots: [ReduceWGSizePlot, ReduceWGCountPlot],
  });
