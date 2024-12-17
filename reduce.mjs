import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpMinU32, BinOpMaxU32 } from "./binop.mjs";

// exports: TestSuites, Primitives

class BaseReduce extends BasePrimitive {
  constructor(args) {
    super(args);
    // every reduce test sets the following
    this.memdestSize = 1;
    if (!this.memsrcSize) {
      this.memsrcSize = this.workgroupSize * this.workgroupCount;
    }
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    /* do I really need these next three? */
    this.numInputBuffers = 1;
    this.numOutputBuffers = 1;
    this.numUniformBuffers = 0;
    /* delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }
}

class BaseU32Reduce extends BaseReduce {
  constructor(args) {
    super(args);
    this.datatype = "u32";
  }
  /** expected format of "buffersIn":
   * { "in": [TypedArray], "out": [TypedArray] }
   * if it's not in that format, bindingsToTypedArrays should convert it
   * bindingsToTypedArrays expects an argument of
   *     { "in": [somethingBufferish], "out": [somethingBufferish] }
   *     or
   *     { "in": TypedArray, "out": TypedArray }
   */
  validate = (buffersIn) => {
    const buffers = this.bindingsToTypedArrays(buffersIn);
    const memsrc = buffers["in"][0];
    const memdest = buffers["out"][0];
    /* "reduction" is a one-element array, initialized to identity */
    const reduction = new Uint32Array([this.binop.identity]);
    for (
      let i = 0;
      i < buffers["in"][0].length;
      reduction[0] = this.binop.op(reduction[0], memsrc[i++])
    ) {
      /* empty on purpose */
    }
    console.log(
      "Should validate to",
      reduction[0],
      this.binop.constructor.name,
      "init was",
      this.binop.identity,
      "length is",
      memsrc.length,
      "memsrc[",
      memsrc.length - 1,
      "] is",
      memsrc[memsrc.length - 1]
    );
    if (memdest[0] != reduction[0]) {
      return `Element ${0}: expected ${reduction[0]}, instead saw ${
        memdest[0]
      }.`;
    } else {
      return "";
    }
  };
}

class BaseF32Reduce extends BaseReduce {
  constructor(args) {
    super(args);
    this.datatype = "f32";
    this.randomizeInput = true;
  }
  validate = (memsrc, memdest) => {
    const sum = new Float32Array([0]); // [0.0]; //
    for (let i = 0; i < memsrc.length; sum[0] += memsrc[i++]);
    if (Math.abs(sum[0] - memdest[0]) / sum[0] > 0.001) {
      return `Element ${0}: expected ${sum[0]}, instead saw ${memdest[0]}.`;
    } else {
      return "";
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
  x: { field: "memsrcSize", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

const ReduceWGCountPlot = {
  x: { field: "memsrcSize", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupCount" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup count",
};

/* needs to be a function if we do string interpolation */
function ReduceWGCountFnPlot() {
  return {
    x: { field: "memsrcSize", label: "Input array size (B)" },
    /* y.field is just showing off that we can have an embedded function */
    /* { field: "bandwidth", ...} would do the same */
    y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "workgroupCount" },
    text_br: (d) => `${d.gpuinfo.description}`,
    caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
  };
}

const ReduceWGSizeBinOpPlot = {
  x: { field: "memsrcSize", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fy: { field: "binop" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export class AtomicGlobalU32Reduce extends BaseU32Reduce {
  constructor(args) {
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

export class NoAtomicPKReduce extends BaseU32Reduce {
  /* persistent kernel, no atomics */
  constructor(args) {
    super(args);
    this.workgroupSize = this.workgroupSize ?? 256;
    this.workgroupCount = Math.min(
      Math.ceil(this.memsrcSize / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    /* args should contain binop, datatype */
  }
  compute() {
    return [
      new AllocateBuffer({ label: "partials", size: this.workgroupCount * 4 }),
      new InitializeMemoryBlock({
        buffer: "partials",
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      new InitializeMemoryBlock({
        buffer: this.outputs[0],
        value: this.binop.identity,
        datatype: this.datatype,
      }),
      /* first kernel: per-workgroup persistent-kernel reduce */
      new Kernel({
        kernel: () => /* wgsl */ `
        enable subgroups;
        /* output */
        @group(0) @binding(2) var<storage, read_write> partials: array<${this.datatype}>;
        /* input */
        @group(0) @binding(1) var<storage, read_write> memSrc: array<${this.datatype}>;////

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
              i < arrayLength(&memSrc);
              i += nwg.x * ${this.workgroupSize}) {
                /* on every iteration, grab wkgpsz items */
                acc = binop(acc, memSrc[i]);
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
              partials[wgid.x] = acc;
            }
        }`,
        label: "noAtomicPKReduceIntoPartials",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [this.workgroupCount];
        },
      }),
      /* second kernel: reduce partials into final output */
      new Kernel({
        kernel: () => /* wgsl */ `
        /* output */
        @group(0) @binding(0) var<storage, read_write> memDest: atomic<${this.datatype}>;
        /* input */
        @group(0) @binding(2) var<storage, read_write> partials: array<${this.datatype}>;////

        ${this.binop.wgslfn}

        @compute @workgroup_size(${this.workgroupSize}) fn noAtomicPKReduceIntoPartials(
          @builtin(global_invocation_id) id: vec3u /* 3D thread id in compute shader grid */,
          @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
          @builtin(local_invocation_index) lid: u32 /* thread index in workgroup */,
        ) {
            let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
            if (i < arrayLength(&partials)) {
              ${this.binop.wgslatomic}(&memDest, partials[i]);
            }
        }`,
        label: "noAtomicPKReduceIntoPartials",
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
  memsrcSize: range(8, 26).map((i) => 2 ** i) /* slowest */,
  maxGSLWorkgroupCount: range(2, 8).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

export const NoAtomicPKReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "no-atomic persistent-kernel u32 sum reduction",
  trials: 1,
  params: PKReduceParams,
  uniqueRuns: ["memsrcSize", "workgroupCount", "workgroupSize"],
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

class AtomicGlobalU32SGReduce extends BaseU32Reduce {
  constructor(args) {
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

class AtomicGlobalU32WGReduce extends BaseU32Reduce {
  constructor(args) {
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

class AtomicGlobalF32WGReduce extends BaseF32Reduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
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

class AtomicGlobalNonAtomicWGF32Reduce extends BaseF32Reduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
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

class AtomicGlobalPrimedNonAtomicWGF32Reduce extends BaseF32Reduce {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(args) {
    super(args);
    this.testsuite =
      "Non-atomic per-workgroup, atomic-f32 sum reduction, with primed atomic";
    this.datatype = "f32";

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
