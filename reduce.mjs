import { range } from "./util.mjs";
import { BasePrimitive, Kernel, InitializeMemoryBlock } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpMinU32 } from "./binop.mjs";

// export both test suites and useful primitives

class BaseReduce extends BasePrimitive {
  constructor(args) {
    super(args);
    // every reduce test sets the following
    this.memsrcSize = this.workgroupSize * this.workgroupCount;
    this.memdestSize = 1;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.trials = 100;
    this.binop = args.binop;
    this.numInputBuffers = 1;
    this.numOutputBuffers = 1;
    this.numUniformBuffers = 0;
  }
  getDispatchGeometry() {
    // todo this is too simple
    let dispatchGeometry = [this.workgroupCount, 1];
    while (
      dispatchGeometry[0] > this.device.limits.maxComputeWorkgroupsPerDimension
    ) {
      /* too big */
      dispatchGeometry[0] = Math.ceil(dispatchGeometry[0] / 2);
      dispatchGeometry[1] *= 2;
    }
    return dispatchGeometry;
  }
}

class BaseU32Reduce extends BaseReduce {
  constructor(args) {
    super(args);
    this.datatype = "u32";
  }
  validate = (memsrc, memdest) => {
    /* "reduction" is a one-element array, initialized to identity */
    const reduction = new Uint32Array([this.binop.identity]);
    for (
      let i = 0;
      i < memsrc.length;
      reduction[0] = this.binop.op(reduction[0], memsrc[i++])
    ) {}
    console.log("Should validate to", reduction[0]);
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

const ReduceWGSizePlot = {
  x: { field: "memsrcSize", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

/* needs to be a function if we do string interpolation */
function ReduceWGCountPlot() {
  return {
    x: { field: "memsrcSize", label: "Input array size (B)" },
    /* y.field is just showing off that we can have an embedded function */
    /* { field: "bandwidth", ...} would do the same */
    y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "workgroupCount" },
    text_br: (d) => `${d.gpuinfo.description}`,
    caption: `${this.category} | ${this.testsuite} | ${this.datatype} | Lines are workgroup count`,
  };
}

export class AtomicGlobalU32Reduce extends BaseU32Reduce {
  constructor(args) {
    super(args);
  }
  compute() {
    return [
      new InitializeMemoryBlock(
        this.outputs[0],
        this.binop.identity,
        this.datatype
      ),
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
  testsuite: "atomic 1 element per thread global-atomic u32 sum reduction",
  datatype: "u32",
  params: ReduceParams,
  primitive: AtomicGlobalU32Reduce,
  // inputs: [{}],
  // outputs: [{}],
  primitiveConfig: {
    binop: BinOpAddU32,
    gputimestamps: true,
    /* set inputs and outputs later */
    // inputs: memsrcuBuffer,
    // outputs: memdestBuffer,
  },
  plots: [ReduceWGSizePlot, ReduceWGCountPlot],
});

class AtomicGlobalU32SGReduce extends BaseU32Reduce {
  constructor(args) {
    super(args);
    this.compute = [
      new InitializeMemoryBlock(
        this.outputs[0],
        this.binop.identity,
        this.datatype
      ),
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
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<u32>;
      var<workgroup> wgAcc: atomic<u32>; // auto-initialized to 0

      @compute @workgroup_size(${this.workgroupSize}) fn globalU32WGReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
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
