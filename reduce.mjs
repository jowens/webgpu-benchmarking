import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
class BaseReduceTest extends BaseTest {
  constructor(params) {
    super(params);
    this.category = "reduce";
    this.datatype = "u32";
    this.memsrcSize = this.workgroupSize * this.workgroupCount;
    this.memdestSize = 1;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.trials = 2;
  }
  validate = (memsrc, memdest) => {
    const sum = new Uint32Array([0]);
    for (let i = 0; i < memsrc.length; sum[0] += memsrc[i++]);
    if (memdest[0] != sum[0]) {
      return `Element ${0}: expected ${sum[0]}, instead saw ${memdest[0]}.`;
    } else {
      return "";
    }
  };
  static plots = [
    {
      x: { field: "memsrcSize", label: "Input array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { field: "workgroupSize" },
      caption:
        "Atomic global reduction, 1 u32 per thread (lines are workgroup size)",
    },
    {
      x: { field: "memsrcSize", label: "Input array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { field: "workgroupCount" },
      caption:
        "Atomic global reduction, 1 u32 per thread (lines are workgroup count)",
    },
  ];
}

const AtomicGlobalU32ReduceTestParams = {
  workgroupSize: range(2, 8).map((i) => 2 ** i),
  workgroupCount: range(5, 20).map((i) => 2 ** i),
};

class AtomicGlobalU32ReduceTestClass extends BaseReduceTest {
  constructor(params) {
    super(params);
    this.testname = "Atomic per-element u32 sum reduction";

    this.kernel = () => /* wgsl */ `
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

      @compute @workgroup_size(${this.workgroupSize}) fn globalU32ReduceKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          atomicAdd(&memDest, memSrc[i]);
      }`;
  }
}

export const AtomicGlobalU32ReduceTestSuite = {
  class: AtomicGlobalU32ReduceTestClass,
  params: AtomicGlobalU32ReduceTestParams,
};

class AtomicGlobalU32SGReduceTestClass extends BaseReduceTest {
  constructor(params) {
    super(params);
    this.testname = "Atomic per-subgroup u32 sum reduction";

    this.kernel = () => /* wgsl */ `
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
      }`;
  }
}

export const AtomicGlobalU32SGReduceTestSuite = {
  class: AtomicGlobalU32SGReduceTestClass,
  params: AtomicGlobalU32ReduceTestParams,
};

class AtomicGlobalU32WGReduceTestClass extends BaseReduceTest {
  constructor(params) {
    super(params);
    this.testname = "Atomic per-workgroup u32 sum reduction";

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

export const AtomicGlobalU32WGReduceTestSuite = {
  class: AtomicGlobalU32WGReduceTestClass,
  params: AtomicGlobalU32ReduceTestParams,
};

class AtomicGlobalF32WGReduceTestClass extends BaseReduceTest {
  // https://github.com/gpuweb/gpuweb/issues/4894
  constructor(params) {
    super(params);
    this.testname = "Atomic per-workgroup f32 sum reduction";
    this.datatype = "f32";

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

export const AtomicGlobalF32WGReduceTestSuite = {
  class: AtomicGlobalF32WGReduceTestClass,
  params: AtomicGlobalU32ReduceTestParams,
};
