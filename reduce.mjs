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
    this.trials = 100;
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
