import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
class BaseReduceTest extends BaseTest {
  category = "reduce";
  trials = 10;
}

const AtomicGlobalU32ReduceTestParams = {
  workgroupSize: range(2, 7).map((i) => 2 ** i),
  workgroupCount: range(5, 15).map((i) => 2 ** i),
};

class AtomicGlobalU32ReduceTestClass extends BaseReduceTest {
  constructor(params) {
    super(params);
    this.testname = "atomic u32 sum reduction, should be very slow";
    this.datatype = "u32";
    this.memsrcSize = this.workgroupSize * this.workgroupCount;
    this.memdestSize = 1;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: atomic<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

      @compute @workgroup_size(${this.workgroupSize}) fn reducePerWGKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          atomicAdd(&memDest, memSrc[i]);
      }`;
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

export const AtomicGlobalU32ReduceTestSuite = {
  class: AtomicGlobalU32ReduceTestClass,
  params: AtomicGlobalU32ReduceTestParams,
};
