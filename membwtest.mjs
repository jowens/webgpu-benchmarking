import { range } from "./util.mjs";
import { BasePrimitive } from "./primitive.mjs";
class BaseMembwTest extends BasePrimitive {
  constructor(params) {
    super(params); // writes parameters into this class
    this.category = "membw";
    this.trials = 10;
  }
  validate = (memsrc, memdest) => {
    for (let i = 0; i < memsrc.length; i++) {
      const expected = memsrc[i] + 1.0;
      if (expected != memdest[i]) {
        return `Element ${i}: expected ${expected}, instead saw ${memdest[i]}.`;
      } else {
        return "";
      }
    }
  };
}

const MembwSimpleTestParams = {
  workgroupSize: range(0, 7).map((i) => 2 ** i),
  memsrcSize: range(10, 25).map((i) => 2 ** i),
};

export class MembwSimpleTestClass extends BaseMembwTest {
  constructor(params) {
    super(params);
    this.testname = "fp32-per-thread";
    this.description =
      "Copies input array to output array. One thread is assigned per 32b input element.";
    this.kernel = () => /* wgsl */ `
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      @compute @workgroup_size(${this.workgroupSize}) fn memcpyKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(workgroup_id) wgid: vec3u) {
          let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
          memDest[i] = memSrc[i] + 1.0;
      }`;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.workgroupCount = this.memsrcSize / this.workgroupSize;
  }
  static plots = [
    {
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { field: "workgroupSize" },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: {
        field: "bandwidthCPU",
        label: "Achieved bandwidth (GB/s) [CPU measurement]",
      },
      stroke: { field: "workgroupSize" },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: "time", label: "GPU time (ns)" },
      y: { field: "cpuns", label: "CPU time (ns)" },
      stroke: { field: "workgroupSize" },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: { field: "cpugpuDelta", label: "CPU - GPU time (ns)" },
      stroke: { field: "workgroupSize" },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
  ];
}

/**
 * grid stride loop, now we don't assign a fixed number of elements per thread
 * background: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */

const MembwGSLTestParams = {
  workgroupSize: range(0, 7).map((i) => 2 ** i),
  memsrcSize: range(10, 25).map((i) => 2 ** i),
  workgroupCount: range(5, 10).map((i) => 2 ** i),
};
export class MembwGSLTestClass extends BaseMembwTest {
  constructor(params) {
    super(params);
    this.testname = "GSL fp32-per-thread";
    this.kernel = (param) => /* wgsl */ `
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      @compute @workgroup_size(${this.workgroupSize}) fn memcpyKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u, // == dispatch
        @builtin(workgroup_id) wgid: vec3u) {
          /* grid-stride loop: assume nwg.y == 1 */
          for (var i = id.x;
               i < arrayLength(&memSrc);
               i += nwg.x * ${this.workgroupSize}) {
            memDest[i] = memSrc[i] + 1.0;
          }
      }`;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.dispatchGeometry = [this.workgroupCount];
  }
  static plots = [
    {
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: "workgroupCount", label: "Workgroup Count" },
      stroke: { field: "workgroupSize", label: "Workgroup Size" },
      caption: "Memory bandwidth test GSL (lines are workgroup size)",
    },
    {
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: "workgroupSize", label: "Workgroup Size" },
      stroke: { field: "workgroupCount", label: "Workgroup Count" },
      caption:
        "Memory bandwidth test GSL (lines are workgroup size). Looks like max throughput doesn't occur until ~512 threads/workgroup.",
    },
  ];
}

class MembwAdditionalPlots extends BaseMembwTest {
  testname = "additional-plots";
  static plots = [
    {
      filter: function (row) {
        return (
          row.category == "membw" /* this.category */ &&
          (row.testname != "GSL fp32-per-thread" || row.workgroupCount == 128)
        );
      },
      x: { field: "memsrcSize", label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: "workgroupSize", label: "Workgroup Size" },
      stroke: { field: "testname" },
      caption:
        "Memory bandwidth test (lines are test name, workgroupCount GSL == 128). Results should indicate that for ~large workgroup sizes, a GSL is at least as good as one thread per item.",
    },
  ];
}

export const MembwSimpleTestSuite = {
  class: MembwSimpleTestClass,
  params: MembwSimpleTestParams,
};

export const MembwGSLTestSuite = {
  class: MembwGSLTestClass,
  params: MembwGSLTestParams,
};

export const MembwAdditionalPlotsSuite = {
  class: MembwAdditionalPlots,
  params: {},
};
