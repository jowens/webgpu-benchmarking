import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
class BaseMembwTest extends BaseTest {
  constructor(params) {
    super(params); // writes parameters into this class
    this.category = "membw";
    this.trials = 10;
    this.validate = (input, output) => {
      return input + 1.0 == output;
    };
    this.bytesTransferred = (memInput, memOutput) => {
      return memInput.byteLength + memOutput.byteLength;
    };
  }
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
    this.kernel = (param) => /* wgsl */ `
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

      @compute @workgroup_size(${param.workgroupSize}) fn memcpyKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u, // == dispatch
        @builtin(workgroup_id) wgid: vec3u) {
          /* grid-stride loop: assume nwg.y == 1 */
          for (var i = id.x;
               i < arrayLength(&memSrc);
               i += nwg.x * ${param.workgroupSize}) {
            memDest[i] = memSrc[i] + 1.0;
          }
      }`;
    this.dispatchGeometry = (param) => {
      return [param.workgroupCount];
    };
  }

  static plots = [
    {
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: (d) => d.param.workgroupCount, label: "Workgroup Count" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption: "Memory bandwidth test GSL (lines are workgroup size)",
    },
    {
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: (d) => d.param.workgroupSize, label: "Workgroup Size" },
      stroke: { field: (d) => d.param.workgroupCount },
      caption:
        "Memory bandwidth test GSL (lines are workgroup size). Looks like max throughput doesn't occur until ~512 threads/workgroup.",
    },
  ];
}

export class MembwAdditionalPlots extends BaseMembwTest {
  testname = "additional-plots";
  plots = [
    {
      filter: function (row) {
        return (
          row.category == "membw" /* this.category */ &&
          (row.testname != "GSL fp32-per-thread" ||
            row.param.workgroupCount == 128)
        );
      },
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      fy: { field: (d) => d.param.workgroupSize, label: "Workgroup Size" },
      stroke: { field: "testname" },
      caption:
        "Memory bandwidth test (lines are test name, workgroupCount GSL == 128). Results should indicate a GSL is at least as good as one thread per item.",
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
