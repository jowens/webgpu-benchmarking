import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
class BaseMembwTest extends BaseTest {
  category = "membw";
  trials = 10;
  validate = (input, output) => {
    return input + 1.0 == output;
  };
  bytesTransferred = (memInput, memOutput) => {
    return memInput.byteLength + memOutput.byteLength;
  };
}

export class MembwSimpleTest extends BaseMembwTest {
  testname = "fp32-per-thread";
  description =
    "Copies input array to output array. One thread is assigned per 32b input element.";
  parameters = {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 25).map((i) => 2 ** i),
  };
  kernel = (param) => /* wgsl */ `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${param.workgroupSize}) fn memcpyKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i = id.y * nwg.x * ${param.workgroupSize} + id.x;
        memDest[i] = memSrc[i] + 1.0;
    }`;
  plots = [
    {
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: {
        field: "bandwidthCPU",
        label: "Achieved bandwidth (GB/s) [CPU measurement]",
      },
      stroke: { field: (d) => d.param.workgroupSize },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: "time", label: "GPU time (ns)" },
      y: { field: "cpuns", label: "CPU time (ns)" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
    {
      x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
      y: { field: "cpugpuDelta", label: "CPU - GPU time (ns)" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption:
        "Memory bandwidth test, 1 fp32 per thread (lines are workgroup size)",
    },
  ];
}

/**
 * grid stride loop, now we don't assign a fixed number of elements per thread
 * background: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */
export class MembwGSLTest extends BaseMembwTest {
  testname = "GSL fp32-per-thread";
  kernel = (param) => /* wgsl */ `
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
  dispatchGeometry = (param) => {
    return [param.workgroupCount];
  };
  parameters = {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 25).map((i) => 2 ** i),
    workgroupCount: range(5, 10).map((i) => 2 ** i),
  };
  plots = [
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
