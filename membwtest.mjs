import { range } from "./util.mjs";
export const membwTest = {
  category: "membw",
  testname: "fp32-per-thread",
  description:
    "Copies input array to output array. One thread is assigned per 32b input element.",
  parameters: {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 25).map((i) => 2 ** i),
  },
  trials: 10,
  kernel: (param) => /* wgsl */ `
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
    }`,
  validate: (input, output) => {
    return input + 1.0 == output;
  },
  bytesTransferred: (memInput, memOutput) => {
    return memInput.byteLength + memOutput.byteLength;
  },
  plots: [
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
  ],
};

/**
 * grid stride loop, now we don't assign a fixed number of elements per thread
 * background: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */
export const membwGSLTest = Object.assign({}, membwTest); // copy from membwTest
membwGSLTest.testname = "GSL fp32-per-thread";
membwGSLTest.kernel = (param) => /* wgsl */ `
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
membwGSLTest.dispatchGeometry = (param) => {
  return [param.workgroupCount];
};
membwGSLTest.parameters = {
  workgroupSize: range(0, 7).map((i) => 2 ** i),
  memsrcSize: range(10, 25).map((i) => 2 ** i),
  workgroupCount: range(5, 10).map((i) => 2 ** i),
};
membwGSLTest.plots = [
  {
    x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fy: { field: (d) => d.param.workgroupCount, label: "Workgroup Count" },
    stroke: { field: (d) => d.param.workgroupSize },
    title_: "Memory bandwidth test (lines are workgroup size)",
  },
  {
    x: { field: (d) => d.param.memsrcSize, label: "Copied array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fy: { field: (d) => d.param.workgroupSize, label: "Workgroup Size" },
    stroke: { field: (d) => d.param.workgroupCount },
    title_: "Memory bandwidth test (lines are workgroup size)",
  },
];
