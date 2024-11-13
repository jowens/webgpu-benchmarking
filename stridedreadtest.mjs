import { range } from "./util.mjs";
export const stridedReadTest = {
  category: "strided-read",
  testname: "fp32-per-thread",
  description:
    "Copies strided input array to output array. One thread is assigned per 32b input element.",
  parameters: {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    stride: range(0, 11).map((i) => 2 ** i),
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
        memDest[i] = memSrc[i*${param.stride}] + 1.0;
    }`,
  memsrcSize: (param) => 2 ** 27 / 4, // device.limits.maxBufferSize / 4,
  memdestSize: function (param) {
    return this.memsrcSize(param) / param.stride;
  },
  workgroupCount: function (param) {
    return this.memdestSize(param) / param.workgroupSize;
  },
  validate: (input, output) => {
    return input + 1.0 == output;
  },
  bytesTransferred: (memInput, memOutput) => {
    return memOutput.byteLength * 2;
  },
  plots: [
    {
      x: {
        field: (d) => d.param.workgroupSize,
        label: "Workgroup size",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { field: (d) => d.param.stride, label: "Stride distance" },
      caption:
        "Strided memory bandwidth test, 1 fp32 per thread (lines are stride distance)",
    },
  ],
};
