import { range } from "./util.mjs";
export const stridedReadTest = {
  category: "strided-read",
  testname: "fp32-per-thread",
  description:
    "Copies strided input array to output array. One thread is assigned per 32b input element.",
  parameters: {
    workgroupSize: range(0, 8).map((i) => 2 ** i),
    log2stride: range(0, 12),
  },
  trials: 10,
  /**
   * Algorithm for ensuring strided access:
   * For stride 2**n: swap bit positions [0:n-1] and [n:2n-1] in the address
   * Bit mask = (1<<n)-1
   * xxxxaaabbb -> xxxxbbbaaa (for log2stride = 3)
   * Assumes that bitmath is free (compared to mem access time)
   */
  kernel: (param) => /* wgsl */ `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${param.workgroupSize}) fn memcpyKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        let low_mask: u32 = (1 << ${param.log2stride}) - 1;
        let high_mask: u32 = low_mask << ${param.log2stride};
        let total_mask: u32 = low_mask | high_mask;
        let low_chunk: u32 = low_mask & i;
        let high_chunk: u32 = high_mask & i;
        let src: u32 = (i & ~total_mask) | (low_chunk << ${param.log2stride}) | (high_chunk >> ${param.log2stride});
        memDest[i] = memSrc[src] + 1.0;
    }`,
  memsrcSize: (param) => 2 ** 27 / 4, // min(device.limits.maxBufferSize, maxStorageBufferBindingSize) / 4,
  memdestSize: function (param) {
    /* need 'function', arrow notation has no 'this' */
    return this.memsrcSize(param);
  },
  workgroupCount: function (param) {
    return this.memdestSize(param) / param.workgroupSize;
  },
  validate: (input, output) => {
    return input + 1.0 == output;
  },
  bytesTransferred: (memInput, memOutput) => {
    /* assumes that strided access time >> coalesced writeback */
    return memOutput.byteLength;
  },
  plots: [
    {
      x: {
        field: (d) => d.param.workgroupSize,
        label: "Workgroup size",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: {
        field: (d) => 2 ** d.param.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 fp32 per thread (lines are stride distance)",
    },
    {
      x: {
        field: (d) => d.param.workgroupSize,
        label: "Workgroup size",
      },
      y: { field: "time", label: "Runtime (ns)" },
      stroke: {
        field: (d) => 2 ** d.param.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 fp32 per thread (lines are stride distance)",
    },
    {
      x: {
        field: function (d) {
          return stridedReadTest.workgroupCount(d.param);
        } /* 'this' doesn't work */,
        label: "Workgroup Count",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: {
        field: (d) => 2 ** d.param.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 fp32 per thread (lines are stride distance)",
    },
  ],
};
