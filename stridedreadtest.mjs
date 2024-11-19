import { range } from "./util.mjs";
export const stridedReadTest = {
  category: "strided-read",
  testname: "u32-per-thread",
  description:
    "Copies strided input array to output array. One thread is assigned per 32b input element.",
  datatype: "u32",
  parameters: {
    workgroupSize: [32, 64, 96, 128, 160, 192, 224, 256], // range(0, 8).map((i) => 2 ** i),
    log2stride: range(/* 0 */ 0, 12),
  },
  trials: 10,
  /**
   * First algorithm for ensuring strided access:
   * For stride 2**n: swap bit positions [0:n-1] and [n:2n-1] in the address
   * Bit mask = (1<<n)-1
   * xxxxaaabbb -> xxxxbbbaaa (for log2stride = 3)
   * Assumes that bitmath is free (compared to mem access time)
   * This turns out to be susceptible to caching
   * (all accesses within 2 ** (2 * log2stride) are in a block), so instead:
   *
   * Rotate by log2stride:
   * xxxxxxxbbb -> bbbxxxxxxx (for log2stride = 3)
   */
  kernel: function (param, numThreads) {
    /* function because we need 'this' */
    return /* wgsl */ `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

    @compute @workgroup_size(${param.workgroupSize}) fn memcpyKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        if (i < ${numThreads}) {
          let address_space_mask: u32 = (1 << ${this.log2memsrcSize}) - 1;
          /* assert: i == i & address_space_mask */
          let lshift: u32 = ${param.log2stride};
          let rshift: u32 = ${this.log2memsrcSize - param.log2stride};
          let src: u32 = ((i << lshift) | (i >> rshift)) & address_space_mask;
          memDest[i] = memSrc[src];// + 1;
          // memDest[i] = src;
        }
    }`;
  },
  log2memsrcSize: 27,
  memsrcSize: function (param) {
    return 2 ** this.log2memsrcSize;
  }, // min(device.limits.maxBufferSize, maxStorageBufferBindingSize) / 4,
  memdestSize: function (param) {
    /* need 'function', arrow notation has no 'this' */
    return this.memsrcSize(param);
  },
  workgroupCount: function (param) {
    return Math.ceil(this.memdestSize(param) / param.workgroupSize);
  },
  validate: (input, output) => {
    /* TODO FIX */
    return input /* + 1.0 */ == output;
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
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
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
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
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
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
    },
  ],
};
