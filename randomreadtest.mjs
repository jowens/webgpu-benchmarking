import { range } from "./util.mjs";
export const randomReadTest = {
  category: "random-read",
  testname: "u32-per-thread",
  description:
    "Fetches from 'random' memory location. One thread is assigned per 32b input element.",
  datatype: "u32",
  parameters: {
    workgroupSize: [32, 64, 96, 128, 160, 192, 224, 256], // range(0, 8).map((i) => 2 ** i),
  },
  trials: 10,
  kernel: function (param) {
    /* function because we need 'this' */
    return /* wgsl */ `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

    @compute @workgroup_size(${param.workgroupSize}) fn randomReadKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        let address_space_mask: u32 = (1 << ${this.log2memsrcSize}) - 1;
        /* assert: i == i & address_space_mask */
        var x: u32 = i;
        /**
         * this is perfect shuffle
        x = ((x & 0x0000FF00) << 8) | ((x >> 8) & 0x0000FF00) | (x & 0xFF0000FF);
        x = ((x & 0x00F000F0) << 4) | ((x >> 4) & 0x00F000F0) | (x & 0xF00FF00F);
        x = ((x & 0x0C0C0C0C) << 2) | ((x >> 2) & 0x0C0C0C0C) | (x & 0xC3C3C3C3);
        x = ((x & 0x22222222) << 1) | ((x >> 1) & 0x22222222) | (x & 0x99999999);
        x &= address_space_mask;
        */
       /** this is bit reverse
        x = ((x & 0x55555555)  <<   1) | ((x & 0xAAAAAAAA) >>  1);
        x = ((x & 0x33333333)  <<   2) | ((x & 0xCCCCCCCC) >>  2);
        x = ((x & 0x0F0F0F0F)  <<   4) | ((x & 0xF0F0F0F0) >>  4);
        x = ((x & 0x00FF00FF)  <<   8) | ((x & 0xFF00FF00) >>  8);
        x = ((x & 0x0000FFFF)  <<  16) | ((x & 0xFFFF0000) >> 16);
        x >>= 32 - ${this.log2memsrcSize};
        */
        x = 28428919 * x + 30555407; // two random primes
        x &= address_space_mask;
        memDest[i] = memSrc[x];// + 1;
        // memDest[i] = src;
    }`;
  },
  log2memsrcSize: 28,
  memsrcSize: function (param) {
    return 2 ** this.log2memsrcSize;
  }, // min(device.limits.maxBufferSize, maxStorageBufferBindingSize) / 4,
  memdestSize: function (param) {
    /* need 'function', arrow notation has no 'this' */
    return this.memsrcSize(param);
  },
  workgroupCount: function (param) {
    return this.memdestSize(param) / param.workgroupSize;
  },
  validate: (input, output) => {
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
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
    {
      x: {
        field: function (d) {
          return randomReadTest.workgroupCount(d.param);
        } /* 'this' doesn't work */,
        label: "Workgroup Count",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
    {
      x: {
        field: (d) => d.param.workgroupSize,
        label: "Workgroup size",
      },
      y: { field: "time", label: "Runtime (ns)" },
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
  ],
};
