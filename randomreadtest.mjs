import { range } from "./util.mjs";
import { BasePrimitive } from "./primitive.mjs";

const RandomReadTestParams = {
  workgroupSize: [32, 64, 96, 128, 160, 192, 224, 256], // range(0, 8).map((i) => 2 ** i),
};

class RandomReadTestClass extends BasePrimitive {
  constructor(params) {
    super(params); // writes parameters into this class
    this.category = "random-read";
    this.testname = "u32-per-thread";
    this.description =
      "Fetches from 'random' memory location. One thread is assigned per 32b input element.";
    this.datatype = "u32";
    this.trials = 10;
    this.kernel = function () {
      /* function because we need 'this' */
      return /* wgsl */ `
        /* output */
        @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
        /* input */
        @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

        @compute @workgroup_size(${this.workgroupSize}) fn randomReadKernel(
          @builtin(global_invocation_id) id: vec3u,
          @builtin(num_workgroups) nwg: vec3u,
          @builtin(workgroup_id) wgid: vec3u) {
            let i: u32 = id.y * nwg.x * ${this.workgroupSize} + id.x;
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
    };
    this.log2memsrcSize = 28;
    this.memsrcSize = 2 ** this.log2memsrcSize;
    // min(device.limits.maxBufferSize, maxStorageBufferBindingSize) / 4,
    this.memdestSize = this.memsrcSize;
    this.workgroupCount = this.memdestSize / this.workgroupSize;
    this.numThreads = this.memsrcSize;
    this.bytesTransferred = this.numThreads * 4;
  }
  validate = (memsrc, memdest) => {
    function randomize(addr, bits) {
      return (28428919 * addr + 30555407) & ((1 << bits) - 1);
    }
    for (let i = 0; i < memsrc.length; i++) {
      const expected = memsrc[randomize(i, this.log2memsrcSize)];
      if (expected != memdest[i]) {
        return `Element ${i}: expected ${expected}, instead saw ${memdest[i]}.`;
      } else {
        return "";
      }
    }
  };
  static plots = [
    {
      x: {
        field: "workgroupSize",
        label: "Workgroup size",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
    {
      x: {
        field: "workgroupCount",
        label: "Workgroup Count",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
    {
      x: {
        field: "workgroupSize",
        label: "Workgroup size",
      },
      y: { field: "time", label: "Runtime (ns)" },
      caption: "Random memory bandwidth test, 1 u32 per thread",
    },
  ];
}

export const RandomReadTestSuite = {
  class: RandomReadTestClass,
  params: RandomReadTestParams,
};
