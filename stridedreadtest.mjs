import { range } from "./util.mjs";
import { BasePrimitive } from "./primitive.mjs";

const StridedReadTestParams = {
  workgroupSize: [32, 64, 96, 128, 160, 192, 224, 256], // range(0, 8).map((i) => 2 ** i),
  log2stride: range(0, 12),
};

class StridedReadTestClass extends BasePrimitive {
  constructor(params) {
    super(params); // writes parameters into this class
    this.category = "strided-read";
    this.testname = "u32-per-thread";
    this.description =
      "Copies strided input array to output array. One thread is assigned per 32b input element.";
    this.datatype = "u32";
    this.trials = 10;
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
    this.kernel = function () {
      /* function because we need 'this' */
      return /* wgsl */ `
        /* output */
        @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
        /* input */
        @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

        @compute @workgroup_size(${this.workgroupSize}) fn memcpyKernel(
          @builtin(global_invocation_id) id: vec3u,
          @builtin(num_workgroups) nwg: vec3u,
          @builtin(workgroup_id) wgid: vec3u) {
            let i: u32 = id.y * nwg.x * ${this.workgroupSize} + id.x;
            if (i < ${this.numThreads}) {
              let address_space_mask: u32 = (1 << ${this.log2memsrcSize}) - 1;
              /* assert: i == i & address_space_mask */
              let lshift: u32 = ${this.log2stride};
              let rshift: u32 = ${this.log2memsrcSize - this.log2stride};
              let src: u32 = ((i << lshift) | (i >> rshift)) & address_space_mask;
              /** different algorithm!
              * The output is wonkier and I think it has more severe cache effects
              * since low bits are ALWAYS zero
              * I think this is just "i << lshift".
              * let src2: u32 =
              *   (i * (1 << ${this.log2stride})) & address_space_mask;
              * */
              memDest[i] = memSrc[src];// + 1;
              // memDest[i] = src;
            }
          }`;
    };

    this.log2memsrcSize = 26;
    this.memsrcSize = 2 ** this.log2memsrcSize;
    this.numThreads = this.memsrcSize;

    // min(device.limits.maxBufferSize, maxStorageBufferBindingSize) / 4,
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = this.numThreads * 4;
    this.workgroupCount = Math.ceil(this.memdestSize / this.workgroupSize);
  }
  validate = (memsrc, memdest) => {
    function rotateLeft(num, shift, bits) {
      return ((num << shift) | (num >>> (bits - shift))) & ((1 << bits) - 1);
    }
    for (let i = 0; i < memsrc.length; i++) {
      const expected =
        memsrc[rotateLeft(i, this.log2stride, this.log2memsrcSize)];
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
      stroke: {
        field: (d) => 2 ** d.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
    },
    {
      x: {
        field: "workgroupCount",
        label: "Workgroup Count",
      },
      y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: {
        field: (d) => 2 ** d.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
    },
    {
      x: {
        field: "workgroupSize",
        label: "Workgroup size",
      },
      y: { field: "time", label: "Runtime (ns)" },
      stroke: {
        field: (d) => 2 ** d.log2stride,
        label: "Stride distance",
      },
      caption:
        "Strided memory bandwidth test, 1 u32 per thread (lines are stride distance)",
    },
  ];
}

export const StridedReadTestSuite = {
  class: StridedReadTestClass,
  params: StridedReadTestParams,
};
