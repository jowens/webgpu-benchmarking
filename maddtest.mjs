import { range } from "./util.mjs";
import { BasePrimitive } from "./primitive.mjs";

const MaddTestParams = {
  workgroupSize: range(0, 7).map((i) => 2 ** i),
  memsrcSize: range(10, 26).map((i) => 2 ** i),
  opsPerThread: range(2, 10).map((i) => 2 ** i),
};

class MaddTestClass extends BasePrimitive {
  constructor(params) {
    super(params); // writes parameters into this class
    this.category = "madd";
    this.description =
      "Computes N multiply-adds per input element. One thread is responsible for one 32b input element.";

    this.trials = 10;
    this.kernel = function () {
      var k = /* wgsl */ `
        /* output */
        @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
        /* input */
        @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

        @compute @workgroup_size(${this.workgroupSize}) fn maddKernel(
          @builtin(global_invocation_id) id: vec3u,
          @builtin(num_workgroups) nwg: vec3u,
          @builtin(workgroup_id) wgid: vec3u) {
            let i = id.y * nwg.x * ${this.workgroupSize} + id.x;
            if (i < arrayLength(&memSrc)) {
              var f = memSrc[i];
              /* 2^-22 = 2.38418579e-7 */
              var b = f * 2.38418579e-7 + 1.0;
              /* b is a float btwn 1 and 2 */`;
      let opt = this.opsPerThread;
      while (opt > 2) {
        k = k + "    f = f * b + b;\n";
        opt -= 2;
      }
      k = k + "    memDest[i] = f;\n}\n}";
      return k;
    };

    this.memdestSize = this.memsrcSize;
    this.numThreads = this.memsrcSize;
    this.workgroupCount = this.memsrcSize / this.workgroupSize;
    this.bytesTransferred = (this.memsrcSize + this.memdestSize) * 4;
    this.gflops = (time) => {
      return (this.numThreads * this.opsPerThread) / time;
    };
  }
  validate = (memsrc, memdest) => {
    const sum = new Uint32Array([0]);
    for (let i = 0; i < memsrc.length; i++) {
      let f = memsrc[i];
      const b = f * 2.38418579e-7 + 1.0;
      /* b is a float btwn 1 and 2 */
      let opsPerThread = this.opsPerThread;
      while (opsPerThread > 2) {
        f = f * b + b;
        opsPerThread -= 2;
      }
      // allow for a bit of FP error
      if (Math.abs(f - memdest[i]) / f > 0.00001) {
        return `Element ${i}: expected ${f}, instead saw ${memdest[i]}.`;
      } else {
        return "";
      }
    }
  };
  static plots = [
    {
      x: { field: "numThreads", label: "Number of threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: "opsPerThread", label: "Ops per Thread" },
      caption_tl: "Workgroup size = 64 (lines are ops per thread)",
      filter: (row) => row.workgroupSize == 64,
      caption:
        "Max GFLOPS test as function of number of threads (lines are ops per thread)",
    },
    {
      x: { field: "numThreads", label: "Number of threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: "workgroupSize" },
      caption_tl: "Each thread does 16 MADDs (lines are workgroup size)",
      filter: (row) => row.opsPerThread == 16,
      caption:
        "Max GFLOPS test as function of number of threads; each thread does 16 ops (lines are workgroup size)",
    },
    {
      x: { field: "numThreads", label: "Number of threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: "workgroupSize" },
      caption_tl: "Each thread does 64 MADDs (lines are workgroup size)",
      filter: (row) => row.opsPerThread == 64,
      caption:
        "Max GFLOPS test as function of number of threads; each thread does 64 ops (lines are workgroup size)",
    },
    {
      x: { field: "numThreads", label: "Number of threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: "workgroupSize" },
      caption_tl: "Each thread does 256 MADDs (lines are workgroup size)",
      filter: (row) => row.opsPerThread == 256,
      caption:
        "Max GFLOPS test as function of number of threads; each thread does 256 ops (lines are workgroup size)",
    },
  ];
}

export const MaddTestSuite = {
  class: MaddTestClass,
  params: MaddTestParams,
};
