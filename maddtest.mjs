import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
export class MaddTest extends BaseTest {
  category = "madd";
  description =
    "Computes N multiply-adds per input element. One thread is responsible for one 32b input element.";
  parameters = {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 26).map((i) => 2 ** i),
    opsPerThread: range(2, 10).map((i) => 2 ** i),
  };
  trials = 10;
  kernel = (param) => {
    /* wsgl */ var k = `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${param.workgroupSize}) fn maddKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i = id.y * nwg.x * ${param.workgroupSize} + id.x;
        if (i < arrayLength(&memSrc)) {
        var f = memSrc[i];
        /* 2^-22 = 2.38418579e-7 */
        var b = f * 2.38418579e-7 + 1.0;
        /* b is a float btwn 1 and 2 */`;
    let opt = param.opsPerThread;
    while (opt > 2) {
      k = k + "    f = f * b + b;\n";
      opt -= 2;
    }
    k = k + "    memDest[i] = f;\n}\n}";
    return k;
  };
  validate = (input, output, param) => {
    let f = input;
    const b = f * 2.38418579e-7 + 1.0;
    /* b is a float btwn 1 and 2 */
    let opsPerThread = param.opsPerThread;
    while (opsPerThread > 2) {
      f = f * b + b;
      opsPerThread -= 2;
    }
    // allow for a bit of FP error
    return Math.abs(f - output) / f < 0.00001;
  };
  bytesTransferred = (memInput, memOutput) => {
    return memInput.byteLength + memOutput.byteLength;
  };
  threadCount = (memInput) => {
    return memInput.byteLength / 4;
  };
  flopsPerThread = (param) => {
    return param.opsPerThread;
  };
  gflops = (threads, flopsPerThread, time) => {
    return (threads * flopsPerThread) / time;
  };
  plots = [
    {
      x: { field: "threadCount", label: "Active threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: (d) => d.param.opsPerThread, label: "Ops per Thread" },
      caption_tl: "Workgroup size = 64 (lines are ops per thread)",
      filter: (data, param) => data.filter((d) => d.param.workgroupSize == 64),
    },
    {
      x: { field: "threadCount", label: "Active threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption_tl: "Each thread does 16 MADDs (lines are workgroup size)",
      filter: (data, param) => data.filter((d) => d.param.opsPerThread == 16),
    },
    {
      x: { field: "threadCount", label: "Active threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption_tl: "Each thread does 64 MADDs (lines are workgroup size)",
      filter: (data, param) => data.filter((d) => d.param.opsPerThread == 64),
    },
    {
      x: { field: "threadCount", label: "Active threads" },
      y: { field: "gflops", label: "GFLOPS" },
      stroke: { field: (d) => d.param.workgroupSize },
      caption_tl: "Each thread does 256 MADDs (lines are workgroup size)",
      filter: (data, param) => data.filter((d) => d.param.opsPerThread == 256),
    },
  ];
}
