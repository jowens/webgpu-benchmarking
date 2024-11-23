import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";
class BaseReduceTest extends BaseTest {
  category = "reduce";
  trials = 10;
}

export class ReducePerWGTest extends BaseReduceTest {
  testname = "reduce per wg";
  workgroupSizes = range(2, 7).map((i) => 2 ** i);
  memsrcSizes = range(16, 17).map((i) => 2 ** i);
  kernel = (workgroupSize) => /* wgsl */ `
    enable subgroups;
    // var<workgroup> sum: f32; // zero initialized?
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${workgroupSize}) fn reducePerWGKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i = id.y * nwg.x * ${workgroupSize} + id.x;
        let sa = subgroupAdd(memSrc[i]);
        memDest[i] = sa;
    }
  `;
  validate = (f) => {
    return f;
  };
}
