import { range } from "./util.mjs";
const subgroupIDBase = {
  category: "subgroups",
  trials: 10,
};

export const subgroupIDTest = Object.assign({}, subgroupIDBase);
subgroupIDTest.description = "Subgroup ID and size";
subgroupIDTest.parameters = {
  workgroupCount: range(0, 7).map((i) => 2 ** i),
  workgroupSize: range(0, 7).map((i) => 2 ** i),
};
subgroupIDTest.datatype = "u32";
subgroupIDTest.kernel = (param) => /* wsgl */ `
    enable subgroups;
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${param.workgroupSize}) fn subgroupIdKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u,
      @builtin(subgroup_size) sgsz: u32,
      @builtin(subgroup_invocation_id) sgid: u32) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        if (i < arrayLength(&memSrc)) {
          memDest[i] = (sgsz << 16) | sgid;
        }
    }`;
subgroupIDTest.memsrcSize = (param) => {
  return param.workgroupCount * param.workgroupSize;
};
subgroupIDTest.bytesTransferred = (memInput, memOutput) => {
  return memInput.byteLength + memOutput.byteLength;
};
subgroupIDTest.threadCount = (memInput) => {
  return memInput.byteLength / 4;
};
subgroupIDTest.plots = [];

/* subgroup sum */

export const subgroupSumSGTest = Object.assign({}, subgroupIDTest);
subgroupIDTest.datatype = "f32";
subgroupIDTest.kernel = (param) => /* wsgl */ `
    enable subgroups;
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    var<workgroup> temp: array<f32, ${param.workgroupSize}>; // zero initialized

    @compute @workgroup_size(${param.workgroupSize}) fn subgroupIdKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u,
      @builtin(subgroup_size) sgsz: u32,
      @builtin(subgroup_invocation_id) sgid: u32) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        memDest[i] = subgroupAdd(memSrc[i]);
    }`;
subgroupSumSGTest.dumpF = true;

export const subgroupSumWGTest = Object.assign({}, subgroupSumSGTest);
subgroupSumWGTest.datatype = "f32";
subgroupSumWGTest.kernel = (param) => /* wsgl */ `
    enable subgroups;
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    var<workgroup> temp: array<f32, ${param.workgroupSize}>; // zero initialized

    @compute @workgroup_size(${param.workgroupSize}) fn subgroupIdKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(local_invocation_id) lid: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u,
      @builtin(subgroup_size) sgsz: u32,
      @builtin(subgroup_invocation_id) sgid: u32) {
        let i: u32 = id.y * nwg.x * ${param.workgroupSize} + id.x;
        var sum: f32 = memSrc[i];
        /* now switch to local IDs only */
        temp[lid.x] = sum;
        workgroupBarrier();
        if (lid.x < sgsz) {
          for (var j: u32 = lid.x + sgsz; j < ${param.workgroupSize}; j += sgsz ) {
            sum += temp[j];
          }
        }
        temp[lid.x] = subgroupAdd(sum);
        workgroupBarrier();
        /* now back to global ID for global writeback */
        memDest[i] = temp[0];
    }`;
