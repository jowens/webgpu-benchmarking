import { range } from "./util.mjs";
import { BaseTest } from "./basetest.mjs";

class SubgroupIDBaseTest extends BaseTest {
  constructor(params) {
    super(params);
    this.category = "subgroups";
    this.trials = 10;
  }
}

const SubgroupIDTestParams = {
  workgroupCount: range(0, 7).map((i) => 2 ** i),
  workgroupSize: range(0, 7).map((i) => 2 ** i),
};

class SubgroupIDTestClass extends SubgroupIDBaseTest {
  constructor(params) {
    super(params);
    this.description = "Subgroup ID and size";
    this.datatype = "u32";
    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      @compute @workgroup_size(${this.workgroupSize}) fn subgroupIdKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i: u32 = id.y * nwg.x * ${this.workgroupSize} + id.x;
          if (i < arrayLength(&memSrc)) {
            memDest[i] = (sgsz << 16) | sgid;
          }
      }`;
    this.memsrcSize = this.workgroupCount * this.workgroupSize;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = this.memsrcSize + this.memdestSize;
    this.numThreads = this.memsrcSize;
  }
  validate(memdest) {
    const sgsz = 32; // TODO get from GPU adapter
    for (let i = 0; i < this.memdestSize; i++) {
      const wgid = Math.floor(i / this.workgroupSize);
      const sgid = (i - wgid * this.workgroupSize) % sgsz;
      const expected = (sgsz << 16) | sgid;
      if (memdest[i] != expected) {
        return `Element ${i}: expected ${expected}, instead saw ${memdest[i]}.`;
      }
    }
    return ""; // passed
  }
  static plots = [];
}

export const SubgroupIDTestSuite = {
  class: SubgroupIDTestClass,
  params: SubgroupIDTestParams,
};

/* subgroup sum */

class SubgroupSumSGTestClass extends SubgroupIDBaseTest {
  constructor(params) {
    super(params);
    this.datatype = "f32";
    this.description =
      "Output of each thread is the sum of inputs in its subgroup";
    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      var<workgroup> temp: array<f32, ${this.workgroupSize}>; // zero initialized

      @compute @workgroup_size(${this.workgroupSize}) fn subgroupSumSGKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i: u32 = id.y * nwg.x * ${this.workgroupSize} + id.x;
          memDest[i] = subgroupAdd(memSrc[i]);
      }`;
    this.memsrcSize = this.workgroupCount * this.workgroupSize;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = this.memsrcSize + this.memdestSize;
    this.numThreads = this.memsrcSize;
    this.dumpF = true;
  }
  validate(memdest) {
    const sgsz = 32; // TODO get from GPU adapter
    for (let wg = 0; wg < this.workgroupCount; wg++) {
      for (let sg = 0; sg < Math.ceil(this.workgroupSize / sgsz); sg++) {
        // first two are bounds WITHIN a workgroup
        const sgStart = sg * sgsz;
        const sgEnd = Math.min((sg + 1) * sgsz, this.workgroupSize); // 1 past end
        // these two are GLOBAL indexes
        const iStart = wg * this.workgroupSize + sgStart;
        const iEnd = iStart + (sgEnd - sgStart);
        let sum = 0;
        // add up the range
        for (let i = iStart; i < iEnd; sum += i, i++);
        // and now check it
        for (let i = iStart; i < iEnd; i++) {
          if (memdest[i] != sum) {
            return `Element ${i}: expected ${sum}, instead saw ${memdest[i]}.`;
          }
        }
      }
    }
    return ""; // passed
  }
  static plots = [];
}

export const SubgroupSumSGTestSuite = {
  class: SubgroupSumSGTestClass,
  params: SubgroupIDTestParams,
};

class SubgroupSumWGTestClass extends SubgroupIDBaseTest {
  constructor(params) {
    super(params);
    this.datatype = "f32";
    this.kernel = () => /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
      /* input */
      @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

      var<workgroup> temp: array<f32, ${this.workgroupSize}>; // zero initialized

      @compute @workgroup_size(${this.workgroupSize}) fn subgroupSumWGKernel(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(local_invocation_index) lid: u32,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i: u32 = id.y * nwg.x * ${this.workgroupSize} + id.x;
          var acc: f32 = memSrc[i];
          /* now switch to local IDs only */
          temp[lid] = acc;
          workgroupBarrier();
          if (lid < sgsz) { /* 0th subgroup */
            for (var j: u32 = lid + sgsz; j < ${this.workgroupSize}; j += sgsz ) {
              acc += temp[j];
            }
          }
          temp[lid] = subgroupAdd(acc);
          workgroupBarrier();
          /* now back to global ID for global writeback */
          memDest[i] = temp[0];
        }`;
    this.memsrcSize = this.workgroupCount * this.workgroupSize;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = this.memsrcSize + this.memdestSize;
    this.numThreads = this.memsrcSize;
  }
  validate(memdest) {
    for (let wg = 0; wg < this.workgroupCount; wg++) {
      const iStart = wg * this.workgroupSize;
      const iEnd = iStart + this.workgroupSize;
      let sum = 0;
      // add up the range
      for (let i = iStart; i < iEnd; sum += i, i++);
      // and now check it
      for (let i = iStart; i < iEnd; i++) {
        if (memdest[i] != sum) {
          return `Element ${i}: expected ${sum}, instead saw ${memdest[i]}.`;
        }
      }
    }
    return ""; // passed
  }
  static plots = [];
}

export const SubgroupSumWGTestSuite = {
  class: SubgroupSumWGTestClass,
  params: SubgroupIDTestParams,
};
