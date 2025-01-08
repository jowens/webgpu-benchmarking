import { range } from "./util.mjs";
import { BasePrimitive } from "./primitive.mjs";

class SubgroupIDBaseTest extends BasePrimitive {
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
  validate(memsrc, memdest) {
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
  validate(memsrc, memdest) {
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
        for (let i = iStart; i < iEnd; sum += memsrc[i], i++);
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
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) lid: u32,
        @builtin(num_workgroups) nwg: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(subgroup_size) sgsz: u32,
        @builtin(subgroup_invocation_id) sgid: u32) {
          let i: u32 = gid.y * nwg.x * ${this.workgroupSize} + gid.x;
          var acc: f32 = memSrc[i];
          /* now switch to local IDs only */
          temp[lid] = acc;
          workgroupBarrier(); /* completely populate shmem */
          if (lid < sgsz) { /* only activate 0th subgroup */
            /* accumulate all other subgroups into acc, in parallel across the subgroup */
            /* tree-sum would be more efficient */
            for (var j: u32 = lid + sgsz; j < ${this.workgroupSize}; j += sgsz ) {
              acc += temp[j];
            }
          }
          workgroupBarrier(); /* why this barrier? */
          temp[lid] = subgroupAdd(acc); /* we only care about result for 0th subgroup;
                                         * actually, we only care about temp[0] */
          workgroupBarrier();
          /* now use global ID for global writeback */
          memDest[i] = temp[0];
        }`;
    /* 64-element workgroup:
     * Validation failed: Element 0: expected 2016, instead saw 49136.
     * 2016 = 63 * 32 = sum(0->63); this is correct
     * 49136 = 16 * 3071
     * Float32Array(64) [32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, buffer: ArrayBuffer(256), byteLength: 256, byteOffset: 0, length: 64, Symbol(Symbol.toStringTag): 'Float32Array']
     * benchmarking.mjs:237 Validation failed: Element 0: expected 2016, instead saw 32.
     * Float32Array(64) [49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 49136, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, 1520, buffer: ArrayBuffer(256), byteLength: 256, byteOffset: 0, length: 64, Symbol(Symbol.toStringTag): 'Float32Array']
     */
    this.memsrcSize = this.workgroupCount * this.workgroupSize;
    this.memdestSize = this.memsrcSize;
    this.bytesTransferred = this.memsrcSize + this.memdestSize;
    this.numThreads = this.memsrcSize;
  }
  validate(memsrc, memdest) {
    for (let wg = 0; wg < this.workgroupCount; wg++) {
      const iStart = wg * this.workgroupSize;
      const iEnd = iStart + this.workgroupSize;
      let sum = 0;
      // add up the range
      for (let i = iStart; i < iEnd; sum += memsrc[i], i++);
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
