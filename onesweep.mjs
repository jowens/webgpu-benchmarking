/** Partially based on:
 * Thomas Smith (2024)
 * MIT License
 * https://github.com/b0nes164/GPUSorting
 */

import { range } from "./util.mjs";
import { Kernel, AllocateBuffer } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { datatypeToBytes } from "./util.mjs";
import { BaseSort } from "./sort.mjs";

export class OneSweepSort extends BaseSort {
  constructor(args) {
    super(args);
    this.additionalKnownBuffers = ["foo"];
    for (const knownBuffer of this.additionalKnownBuffers) {
      this.knownBuffers.push(knownBuffer);
    }
  }

  sortOneSweepWGSL = () => {
    let kernel = /* wgsl */ `
    /* enables subgroups ONLY IF it was requested when creating device */
    /* WGSL requires this declaration is first in the shader */
    ${this.fnDeclarations.enableSubgroupsIfAppropriate}
    struct InfoStruct {
        size: u32,
        shift: u32,
        thread_blocks: u32,
        seed: u32,
    };

    @group(0) @binding(0)
    var<uniform> info : InfoStruct;

    @group(0) @binding(1)
    var<storage, read_write> bump: array<atomic<u32>>;

    @group(0) @binding(2)
    var<storage, read_write> sort: array<u32>;

    @group(0) @binding(3)
    var<storage, read_write> alt: array<u32>;

    @group(0) @binding(4)
    var<storage, read_write> payload: array<u32>;

    @group(0) @binding(5)
    var<storage, read_write> alt_payload: array<u32>;

    @group(0) @binding(6)
    var<storage, read_write> hist: array<atomic<u32>>;

    @group(0) @binding(7)
    var<storage, read_write> pass_hist: array<atomic<u32>>;

    @group(0) @binding(8)
    var<storage, read_write> err: array<u32>;

    const SORT_PASSES = 4u;
    const BLOCK_DIM = 256u;
    const MIN_SUBGROUP_SIZE = 4u;
    const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

    const FLAG_NOT_READY = 0u;
    const FLAG_REDUCTION = 1u;
    const FLAG_INCLUSIVE = 2u;
    const FLAG_MASK = 3u;

    const RADIX = 256u;
    const ALL_RADIX = RADIX * SORT_PASSES;
    const RADIX_MASK = 255u;
    const RADIX_LOG = 8u;

    const KEYS_PER_THREAD = 15u;
    const PART_SIZE = KEYS_PER_THREAD * BLOCK_DIM;

    const REDUCE_BLOCK_DIM = 128u;
    const REDUCE_KEYS_PER_THREAD = 30u;
    const REDUCE_HIST_SIZE = REDUCE_BLOCK_DIM / 64u * ALL_RADIX;
    const REDUCE_PART_SIZE = REDUCE_KEYS_PER_THREAD * REDUCE_BLOCK_DIM;

    var<workgroup> wg_globalHist: array<atomic<u32>, REDUCE_HIST_SIZE>;

    /** If we're making subgroup calls and we don't have subgroup hardware,
     * this sets up necessary declarations (workgroup memory, subgroup vars) */
    ${this.fnDeclarations.subgroupEmulation}

    @compute @workgroup_size(REDUCE_BLOCK_DIM, 1, 1)
    fn global_hist(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lid.x / sgsz;  //Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

      //Clear shared memory
      for (var i = builtinsNonuniform.lid.x; i < REDUCE_HIST_SIZE; i += REDUCE_BLOCK_DIM) {
        atomicStore(&wg_globalHist[i], 0u);
      }
      workgroupBarrier();

      //64 threads: 1 histogram in shared memory
      let radix_shift = info.shift;
      let hist_offset = builtinsNonuniform.lid.x / 64u * ALL_RADIX;
      {
        var i = builtinsNonuniform.lid.x + builtinsUniform.wgid.x * REDUCE_PART_SIZE;
        if (builtinsUniform.wgid.x < info.thread_blocks - 1) {
          for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
            let key = sort[i];
            atomicAdd(&wg_globalHist[(key & RADIX_MASK) + hist_offset], 1u);
            atomicAdd(&wg_globalHist[((key >> 8u) & RADIX_MASK) + hist_offset + 256u], 1u);
            atomicAdd(&wg_globalHist[((key >> 16u) & RADIX_MASK) + hist_offset + 512u], 1u);
            atomicAdd(&wg_globalHist[((key >> 24u) & RADIX_MASK) + hist_offset + 768u], 1u);
            i += REDUCE_BLOCK_DIM;
          }
        }

        if (builtinsUniform.wgid.x == info.thread_blocks - 1) {
          for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
            if (i < info.size) {
              let key = sort[i];
              atomicAdd(&wg_globalHist[(key & RADIX_MASK) + hist_offset], 1u);
              atomicAdd(&wg_globalHist[((key >> 8u) & RADIX_MASK) + hist_offset + 256u], 1u);
              atomicAdd(&wg_globalHist[((key >> 16u) & RADIX_MASK) + hist_offset + 512u], 1u);
              atomicAdd(&wg_globalHist[((key >> 24u) & RADIX_MASK) + hist_offset + 768u], 1u);
            }
            i += REDUCE_BLOCK_DIM;
          }
        }
      }
      workgroupBarrier();

      for (var i = builtinsNonuniform.lid.x; i < RADIX; i += REDUCE_BLOCK_DIM) {
        atomicAdd(&hist[i], atomicLoad(&wg_globalHist[i]) + atomicLoad(&wg_globalHist[i + ALL_RADIX]));
        atomicAdd(&hist[i + 256u], atomicLoad(&wg_globalHist[i + 256u]) + atomicLoad(&wg_globalHist[i + 256u + ALL_RADIX]));
        atomicAdd(&hist[i + 512u], atomicLoad(&wg_globalHist[i + 512u]) + atomicLoad(&wg_globalHist[i + 512u + ALL_RADIX]));
        atomicAdd(&hist[i + 768u], atomicLoad(&wg_globalHist[i + 768u]) + atomicLoad(&wg_globalHist[i + 768u + ALL_RADIX]));
      }
    }

    //Assumes block dim 256
    const SCAN_MEM_SIZE = RADIX / MIN_SUBGROUP_SIZE;
    var<workgroup> wg_scan: array<u32, SCAN_MEM_SIZE>;

    ${this.fnDeclarations.commonDefinitions}

    @compute @workgroup_size(BLOCK_DIM, 1, 1)
    fn onesweep_scan(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lid.x / sgsz;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
      let index = builtinsNonuniform.lid.x + builtinsUniform.wgid.x * RADIX;
      let scan = atomicLoad(&hist[index]);
      let red = subgroupAdd(scan);
      if (builtinsNonuniform.sgid == 0u){
        wg_scan[sid] = red;
      }
      workgroupBarrier();

      //Non-divergent subgroup agnostic inclusive scan across subgroup reductions
      {
        var offset0 = 0u;
        var offset1 = 0u;
        let lane_log = u32(countTrailingZeros(sgsz));
        let spine_size = BLOCK_DIM >> lane_log;
        let aligned_size = 1u << ((u32(countTrailingZeros(spine_size)) + lane_log - 1u) / lane_log * lane_log);
        for (var j = sgsz; j <= aligned_size; j <<= lane_log){
          let i0 = ((builtinsNonuniform.lid.x + offset0) << offset1) - select(0u, 1u, j != sgsz);
          let pred0 = i0 < spine_size;
          let t0 = subgroupInclusiveAdd(select(0u, wg_scan[i0], pred0));
          if (pred0){
            wg_scan[i0] = t0;
          }
          workgroupBarrier();

          if (j != sgsz){
            let rshift = j >> lane_log;
            let i1 = builtinsNonuniform.lid.x + rshift;
            if ((i1 & (j - 1u)) >= rshift){
              let pred1 = i1 < spine_size;
              let t1 = select(0u, wg_scan[((i1 >> offset1) << offset1) - 1u], pred1);
              if (pred1 && ((i1 + 1u) & (rshift - 1u)) != 0u){
                wg_scan[i1] += t1;
              }
            }
          } else {
            offset0 += 1u;
          }
          offset1 += lane_log;
        }
      }
      workgroupBarrier();

      let pass_index = builtinsNonuniform.lid.x + info.thread_blocks * builtinsUniform.wgid.x * RADIX;
      atomicStore(&pass_hist[pass_index], ((subgroupExclusiveAdd(scan) +
        select(0u, wg_scan[sid - 1u], builtinsNonuniform.lid.x >= sgsz)) << 2u) | FLAG_INCLUSIVE);
    }

    var<workgroup> wg_warpHist: array<atomic<u32>, PART_SIZE>;
    var<workgroup> wg_localHist: array<u32, RADIX>;
    var<workgroup> wg_broadcast: u32;

    fn WLMS(key: u32, shift: u32, sgid: u32, sgsz: u32, lane_mask_lt: u32, s_offset: u32) -> u32 {
      var eq_mask = 0xffffffffu;
      for (var k = 0u; k < RADIX_LOG; k += 1u) {
        let curr_bit = 1u << (k + shift);
        let pred = (key & curr_bit) != 0u;
        let ballot = subgroupBallot(pred);
        eq_mask &= select(~ballot.x, ballot.x, pred);
      }
      var out = countOneBits(eq_mask & lane_mask_lt);
      let highest_rank_peer = sgsz - countLeadingZeros(eq_mask) - 1u;
      var pre_inc = 0u;
      if (sgid == highest_rank_peer) {
        pre_inc = atomicAdd(&wg_warpHist[((key >> shift) & RADIX_MASK) + s_offset], out + 1u);
      }
      workgroupBarrier();
      out += subgroupShuffle(pre_inc, highest_rank_peer);
      return out;
    }

    fn fake_wlms(key: u32, shift: u32, sgid: u32, sgsz: u32, lane_mask_lt: u32, s_offset: u32) -> u32 {
      return 0u;
    }

    @compute @workgroup_size(BLOCK_DIM, 1, 1)
    fn onesweep_pass(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lid.x / sgsz;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec

      let warp_hists_size = clamp(BLOCK_DIM / sgsz * RADIX, 0u, PART_SIZE);
      for (var i = builtinsNonuniform.lid.x; i < warp_hists_size; i += BLOCK_DIM) {
        atomicStore(&wg_warpHist[i], 0u);
      }

      if (builtinsNonuniform.lid.x == 0u) {
        wg_broadcast = atomicAdd(&bump[info.shift >> 3u], 1u);
      }
      let partid = workgroupUniformLoad(&wg_broadcast);

      var keys = array<u32, KEYS_PER_THREAD>();
      {
        let dev_offset = partid * PART_SIZE;
        let s_offset = sid * sgsz * KEYS_PER_THREAD;
        var i = builtinsNonuniform.sgid + s_offset + dev_offset;
        if (partid < info.thread_blocks - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            keys[k] = sort[i];
            i += sgsz;
          }
        }

        if (partid == info.thread_blocks - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            keys[k] = select(0xffffffffu, sort[i], i < info.size);
            i += sgsz;
          }
        }
      }

      var offsets = array<u32, KEYS_PER_THREAD>();
      {
        let shift = info.shift;
        let lane_mask_lt = (1u << builtinsNonuniform.sgid) - 1u;
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          offsets[k] = WLMS(keys[k], shift, builtinsNonuniform.sgid, sgsz, lane_mask_lt, s_offset);
        }
      }
      workgroupBarrier();

      var local_reduction = 0u;
      if (builtinsNonuniform.lid.x < RADIX) {
        local_reduction = atomicLoad(&wg_warpHist[builtinsNonuniform.lid.x]);
        for (var i = builtinsNonuniform.lid.x + RADIX; i < warp_hists_size; i += RADIX) {
          local_reduction += atomicLoad(&wg_warpHist[i]);
          atomicStore(&wg_warpHist[i], local_reduction - atomicLoad(&wg_warpHist[i]));
        }

        if (partid < info.thread_blocks - 1u) {
          let pass_index = builtinsNonuniform.lid.x + (partid + 1u) * RADIX +
            info.thread_blocks * (info.shift >> 3u) * RADIX;
          atomicStore(&pass_hist[pass_index], (local_reduction << 2u) | FLAG_REDUCTION);
        }

        let lane_mask = sgsz - 1u;
        let circular_lane_shift = (builtinsNonuniform.sgid + lane_mask) & lane_mask;
        let t = subgroupInclusiveAdd(local_reduction);
        wg_localHist[builtinsNonuniform.lid.x] = subgroupShuffle(t, circular_lane_shift);
      }
      workgroupBarrier();

      if (builtinsNonuniform.lid.x < sgsz) {
        let pred = builtinsNonuniform.lid.x < RADIX / sgsz;
        let t = subgroupExclusiveAdd(select(0u, wg_localHist[builtinsNonuniform.lid.x * sgsz], pred));
        if (pred) {
          wg_localHist[builtinsNonuniform.lid.x * sgsz] = t;
        }
      }
      workgroupBarrier();

      if (builtinsNonuniform.lid.x < RADIX && builtinsNonuniform.sgid != 0u) {
        wg_localHist[builtinsNonuniform.lid.x] += wg_localHist[builtinsNonuniform.lid.x / sgsz * sgsz];
      }
      workgroupBarrier();

      if (builtinsNonuniform.lid.x >= sgsz) {
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          let t = (keys[k] >> info.shift) & RADIX_MASK;
          offsets[k] += wg_localHist[t] + atomicLoad(&wg_warpHist[t + s_offset]);
        }
      } else {
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          offsets[k] += wg_localHist[(keys[k] >> info.shift) & RADIX_MASK];
        }
      }
      workgroupBarrier();

      for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
        atomicStore(&wg_warpHist[offsets[k]], keys[k]);
      }

      if (builtinsNonuniform.lid.x < RADIX) {
        var prev_reduction = 0u;
        var lookbackid = partid;
        let part_offset = info.thread_blocks * (info.shift >> 3u) * RADIX;
        while (true) {
          let flagPayload = atomicLoad(&pass_hist[builtinsNonuniform.lid.x + part_offset + lookbackid * RADIX]);
          if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
            prev_reduction += flagPayload >> 2u;
            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
              if (partid < info.thread_blocks - 1u) {
                atomicStore(&pass_hist[builtinsNonuniform.lid.x + part_offset + (partid + 1u) * RADIX],
                  ((prev_reduction + local_reduction) << 2u) | FLAG_INCLUSIVE);
              }
              wg_localHist[builtinsNonuniform.lid.x] = prev_reduction - wg_localHist[builtinsNonuniform.lid.x];
              break;
            } else {
              lookbackid -= 1u;
            }
          }
        }
      }
      workgroupBarrier();

      if (partid < info.thread_blocks - 1u) {
        var i = builtinsNonuniform.lid.x;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          var whi = atomicLoad(&wg_warpHist[i]);
          alt[wg_localHist[(whi >> info.shift) & RADIX_MASK] + i] = whi;
          i += BLOCK_DIM;
        }
      }

      if (partid == info.thread_blocks - 1u) {
        let final_size = info.size - partid * PART_SIZE;
        for (var i = builtinsNonuniform.lid.x; i < final_size; i += BLOCK_DIM) {
          var whi = atomicLoad(&wg_warpHist[i]);
          alt[wg_localHist[(whi >> info.shift) & RADIX_MASK] + i] = whi;
        }
      }
    }`;
    return kernel;
  };

  finalizeRuntimeParameters() {
    this.RADIX = 256;
    this.RADIX_BITS = 8;
    this.KEY_BITS = 32;
    this.SORT_PASSES = this.KEY_BITS / this.RADIX_BITS;
  }
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "global_hist",
        bufferTypes: [
          [
            "read-only-storage",
            "storage",
            "uniform",
            "storage",
            "storage",
            "storage",
          ],
        ],
        bindings: [
          [
            "inputBuffer",
            "outputBuffer",
            "scanParameters",
            "scanBump",
            "spine",
            "misc",
          ],
        ],
        label: `Thomas Smith's scan (${this.type}) with decoupled lookback/decoupled fallback [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return this.getSimpleDispatchGeometry();
        },
      }),
    ];
  }
}

const SortOneSweepRegressionParams = {
  inputLength: range(12, 25).map((i) => 2 ** i),
  datatype: ["u32"],
  disableSubgroups: [false],
};

export const SortOneSweepRegressionSuite = new BaseTestSuite({
  category: "sort",
  testSuite: "onesweep",
  trials: 2,
  params: SortOneSweepRegressionParams,
  primitive: OneSweepSort,
});
