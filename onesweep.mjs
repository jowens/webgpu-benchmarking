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
    this.additionalKnownBuffers = [
      "sortParameters",
      "sortBump",
      "hist",
      "passHist",
      "err",
    ];
    for (const knownBuffer of this.additionalKnownBuffers) {
      this.knownBuffers.push(knownBuffer);
    }
  }

  sortOneSweepWGSL = () => {
    let kernel = /* wgsl */ `
    /* enables subgroups ONLY IF it was requested when creating device */
    /* WGSL requires this declaration is first in the shader */
    ${this.fnDeclarations.enableSubgroupsIfAppropriate}
    struct SortParameters {
        size: u32,
        shift: u32,
        thread_blocks: u32,
        seed: u32,
    };

    @group(0) @binding(0)
    var<storage, read> keysIn: array<u32>;

    @group(0) @binding(1)
    var<storage, read_write> keysOut: array<u32>;

    @group(0) @binding(2)
    var<storage, read> payloadIn: array<u32>;

    @group(0) @binding(3)
    var<storage, read_write> payloadOut: array<u32>;

    @group(0) @binding(4)
    var<uniform> sortParameters : SortParameters;

    @group(0) @binding(5)
    var<storage, read_write> sortBump: array<atomic<u32>>;

    @group(0) @binding(6)
    var<storage, read_write> hist: array<atomic<u32>>;

    @group(0) @binding(7)
    var<storage, read_write> passHist: array<atomic<u32>>;

    const SORT_PASSES = 4u;
    const BLOCK_DIM = 256u;
    const MIN_SUBGROUP_SIZE = 4u;
    const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

    const FLAG_NOT_READY = 0u;
    const FLAG_REDUCTION = 1u;
    const FLAG_INCLUSIVE = 2u;
    const FLAG_MASK = 3u;

    /** NOTE kernel code below requires that RADIX is multiple of subgroup size
     * there's a diagnostic that asserts subgroup uniformity that depends on this
     */
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
    ${this.fnDeclarations.subgroupZero}

    /**
     * Input: global keysIn[]
     * Output: global hist[1024]
     * Action: Construct local histogram(s) of global keysIn[] input, add
     *   those histograms into global hist[] histogram
     * Structure of wg_globalHist array:
     * - Each group of 64 threads writes into a different segment
     * - Segment 0 (threads 0-63) histograms its inputs:
     *   - [0, 256) has 256 buckets that count keysIn[7:0]
     *   - [256, 512) ...                      keysIn[15:8]
     *   - [512, 768) ...                      keysIn[23:16]
     *   - [768, 1024) ...                     keysIn[31:24]
     * - Segment 1 (threads 64-127) writes into [1024, 2048)
     */
    @compute @workgroup_size(REDUCE_BLOCK_DIM, 1, 1)
    fn global_hist(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lidx / sgsz;  //Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

      // Reset histogram to zero
      // this may be unnecessary in WGSL (vars are zero-initialized)
      for (var i = builtinsNonuniform.lidx; i < REDUCE_HIST_SIZE; i += REDUCE_BLOCK_DIM) {
        atomicStore(&wg_globalHist[i], 0u);
      }
      workgroupBarrier();

      // 64 threads: 1 histogram in shared memory
      let radix_shift = sortParameters.shift;
      let hist_offset = builtinsNonuniform.lidx / 64u * ALL_RADIX;
      {
        /* not last workgroup */
        var i = builtinsNonuniform.lidx + builtinsUniform.wgid.x * REDUCE_PART_SIZE;
        if (builtinsUniform.wgid.x < sortParameters.thread_blocks - 1) {
          for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
            let key = keysIn[i];
            atomicAdd(&wg_globalHist[(key & RADIX_MASK) + hist_offset], 1u);
            atomicAdd(&wg_globalHist[((key >> 8u) & RADIX_MASK) + hist_offset + 256u], 1u);
            atomicAdd(&wg_globalHist[((key >> 16u) & RADIX_MASK) + hist_offset + 512u], 1u);
            atomicAdd(&wg_globalHist[((key >> 24u) & RADIX_MASK) + hist_offset + 768u], 1u);
            i += REDUCE_BLOCK_DIM;
          }
        }

        /* last workgroup */
        if (builtinsUniform.wgid.x == sortParameters.thread_blocks - 1) {
          for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
            if (i < sortParameters.size) {
              let key = keysIn[i];
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

      /* add all the segments together and add result to global hist[] array */
      for (var i = builtinsNonuniform.lidx; i < RADIX; i += REDUCE_BLOCK_DIM) {
        atomicAdd(&hist[i], atomicLoad(&wg_globalHist[i]) + atomicLoad(&wg_globalHist[i + ALL_RADIX]));
        atomicAdd(&hist[i + 256u], atomicLoad(&wg_globalHist[i + 256u]) + atomicLoad(&wg_globalHist[i + 256u + ALL_RADIX]));
        atomicAdd(&hist[i + 512u], atomicLoad(&wg_globalHist[i + 512u]) + atomicLoad(&wg_globalHist[i + 512u + ALL_RADIX]));
        atomicAdd(&hist[i + 768u], atomicLoad(&wg_globalHist[i + 768u]) + atomicLoad(&wg_globalHist[i + 768u + ALL_RADIX]));
      }
    }

    // Assumes block dim 256
    const SCAN_MEM_SIZE = RADIX / MIN_SUBGROUP_SIZE;
    var<workgroup> wg_scan: array<u32, SCAN_MEM_SIZE>;

    ${this.fnDeclarations.commonDefinitions}

    /**
     * Input: global hist[1024]
     *   - [0, 256) has 256 buckets that count keysIn[7:0], assigned to wkgp 0
     *   - [256, 512) ...                      keysIn[15:8], ...             1
     *   - [512, 768) ...                      keysIn[23:16] ...             2
     *   - [768, 1024) ...                     keysIn[31:24] ...             3
     * Output: global passHist[]
     * Launch parameters: SORT_PASSES workgroups (32b keys / 8b radix = 4 wkgps)
     * Action: Compute the global bin offset for every digit in each digit place.
     *         Each workgroup computes a different digit place.
     */
    @compute @workgroup_size(BLOCK_DIM, 1, 1)
    fn onesweep_scan(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lidx / sgsz;  // Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec
      /* seems like RADIX must equal BLOCK_DIM here */
      let index = builtinsNonuniform.lidx + builtinsUniform.wgid.x * RADIX;
      let scan = atomicLoad(&hist[index]);
      let red = subgroupAdd(scan);
      if (builtinsNonuniform.sgid == 0u) {
        wg_scan[sid] = red;
      }
      /* wg_scan[sid] contains sum of histogram buckets across the subgroup */
      workgroupBarrier();

      // Non-divergent subgroup agnostic inclusive scan across subgroup reductions (on wg_scan[])
      {
        var offset0 = 0u;
        var offset1 = 0u;
        let lane_log = u32(countTrailingZeros(sgsz));
        let spine_size = BLOCK_DIM >> lane_log;
        let aligned_size = 1u << ((u32(countTrailingZeros(spine_size)) + lane_log - 1u) / lane_log * lane_log);
        for (var j = sgsz; j <= aligned_size; j <<= lane_log){
          let i0 = ((builtinsNonuniform.lidx + offset0) << offset1) - select(0u, 1u, j != sgsz);
          let pred0 = i0 < spine_size;
          let t0 = subgroupInclusiveAdd(select(0u, wg_scan[i0], pred0));
          if (pred0){
            wg_scan[i0] = t0;
          }
          workgroupBarrier();

          if (j != sgsz) {
            let rshift = j >> lane_log;
            let i1 = builtinsNonuniform.lidx + rshift;
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

      /** passHist[] now contains the global bin offset for every digit in each digit place,
       * tagged with FLAG_INCLUSIVE */
      let pass_index = builtinsNonuniform.lidx + sortParameters.thread_blocks * builtinsUniform.wgid.x * RADIX;
      atomicStore(&passHist[pass_index], ((subgroupExclusiveAdd(scan) +
        select(0u, wg_scan[sid - 1u], builtinsNonuniform.lidx >= sgsz)) << 2u) | FLAG_INCLUSIVE);
    }

    var<workgroup> wg_warpHist: array<atomic<u32>, PART_SIZE>;
    var<workgroup> wg_localHist: array<u32, RADIX>;
    var<workgroup> wg_broadcast: u32;

    /* warp-level multisplit function */
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

    /**
     * * FIX
     * Input: global keysIn[]
     * Output: global keysOut[]
     * Action:
     */
    /* this is called 4 times, once for each digit-place. sortParameters.shift is {0,8,16,24}. */
    @compute @workgroup_size(BLOCK_DIM, 1, 1)
    fn onesweep_pass(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lidx / sgsz;  // Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

      /* set all of wg_warpHist to 0 */
      let warp_hists_size = clamp(BLOCK_DIM / sgsz * RADIX, 0u, PART_SIZE);
      for (var i = builtinsNonuniform.lidx; i < warp_hists_size; i += BLOCK_DIM) {
        atomicStore(&wg_warpHist[i], 0u);
      }

      /* get unique ID for my workgroup (stored in partid) */
      if (builtinsNonuniform.lidx == 0u) {
        wg_broadcast = atomicAdd(&sortBump[sortParameters.shift >> 3u], 1u);
      }
      let partid = workgroupUniformLoad(&wg_broadcast);

      /** copy from global keysIn[] into local keys[]
       * note this copy is strided (thread i gets i, i+sgsz, i+2*sgsz, ...)
       */
      var keys = array<u32, KEYS_PER_THREAD>();
      {
        let dev_offset = partid * PART_SIZE;
        let s_offset = sid * sgsz * KEYS_PER_THREAD;
        var i = builtinsNonuniform.sgid + s_offset + dev_offset;
        if (partid < sortParameters.thread_blocks - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            keys[k] = keysIn[i];
            i += sgsz;
          }
        }

        if (partid == sortParameters.thread_blocks - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            /* warning: u32-specific */
            keys[k] = select(0xffffffffu, keysIn[i], i < sortParameters.size);
            i += sgsz;
          }
        }
      }

      /** each key gets an offset for future placement
       * Note the call to WLMS changes wg_warpHist
       */
      var offsets = array<u32, KEYS_PER_THREAD>();
      {
        let shift = sortParameters.shift;
        let lane_mask_lt = (1u << builtinsNonuniform.sgid) - 1u;
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          offsets[k] = WLMS(keys[k], shift, builtinsNonuniform.sgid, sgsz, lane_mask_lt, s_offset);
        }
      }
      workgroupBarrier();

      var local_reduction = 0u;
      /* diagnostic is UNSAFE: this relies on RADIX being a multiple of (or equal to) subgroup size */
      @diagnostic(off, subgroup_uniformity)
      if (builtinsNonuniform.lidx < RADIX) {
        local_reduction = atomicLoad(&wg_warpHist[builtinsNonuniform.lidx]);
        for (var i = builtinsNonuniform.lidx + RADIX; i < warp_hists_size; i += RADIX) {
          local_reduction += atomicLoad(&wg_warpHist[i]);
          atomicStore(&wg_warpHist[i], local_reduction - atomicLoad(&wg_warpHist[i]));
        }

        if (partid < sortParameters.thread_blocks - 1u) {
          let pass_index = builtinsNonuniform.lidx + (partid + 1u) * RADIX +
            sortParameters.thread_blocks * (sortParameters.shift >> 3u) * RADIX;
          atomicStore(&passHist[pass_index], (local_reduction << 2u) | FLAG_REDUCTION);
        }

        let lane_mask = sgsz - 1u;
        let circular_lane_shift = (builtinsNonuniform.sgid + lane_mask) & lane_mask;
        let t = subgroupInclusiveAdd(local_reduction);
        wg_localHist[builtinsNonuniform.lidx] = subgroupShuffle(t, circular_lane_shift);
      }
      workgroupBarrier();

      @diagnostic(off, subgroup_uniformity)
      /* diagnostic off because this code is "am I subgroup 0" */
      if (isSubgroupZero(builtinsNonuniform.lidx, sgsz)) {
        let pred = builtinsNonuniform.lidx < RADIX / sgsz;
        let t = subgroupExclusiveAdd(select(0u, wg_localHist[builtinsNonuniform.lidx * sgsz], pred));
        if (pred) {
          wg_localHist[builtinsNonuniform.lidx * sgsz] = t;
        }
      }
      workgroupBarrier();

      if (builtinsNonuniform.lidx < RADIX && builtinsNonuniform.sgid != 0u) {
        wg_localHist[builtinsNonuniform.lidx] += wg_localHist[builtinsNonuniform.lidx / sgsz * sgsz];
      }
      workgroupBarrier();

      if (builtinsNonuniform.lidx >= sgsz) {
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          let t = (keys[k] >> sortParameters.shift) & RADIX_MASK;
          offsets[k] += wg_localHist[t] + atomicLoad(&wg_warpHist[t + s_offset]);
        }
      } else {
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          offsets[k] += wg_localHist[(keys[k] >> sortParameters.shift) & RADIX_MASK];
        }
      }
      workgroupBarrier();

      for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
        atomicStore(&wg_warpHist[offsets[k]], keys[k]);
      }

      if (builtinsNonuniform.lidx < RADIX) {
        var prev_reduction = 0u;
        var lookbackid = partid;
        let part_offset = sortParameters.thread_blocks * (sortParameters.shift >> 3u) * RADIX;
        while (true) {
          let flagPayload = atomicLoad(&passHist[builtinsNonuniform.lidx + part_offset + lookbackid * RADIX]);
          if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
            prev_reduction += flagPayload >> 2u;
            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
              if (partid < sortParameters.thread_blocks - 1u) {
                atomicStore(&passHist[builtinsNonuniform.lidx + part_offset + (partid + 1u) * RADIX],
                  ((prev_reduction + local_reduction) << 2u) | FLAG_INCLUSIVE);
              }
              wg_localHist[builtinsNonuniform.lidx] = prev_reduction - wg_localHist[builtinsNonuniform.lidx];
              break;
            } else {
              lookbackid -= 1u;
            }
          }
        }
      }
      workgroupBarrier();

      if (partid < sortParameters.thread_blocks - 1u) {
        var i = builtinsNonuniform.lidx;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          var whi = atomicLoad(&wg_warpHist[i]);
          keysOut[wg_localHist[(whi >> sortParameters.shift) & RADIX_MASK] + i] = whi;
          i += BLOCK_DIM;
        }
      }

      if (partid == sortParameters.thread_blocks - 1u) {
        let final_size = sortParameters.size - partid * PART_SIZE;
        for (var i = builtinsNonuniform.lidx; i < final_size; i += BLOCK_DIM) {
          var whi = atomicLoad(&wg_warpHist[i]);
          keysOut[wg_localHist[(whi >> sortParameters.shift) & RADIX_MASK] + i] = whi;
        }
      }
    }`;
    return kernel;
  };

  finalizeRuntimeParameters() {
    const inputSize = this.getBuffer("keysIn").size; // bytes
    const inputLength = inputSize / 4; /* 4 is size of key */
    this.RADIX = 256;
    this.RADIX_BITS = 8;
    this.KEY_BITS = 32;
    this.SORT_PASSES = this.KEY_BITS / this.RADIX_BITS;
    this.workgroupCount = 0;
    // sortParameters is size, shift, thread_blocks, seed [all u32]
    this.sortParameters = new Uint32Array([
      inputLength,
      0 /* each pass: {0,8,16,24} */,
      this.workgroupCount,
      0 /* currently unused */,
    ]);
    this.sortBumpSize = 4 * 4; // size: (4usize * std::mem::size_of::<u32>())
    this.histSize = this.SORT_PASSES * this.RADIX * 4; // (((SORT_PASSES * RADIX) as usize) * std::mem::size_of::<u32>())
    this.passHistSize = this.workgroupCount * this.histSize; // ((thread_blocks * (RADIX * SORT_PASSES) as usize) * std::mem::size_of::<u32>())
  }
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "sortParameters",
        size: this.sortParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        populateWith: this.sortParameters,
      }),
      new AllocateBuffer({
        label: "sortBump",
        size: this.sortBumpSize,
      }),
      new AllocateBuffer({
        label: "hist",
        size: this.histSize,
      }),
      new AllocateBuffer({
        label: "passHist",
        size: this.passHistSize,
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "global_hist",
        bufferTypes: [
          [
            "read-only-storage",
            "storage",
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
            "keysIn",
            "keysOut",
            "payloadIn",
            "payloadOut",
            "sortParameters",
            "sortBump",
            "hist",
            "passHist",
          ],
        ],
        label: `OneSweep sort (${this.type}) with decoupled lookback/decoupled fallback [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return this.getSimpleDispatchGeometry();
        },
      }),
    ];
  }
}

/* fn set_compute_pass(
  pass_index: u32,
  query: &wgpu::QuerySet,
  cs: &ComputeShader,
  com_encoder: &mut wgpu::CommandEncoder,
  thread_blocks: u32,
  timestamp_offset: u32,


        set_compute_pass(
            0u32,
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.global_hist,
            com_encoder,
            tester.reduce_thread_blocks,
            0u32,
        );
        set_compute_pass(
            0u32,
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.onesweep_scan,
            com_encoder,
            SORT_PASSES,
            2u32,
        );

        for i in 0u32..SORT_PASSES {
            if i != 0u32 {
                update_info(i, tester, com_encoder);
            }
            set_compute_pass(
                i,
                &tester.gpu_context.query_set,
                &tester.gpu_shaders.onesweep_pass,
                com_encoder,
                tester.pass_thread_blocks,
                i * 2u32 + 4u32,
            );
        }*/

const SortOneSweepRegressionParams = {
  inputLength: range(12, 25).map((i) => 2 ** i),
  datatype: ["u32"],
  type: ["keysonly" /* "keyvalue", */],
  disableSubgroups: [false],
};

export const SortOneSweepRegressionSuite = new BaseTestSuite({
  category: "sort",
  testSuite: "onesweep",
  trials: 2,
  params: SortOneSweepRegressionParams,
  primitive: OneSweepSort,
});
