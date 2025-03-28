/** Partially based on:
 * Thomas Smith (2024)
 * MIT License
 * https://github.com/b0nes164/GPUSorting
 */

import { range } from "./util.mjs";
import { Kernel, AllocateBuffer, WriteGPUBuffer } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { divRoundUp } from "./util.mjs";
import { BaseSort } from "./sort.mjs";

export class OneSweepSort extends BaseSort {
  constructor(args) {
    super(args);

    this.knownBuffers = [
      "keysInOut",
      "keysTemp",
      "payloadInOut",
      "payloadTemp",
      "sortParameters",
      "sortBump",
      "hist",
      "passHist",
      "err",
    ];

    for (const knownBuffer of this.knownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
        delete this[knownBuffer]; // let's make sure it's in one place only
      }
    }
  }

  get bytesTransferred() {
    /* TODO: make this work for key-payload also */
    return this.getBuffer("keysInOut").size * 2;
  }

  sortOneSweepWGSL = () => {
    /** Big picture OneSweep algorithm.
     * We wish to compute a permutation of all keys. This permutation orders
     *   the keys by the value of a particular digit in that key. We loop through
     *   the digits starting with the least significant digit. For concreteness,
     *   the code is focused on 32b keys and 4 passes, each focusing on one 8-bit
     *   digit. With 8-bit digits, there are 256 digit values, so on each pass,
     *   we classify each key into one of 256 buckets (radix 256). We perform a
     *   permutation with each sorting pass.
     *
     * For this permutation, we have to compute a destination address for each
     *   key. That address is the sum of:
     * (1) The number of keys that fall into a bucket before my bucket
     * (2) The number of keys that are in my bucket and are processed by a previous
     *    workgroup
     * (3) The number of keys that are in my bucket and are processed earlier than
     *    my key within my workgroup
     *
     * We compute (1) by constructing a global histogram of all bucket sizes
     *   (kernel: global_hist, which constructs a separate histogram for each
     *   digit) and then running exclusive-sum-scan on these histograms (kernel:
     *   onesweep_scan).
     * We compute (2) by first computing the number of keys per bucket within my
     *   histogram and then using a chained sum scan to retrieve the sum of the
     *   sizes of the previous workgroup's buckets (kernel: onesweep_pass). We
     *   add in (1) at the start of this chained scan, so we actually chained-scan
     *   (1) + (2). The chained scan requires lookback to look at the results of
     *   this scan, and if the hardware does not offer forward progress guarantees,
     *   the chained scan also requires fallback to redundantly compute this value.
     * We compute (3) in (kernel: onesweep_pass) as well, but it is workgroup-
     *   local only; it does not participate in the chained-scan.
     *
     * Given this computed address per key, we could directly scatter each key
     *   to its location in global memory. However, to improve memory coaclescing,
     *   we first write keys into workgroup memory and scatter from there. This puts
     *   neighboring keys next to each other in workgroup memory and significantly
     *   improves the throughput of the global scatter.
     */

    let kernel = /* wgsl */ `
    /* enables subgroups ONLY IF it was requested when creating device */
    /* WGSL requires this declaration is first in the shader */
    ${this.fnDeclarations.enableSubgroupsIfAppropriate}
    struct SortParameters {
        length: u32,
        shift: u32,
        thread_blocks: u32, // number of tiles in the onesweep_pass kernel
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
    var<storage, read_write> sortBump: array<atomic<u32>>;

    @group(0) @binding(5)
    var<storage, read_write> hist: array<atomic<u32>>;
    /**
     * hist keeps NUM_PASSES histograms, one per digit, each having RADIX entries
     * Structure of hist:
     *   hist[  0,  256) has 256 buckets that count keysIn[7:0]
     *   hist[256,  512) has 256 buckets that count keysIn[15:8]
     *   hist[512,  768) has 256 buckets that count keysIn[23:16]
     *   hist[768, 1024) has 256 buckets that count keysIn[31:24]
     */

    @group(0) @binding(6)
    var<storage, read_write> passHist: array<atomic<u32>>;
    /**
     * Structure of passHist:
     * (note we don't store anything for the last workgroup
     *  because no other workgroup needs it)
     * - We have NUM_PASSES segments, one per digit-place
     * - Each segment has:
     *   - One global histogram for its digit-place
     *   - One local histogram for each thread block. We chained-scan
     *     through these histograms.
     * - We compute the global histogram in onesweep_scan
     *   and tag it with FLAG_INCLUSIVE.
     * Why is it structured like this? In chained-scan, we look at
     *   the previous workgroup's result. Workgroup 0 instead looks
     *   back at the _global_ histogram, which it must add into the
     *   per-key address anyway. Thus the global histogram is placed
     *   in memory directly before workgroup 0's local histogram.
     *   Control is simpler because we need not specialize workgroup 0.
     * Consequence: When we start looking back, we look back at our
     *   own partition ID, because that corresponds to the block
     *   written by the previous workgroup.
     *
     * Global Hist Post Scan*** SORT PASS_0:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * Workgroup 0 SORT_PASS_0:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * ... (repeat up to Workgroup last-1)
     *
     * Global Hist Post Scan*** SORT PASS_1:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * Workgroup 0 SORT_PASS_1:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * ...
     * Global Hist Post Scan*** SORT PASS_2:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * Workgroup 0 SORT_PASS_2:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * ...
     * Global Hist Post Scan*** SORT PASS_3:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     * Workgroup 0 SORT_PASS_3:
     * +------------------------------------------+
     * | Bin  0 | Bin  1 | Bin  2 | ... | Bin 255 |
     * |   ?    |   ?    |   ?    | ... |   ?     |
     * +------------------------------------------+
     */

    @group(1) @binding(0)
    var<uniform> sortParameters : SortParameters;

    const SORT_PASSES = 4u;
    const BLOCK_DIM = 256u;
    const MIN_SUBGROUP_SIZE = 4u;
    const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

    const FLAG_NOT_READY = 0u << 30;
    const FLAG_REDUCTION = 1u << 30;
    const FLAG_INCLUSIVE = 2u << 30;
    const FLAG_MASK = 3u << 30;

    const MAX_SPIN_COUNT = 4u;
    const LOCKED = 1u;
    const UNLOCKED = 0u;

    /** NOTE kernel code below requires that RADIX is multiple of subgroup size
     * there's a diagnostic that asserts subgroup uniformity that depends on this
     *
     * Limitation: Keys must be no larger than 30 bits. "Doing the split here would
     * be wicked. No idea how the performance will be affected (probably bad). For
     * reference CUB also throws in the towel and says '30 bits is fine'":
     *   "The layout of a single counter is depicted in Figure 6. We use the two
     *    higher-order bits to store status, with the remaining 30 bits storing
     *    value."
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
     * Output: global hist[SORT_PASSES * RADIX = 1024]
     * Configuration:
     *   - REDUCE_BLOCK_DIM (128) threads per workgroup
     *   - Each thread handles REDUCE_KEYS_PER_THREAD (30)
     *   - We launch enough thread blocks to cover the entire input
     *     (basically, ceil(size of keysIn / (REDUCE_BLOCK_DIM * REDUCE_KEYS_PER_THREAD)))
     * Action: Construct local histogram(s) of global keysIn[] input, sum
     *   those histograms into global hist[] histogram
     * Structure of wg_globalHist array:
     * - Each group of 64 threads writes into a different segment
     * - Segment 0 (threads 0-63) histograms its inputs:
     *   - wg_globalHist[0, 256) has 256 buckets that count keysIn[7:0]
     *   -              [256, 512) ...                      keysIn[15:8]
     *   -              [512, 768) ...                      keysIn[23:16]
     *   -              [768, 1024) ...                     keysIn[31:24]
     * - Segment 1 (threads 64-127) writes into wg_globalHist[1024, 2048)
     * - Final step adds the two segments together and atomic-adds them to hist[]
     * - hist[0:1024] has the same structure as wg_globalHist[0:1024]
     */
    @compute @workgroup_size(REDUCE_BLOCK_DIM, 1, 1)
    fn global_hist(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lidx / sgsz;  //Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

      // Reset histogram to zero
      // this may be unnecessary in WGSL? (vars are zero-initialized)
      for (var i = builtinsNonuniform.lidx; i < REDUCE_HIST_SIZE; i += REDUCE_BLOCK_DIM) {
        atomicStore(&wg_globalHist[i], 0u);
      }
      workgroupBarrier();

      // 64 threads: 1 histogram in shared memory
      // segment 0 (threads [0,63]) -> wg_globalHist[0:1024),      hist_offset = 0
      // segment 1 (threads [64,127]) -> wg_globalHist[1024,2048), hist_offset = 1024
      let hist_offset = builtinsNonuniform.lidx / 64u * ALL_RADIX;
      {
        var i = builtinsNonuniform.lidx + builtinsUniform.wgid.x * REDUCE_PART_SIZE;
        /* not last workgroup */
        /* assumes nwg is [x, 1, 1] */
        if (builtinsUniform.wgid.x < builtinsUniform.nwg.x - 1) {
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
        if (builtinsUniform.wgid.x == builtinsUniform.nwg.x - 1) {
          for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
            if (i < arrayLength(&keysIn)) {
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
        /* first loop through: i in [0,128); second loop: i in [128,256) */
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
     * Input: global hist[4*256]
     *   - [0, 256) has 256 buckets that count keysIn[7:0], assigned to wkgp 0
     *   - [256, 512) ...                      keysIn[15:8], ...             1
     *   - [512, 768) ...                      keysIn[23:16] ...             2
     *   - [768, 1024) ...                     keysIn[31:24] ...             3
     * Output: global passHist[4*256, plus more that we're not worried about yet]
     *   - see diagram of how passHist is laid out where it is allocated at top of file
     *   - passHist has 4 segments (bits = {7:0, 15:8, 23:16, 31:24}) corresponding
     *     to keysIn[bits]), each of which has:
     *     - 1 256-element global offset: Exclusive scan of corresponding
     *       256-element block of hist[]
     *     - N * 256-element local offset, where N is the number of thread blocks
     *       in global_hist. These regions are completely unused in this kernel.
     *   - The global and local offsets are all in one memory region simply so we
     *     don't have to bind multiple memory regions
     *   - passHist stores 30b values that are tagged (in the 2 MSBs) with FLAG_*
     * Launch parameters:
     *   - SORT_PASSES workgroups (32b keys / 8b radix = 4 wkgps)
     *   - Each workgroup has BLOCK_DIM threads == RADIX
     * Action: Compute the global bin offset for every digit in each digit place.
     *         Each workgroup computes a different digit place.
     *         This is just an exclusive prefix-sum of each digit-place's hist bucket.
     *         Also tags each global-bin-offset entry in that prefix-sum with
     *           INCLUSIVE_FLAG in preparation for the scan in onesweep_pass.
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

      /** after the store below, passHist[] contains the global bin offset for every digit
       *   in each digit place, tagged with FLAG_INCLUSIVE. ("This makes the logic later
       *   on a little easier because we know for sure the inclusive flag exists.")
       * In this kernel, passHist[] only stores into the Global Hist Post Scan tables (see
       *   top). We don't use the other segments.
       * sortParameters.thread_blocks is the expansion factor that leaves room for
       *   all the local per-thread-block offsets. We will run chained-scan later on these
       *   local per-thread-block offsets.
       */
      let pass_index = builtinsNonuniform.lidx + sortParameters.thread_blocks * builtinsUniform.wgid.x * RADIX;
      atomicStore(&passHist[pass_index], ((subgroupExclusiveAdd(scan) +
        select(0u, wg_scan[sid - 1u], builtinsNonuniform.lidx >= sgsz)) & ~FLAG_MASK) | FLAG_INCLUSIVE);
    }

    var<workgroup> wg_warpHist: array<atomic<u32>, PART_SIZE>; /* KEYS_PER_THREAD * BLOCK_DIM */
    /* wg_warpHist has two roles:
     * 1) Initially, for the WLMS phase, each subgroup requires its own, radixDigit-sized portion
     *    of the workgroup memory to histogram with. That's how you get subgroupID * RADIX_DIGIT.
     *    Idiomatically: "Histogramming to my subgroup's histogram in workgroup memory."
     * 2) After the local scatter locations have been computed, we scatter the keys/values
     *    into workgroup memory. Workgroup memory scattering significantly improves the speed of the
     *    global scattering, because the keys/values are much more likely to share the same
     *    cache-line. Super important.
     * Size: It needs to be large enough to hold all histograms across the whole workgroup.
     * Challenge: What if subgroup size is small? That requires a LOT of storage. So:
     * - 1) manually bit-pack the workgroup mem to u16.
     * - 2) We use a technique Thomas calls "serial warp histogramming," where warps share
     *      histograms forcing some operations to go in serial. (Beats losing occupancy though).
     * - In this case, we completely max out the workgroup mem, whatever the size is. So we
     *   know the total size of the histograms must be the partition tile size.
     * Given the above, the use of "min" to set warp_hists_size below is correct.
     */

    var<workgroup> wg_localHist: array<u32, RADIX>;
    var<workgroup> wg_broadcast_tile_id: u32;
    /* the next two could be folded into wg_warpHist but are separate for now */
    var<workgroup> wg_sgCompletionCount: atomic<u32>; /* have all our subgroups succeeded? */
    var<workgroup> wg_sgCompletionCount_nonatomic: u32; /* shadows wg_sgCompletionCount, necessary b/c no workgroupUniformAtomicLoad */
    var<workgroup> wg_incomplete: u32; /* did anyone see a thread that didn't complete? */
    var<workgroup> wg_fallback: array<atomic<u32>, RADIX>; /* we can probably find another place to put this that doesn't require a separate allocation */

    /* warp-level multisplit function */
    /* this appears to be limited to subgroup sizes <= 32 */
    /* more portable version, HLSL:
     * https://github.com/b0nes164/GPUSorting/blob/09a6081d964b682bbb58838b83ab75aebe7f4e05/GPUSortingD3D12/Shaders/SortCommon.hlsl#L400-L424
     * https://github.com/b0nes164/GPUSorting/blob/09a6081d964b682bbb58838b83ab75aebe7f4e05/GPUSortingD3D12/Shaders/SortCommon.hlsl#L359-L375
     * Thomas: "I should also note that in my HLSL implementation, and I definitely think
     * this is the way to do this, I ended splitting the implementation into a 'big subgroup size' 16+
     * and 'small subgroup size < 16'" ... "The defining issue, is that the number of histograms is
     * inversely proportional to the subgroup size. At that low of a subgroup size, you have to start
     * doing more exotic techniques to prevent losing occupancy. Because basically you want a shared
     * mem histogram per warp."
    */
    fn WLMS(key: u32, shift: u32, sgid: u32, sgsz: u32, lane_mask_lt: u32, s_offset: u32) -> u32 {
      var eq_mask = 0xffffffffu;
      /* loop through the bits in this digit place */
      for (var k = 0u; k < RADIX_LOG; k += 1u) {
        let curr_bit = 1u << (k + shift);
        let pred = (key & curr_bit) != 0u;
        let ballot = subgroupBallot(pred);
        eq_mask &= select(~ballot.x, ballot.x, pred);
      }
      /* eq_mask marks the threads that have the same digit as me ... */
      var out = countOneBits(eq_mask & lane_mask_lt);
      /* ... and out marks those threads in eq_mask whose ids are less than me */
      /* what is the largest-sgid thread in my subgroup that has the same digit as me? */
      let highest_rank_peer = sgsz - countLeadingZeros(eq_mask) - 1u;
      var pre_inc = 0u;
      if (sgid == highest_rank_peer) {
        /* add to wg_warpHist for my digit value: total number of threads with my digit value */
        /* since I'm the highest rank peer, that's the number of threads ranked below me with
         * my digit value, plus one for me */
        pre_inc = atomicAdd(&wg_warpHist[((key >> shift) & RADIX_MASK) + s_offset], out + 1u);
        /* pre_inc now reflects the number of threads seen to date that have MY digit
         * from previous calls to WLMS */
        /* wg_warpHist[s_offset:s_offset+RADIX] is now a per-digit histogram up to and including
         * all digits seen so far by this subgroup */
      }

      workgroupBarrier();
      /** out is a unique and densely packed index [0...n] for each thread with the same digit
       * out means "This is the nth instance of this digit I have seen so far in my warp"
       * it is the sum of "count of this digit for all previous subgroups" (pre_inc from
       * highest rank peer) and "number of threads below me in my subgroup with this digit"
       * */
      out += subgroupShuffle(pre_inc, highest_rank_peer);
      return out;
    }

    /**
     * Input: global keysIn[]
     * Output: global keysOut[]
     *   (input/output bindings switch on odd calls)
     * Launch parameters:
     * - Workgroup size: BLOCK_DIM == 256
     * - Each workgroup is responsible for PART_SIZE = KEYS_PER_THREAD (15) * BLOCK_DIM (256) keys
     * - Launch enough workgroups to cover all input keys (divRoundUp(inputLength, PART_SIZE))
     * Call this kernel 4 times, once for each digit-place.
     *   sortParameters.shift has to be set {0,8,16,24}.
     * Compute steps in this kernel:
     * - Load keys
     * - Warp level multisplit -> Make a per-warp histogram
     * - Exclusive scan / circular shift up the warp histograms
     * - Exclusive scan across the histogram total
     * - Update the in-register offsets so we can scatter to shared memory
     * - Lookback
     * - (Fallback)
     * - Scatter to shared memory
     * - Scatter from shared to global
     */
    @compute @workgroup_size(BLOCK_DIM, 1, 1)
    fn onesweep_pass(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      ${this.fnDeclarations.initializeSubgroupVars}

      let sid = builtinsNonuniform.lidx / sgsz;  // Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

      /** Q: how many RADIX-sized blocks do we allocate within wg_warpHist?
       * A: Minimum of KEYS_PER_THREAD and number of subgroups
       * The size of our histograms will typically be less than the size of the partition
       * tile. We only need to clear the shared mem for those hists, not the entire partition size.
       * The purpose of the warpHists variable is to hold the total size of histograms across the workgroup.
       */
      let num_subgroups = BLOCK_DIM / sgsz;
      let warp_hists_size = min(num_subgroups * RADIX, PART_SIZE);
      /* set the histogram portion of wg_warpHist to 0 */
      for (var i = builtinsNonuniform.lidx; i < warp_hists_size; i += BLOCK_DIM) {
        atomicStore(&wg_warpHist[i], 0u);
      }

      /* get unique ID for my workgroup (stored in partid) */
      if (builtinsNonuniform.lidx == 0u) {
        wg_broadcast_tile_id = atomicAdd(&sortBump[sortParameters.shift >> 3u], 1u);
        atomicStore(&wg_sgCompletionCount, 0);
        wg_sgCompletionCount_nonatomic = 0;
        wg_incomplete = 0;
      }
      let partid = workgroupUniformLoad(&wg_broadcast_tile_id);

      /** copy KEYS_PER_THREAD from global keysIn[] into local keys[]
       * Subgroups fetch a contiguous block of (sgsz * KEYS_PER_THREAD) keys
       * note this copy is strided (thread i gets i, i+sgsz, i+2*sgsz, ...)
       */
      var keys = array<u32, KEYS_PER_THREAD>();
      {
        let dev_offset = partid * PART_SIZE;
        let s_offset = sid * sgsz * KEYS_PER_THREAD;
        var i = builtinsNonuniform.sgid + s_offset + dev_offset;
        if (partid < builtinsUniform.nwg.x - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            keys[k] = keysIn[i];
            i += sgsz;
          }
        }

        if (partid == builtinsUniform.nwg.x - 1) {
          for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            /* warning: u32-specific */
            keys[k] = select(0xffffffffu, keysIn[i], i < arrayLength(&keysIn));
            i += sgsz;
          }
        }
      }

      /** We are computing a histogram per subgroup
       * Note the call to WLMS also populates wg_warpHist
       */
      var offsets = array<u32, KEYS_PER_THREAD>();
      {
        let shift = sortParameters.shift;
        let lane_mask_lt = (1u << builtinsNonuniform.sgid) - 1u;
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          /* offsets[k] reflects the number of digits seen by this subgroup
           * to date that match MY digit */
          offsets[k] = WLMS(keys[k], shift, builtinsNonuniform.sgid, sgsz, lane_mask_lt, s_offset);
        }
      }
      workgroupBarrier();

      var local_reduction = 0u;
      /* diagnostic is UNSAFE: this relies on
       * - RADIX being a multiple of (or equal to) subgroup size
       * - Workgroup size is 1D (it is)
       * - Subgroups are guaranteed to be laid out linearly within the workgroup (probably (?))
       */
      @diagnostic(off, subgroup_uniformity)
      if (builtinsNonuniform.lidx < RADIX) {
        /* activate only RADIX threads */
        /** let's look at the counts for thread i of wg_warpHist[i,i+RADIX,i+2*RADIX,...]
         * and call them [a,b,c,...]
         */
        local_reduction = atomicLoad(&wg_warpHist[builtinsNonuniform.lidx]);
        /* local_reduction now equals a */
        /* this for loop walks through the results of different subgroup histograms */
        for (var i = builtinsNonuniform.lidx + RADIX; i < warp_hists_size; i += RADIX) {
          /* FIX: this could be made simpler, it's an attempt to save registers that wasn't profitable */
          local_reduction += atomicLoad(&wg_warpHist[i]);
          atomicStore(&wg_warpHist[i], local_reduction - atomicLoad(&wg_warpHist[i]));
        }
        /* local_reduction for thread i now contains the total number of seen digits with value i */
        /* now wg_warpHist[i,i+RADIX,i+2*RADIX,...] contains [a,a,a+b,a+b+c,...] */
        /* this seems odd, but we'll use these values below when we do a circular shift */

        /* last workgroup does nothing because no other workgroup needs its data */
        if (partid < builtinsUniform.nwg.x - 1u) { /* not the last workgroup */
          /* Update the spine with local_reduction. The address within the spine is
           * the sum of which digit, which thread-block-local histogram within that
           * digit, and which of the RADIX buckets within that t-b-l histogram */
          /** For the purpose of the lookback, we need to post the total count per digit
           *    into the mega-spine.
           * However, for the correct shared memory offsets, we need to do that second,
           *    workgroup-wide exclusive scan over the total count per digit.
           * So posting the digit counts HAS to happen there.
          */
          let pass_index =
            builtinsUniform.nwg.x * (sortParameters.shift >> 3u) * RADIX + /* which digit */
            (partid + 1u) * RADIX + /* which thread-block-local histogram within that digit */
            builtinsNonuniform.lidx; /* which of the RADIX buckets within that t-b-l histogram */
          /* this next RADIX-sized store is what is being chain-scanned in this kernel */
          atomicStore(&passHist[pass_index], (local_reduction & ~FLAG_MASK) | FLAG_REDUCTION);
        }
        /**
         * First we scan "up" the histograms. Each thread does a inclusive circular shift
         * scan per digit, up the warp histograms, which at this point in the algorithm
         * contain the results from WLMS. Instead of rotating around the subgroup, we
         * rotate around the histograms. The histogram in shared memory at the subgroupID 0
         * position now contains the total count, per digit of the tile.
         */

        /* Once posted, we do a standard workgroup-wide exclusive scan over RADIX threads,
         * which starts here. */
        let lane_mask = sgsz - 1u;
        let circular_lane_shift = (builtinsNonuniform.sgid + lane_mask) & lane_mask;
        let t = subgroupInclusiveAdd(local_reduction);
        wg_localHist[builtinsNonuniform.lidx] = subgroupShuffle(t, circular_lane_shift);
      }
      workgroupBarrier();

      @diagnostic(off, subgroup_uniformity)
      /* diagnostic off because this code is "am I subgroup 0" and thus uniform */
      /* Warning: not size-agnostic */
      if (isSubgroupZero(builtinsNonuniform.lidx, sgsz)) {
        let pred = builtinsNonuniform.lidx < RADIX / sgsz; /* activate threads = number of subgroups */
        let t = subgroupExclusiveAdd(select(0u, wg_localHist[builtinsNonuniform.lidx * sgsz], pred));
        if (pred) {
          /* this populates the local spine */
          wg_localHist[builtinsNonuniform.lidx * sgsz] = t;
        }
      }
      workgroupBarrier();

      if (builtinsNonuniform.lidx < RADIX && builtinsNonuniform.sgid != 0u) {
        /* If I'm not the first thread in my subgroup, add the value from the local
         * spine to my local val. This completes the scan. */
        wg_localHist[builtinsNonuniform.lidx] += wg_localHist[builtinsNonuniform.lidx / sgsz * sgsz];
      }
      workgroupBarrier();
      /* end workgroup-wide exclusive scan */

      /**
       * Next: we update the per key offsets to get the correct scattering
       * locations into shared memory. Idiomatically that's: "Add my
       * subgroup-private result, to the result at my subgroup histogram, to the
       * result across the whole workgroup (which we stored in subgroup histogram 0)."
       * The trick here is that for sgid 0, the result at my subgroup histogram
       * is the result across the workgroup.
       */
      if (isSubgroupZero(builtinsNonuniform.lidx, sgsz)) {
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          offsets[k] += wg_localHist[(keys[k] >> sortParameters.shift) & RADIX_MASK];
        }
      } else { /* subgroup 1+ */
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          let t = (keys[k] >> sortParameters.shift) & RADIX_MASK;
          offsets[k] += wg_localHist[t] + atomicLoad(&wg_warpHist[t + s_offset]);
        }
      }
      /* we no longer need the data in wg_warpHist */
      workgroupBarrier();

      /** Warp histograms are no longer needed; let's now use warpHist
       * to store keys instead. Scatter keys into warpHist. */
      for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
        atomicStore(&wg_warpHist[offsets[k]], keys[k]);
      }

      /** Lookback/fallback strategy.
       *
       * For performance reasons, we would like to use the entire workgroup to look back.
       * However, any thread in the lookback may find a value that is NOT_READY. Our
       * lookback only succeeds if _every_ thread finds a ready value. So we must ensure
       * that all threads find a ready value before concluding that the lookback has
       * succeeded. If any of them are not successful, we (must) drop into fallback.
       *
       * How do we determine if they have succeeded? We keep track of completion both
       * per-thread (lookbackComplete) and per-subgroup (sgLookbackComplete).
       *
       * In workgroup memory, we use
       * - wg_sgCompletionCount: counts number of successful subgroups
       *   - wg_sgCompletionCount_nonatomic: shadows wg_sgCompletionCount
       * - wg_incomplete: did anyone see a thread that didn't complete?
       * both initialized to zero at the top of the kernel
       */

      {
        var lookbackAccumulate = 0u; /* we will update our spine with this value. No flags. */
        var lookbackComplete = select(true, false, builtinsNonuniform.lidx < RADIX); /* thread-local */
        var sgLookbackComplete = select(true, false, builtinsNonuniform.lidx < RADIX); /* uniform per workgroup */
        var lookbackid = partid; /* passHist is shifted by 1 so we are actually "looking back" to our own partid */
        let part_offset = builtinsUniform.nwg.x * (sortParameters.shift >> 3u) * RADIX;
        var lookbackIndex = (builtinsNonuniform.lidx & RADIX_MASK) + part_offset + lookbackid * RADIX; /* ptr into passHist */

        while (workgroupUniformLoad(&wg_sgCompletionCount_nonatomic) < num_subgroups) { /* loop until every subgroup is done */
          var spinCount = 0u;
          // Read the preceding histogram from the spine (stored in passHist)
          var flagPayload = 0u;
          /* turning off diagnostic for sgLookbackComplete is safe; it's only updated an entire subgroup at a time */
          @diagnostic(off, subgroup_uniformity)
          if (!sgLookbackComplete) {
            if (!lookbackComplete) {
              while (spinCount < MAX_SPIN_COUNT) { /* spin until we get something better than FLAG_NOT_READY */
                flagPayload = atomicLoad(&passHist[builtinsNonuniform.lidx + part_offset + lookbackid * RADIX]);
                if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) {
                  break;
                } else {
                  spinCount++;
                }
              }
              // Did we find anything with FLAG_NOT_READY? Subgroup thread 0 posts it.
              if (subgroupAny(spinCount == MAX_SPIN_COUNT) && (sgid == 0)) {
                wg_incomplete = 1; // possibly a race?
              }
            }
          }

          workgroupBarrier(); /* all subgroups reach this point before next check */
          var mustFallback = workgroupUniformLoad(&wg_incomplete);

          /* At least one thread saw FLAG_NOT_READY. Must fallback. */
          if (mustFallback != 0) {
            /* clear wg_fallback array ... */
            if (builtinsNonuniform.lidx < RADIX) {
              atomicStore(&wg_fallback[builtinsNonuniform.lidx], 0);
            }
            /* ... and reset wg_incomplete */
            if (builtinsNonuniform.lidx == 0) {
              wg_incomplete = 0;
            }
            workgroupBarrier();
            /* now let's build a local histogram of this tile in wg_fallback */
            /* fallbackStart and fallbackEnd bound the region of interest in keysIn[] */
            var fallbackStart = PART_SIZE * ((lookbackIndex >> RADIX_LOG) - builtinsUniform.nwg.x * (sortParameters.shift >> 3u) - 1);
            var fallbackEnd = PART_SIZE * ((lookbackIndex >> RADIX_LOG) - builtinsUniform.nwg.x * (sortParameters.shift >> 3u));
            for (var i = builtinsNonuniform.lidx + fallbackStart; i < fallbackEnd; i += BLOCK_DIM) {
              var key = keysIn[i];
              var digit = (key >> sortParameters.shift) & RADIX_MASK;
              atomicAdd(&wg_fallback[digit], 1u);
              /* if we're ever doing something other than u32, here's where to change that */
              /* see FallbackHistogram in SweepCommon.hlsl */
            }
            workgroupBarrier();
            /* Local histogram is complete in wg_fallback. Now let's use that to make progress. */
            var lookbackSpine = 0u;
            if (builtinsNonuniform.lidx < RADIX) {
              var myFBHistogramEntry = atomicLoad(&wg_fallback[builtinsNonuniform.lidx]);
              /* I would like to UPDATE the value at lookbackid that is already there, but only
               * if our computed fallback value is better. atomicMax will do this! Also store
               * what was already there in lookbackSpine */
              lookbackSpine = atomicMax(&passHist[builtinsNonuniform.lidx + part_offset + lookbackid * RADIX],
                                       (myFBHistogramEntry & ~FLAG_MASK) | FLAG_REDUCTION);
            }
            if (!lookbackComplete) {
              if ((lookbackSpine & FLAG_MASK) == FLAG_INCLUSIVE) { /** oh cool, someone else already
                                                                   * updated the spine to INCLUSIVE.
                                                                   * let's finish the lookback! */
                /* this is the end of our lookback */
                lookbackAccumulate += lookbackSpine & ~FLAG_MASK;
                if (partid < builtinsUniform.nwg.x - 1u) { /* not the last workgroup */
                 /** update MY spine entry, which has FLAG_REDUCTION already, thus when we add
                  * another FLAG_REDUCTION to it, we get FLAG_INCLUSIVE */
                  atomicAdd(&passHist[builtinsNonuniform.lidx + part_offset + (partid + 1u) * RADIX],
                            lookbackAccumulate | FLAG_REDUCTION);
                } /* last workgroup doesn't need to post to spine, no one needs that value */
                lookbackComplete = true;
              } else { /* use the fallback we just computed */
                lookbackAccumulate += atomicLoad(&wg_fallback[builtinsNonuniform.lidx]);
              }
            }
          } else { /* no fallback needed, all flags are either REDUCTION or INCLUSIVE */
            if (!lookbackComplete) {
              lookbackAccumulate += flagPayload & ~FLAG_MASK; /* accumulate no matter what */
              if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                if (partid < builtinsUniform.nwg.x - 1u) { /* not the last workgroup */
                  /** update my spine entry, which has FLAG_REDUCTION already, thus when we add
                   * another FLAG_REDUCTION to it, we get FLAG_INCLUSIVE */
                   atomicAdd(&passHist[builtinsNonuniform.lidx + part_offset + (partid + 1u) * RADIX],
                             lookbackAccumulate | FLAG_REDUCTION);
                } /* last workgroup doesn't need to post to spine, no one needs that value */
                lookbackComplete = true;
              } /* do nothing extra for FLAG_REDUCTION */
            }
          } /* end check if we need to fallback */
          lookbackIndex -= RADIX; // This workgroup looks back in lockstep
          lookbackid--;
          // Have all digits completed their lookbacks?
          /* turning off diagnostic for sgLookbackComplete is safe; it's only updated an entire subgroup at a time */
          @diagnostic(off, subgroup_uniformity)
          if (!sgLookbackComplete) {
            sgLookbackComplete = subgroupAll(lookbackComplete);
            if (sgLookbackComplete && (sgid == 0)) {
              /* tell the outermost loop "my subgroup is done" */
              atomicAdd(&wg_sgCompletionCount, 1);
            }
          }
          workgroupBarrier();
          if (builtinsNonuniform.lidx == 0) {
            wg_sgCompletionCount_nonatomic = atomicLoad(&wg_sgCompletionCount);
          }
          workgroupBarrier();
        } /* end if (wg_sgCompletionCount < subgroup_size) */

        // Post results into shared memory
        if (builtinsNonuniform.lidx < RADIX) {
          /** This curious-looking line is necessary to get the right scattering location in the next
            * code block. */
          wg_localHist[builtinsNonuniform.lidx] = lookbackAccumulate - wg_localHist[builtinsNonuniform.lidx];
        }
        workgroupBarrier();
      } /* end code block */

      /** Scatter keys from shared memory to global memory. Getting the correct scattering location
       * is dependent on the curious-looking line above. */
      if (partid < builtinsUniform.nwg.x - 1u) { /* not the last workgroup */
        var i = builtinsNonuniform.lidx;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
          var whi = atomicLoad(&wg_warpHist[i]);
          keysOut[wg_localHist[(whi >> sortParameters.shift) & RADIX_MASK] + i] = whi;
          i += BLOCK_DIM;
        }
      }

      if (partid == builtinsUniform.nwg.x - 1u) { /* last workgroup */
        let final_size = arrayLength(&keysIn) - partid * PART_SIZE;
        for (var i = builtinsNonuniform.lidx; i < final_size; i += BLOCK_DIM) {
          var whi = atomicLoad(&wg_warpHist[i]);
          keysOut[wg_localHist[(whi >> sortParameters.shift) & RADIX_MASK] + i] = whi;
        }
      }
    }`;
    return kernel;
  };

  finalizeRuntimeParameters() {
    const inputSize = this.getBuffer("keysInOut").size; // bytes
    this.inputLength = inputSize / 4; /* 4 is size of key */
    this.RADIX = 256;
    this.RADIX_BITS = 8;
    this.KEY_BITS = 32;
    this.SORT_PASSES = this.KEY_BITS / this.RADIX_BITS;

    /* the following better match what's in the shader! */
    this.BLOCK_DIM = 256;
    this.KEYS_PER_THREAD = 15;
    this.PART_SIZE = this.KEYS_PER_THREAD * this.BLOCK_DIM;
    this.REDUCE_BLOCK_DIM = 128;
    this.REDUCE_KEYS_PER_THREAD = 30;
    this.REDUCE_PART_SIZE = this.REDUCE_KEYS_PER_THREAD * this.REDUCE_BLOCK_DIM;
    /* end copy from shader */
    this.passWorkgroupCount = divRoundUp(this.inputLength, this.PART_SIZE);
    this.reduceWorkgroupCount = divRoundUp(
      this.inputLength,
      this.REDUCE_PART_SIZE
    );
    /* let's pack all uniform buffers (sortParameters) into one buffer
     * sortParameters[0]: for global_hist and onesweep_scan
     * sortParameters[1,2,3,4]: for onesweep_pass[0,1,2,3]
     * sortParameters is size, shift, thread_blocks, seed [all u32]
     *
     * In the below calculations, we are using _length_ (in elements)
     *   not _size_ (in bytes)
     */
    this.uniformBufferLength = 4; // 4 elements in uniformBuffer
    this.uniformBufferOffsetLength = Math.max(
      /* how much space per entry? BUT it better be at least minUniform... */
      this.uniformBufferLength,
      this.device.limits.minUniformBufferOffsetAlignment /
        Uint32Array.BYTES_PER_ELEMENT
    );
    this.uniformBufferOffsetSize =
      /* used in creating bindings below */
      this.uniformBufferOffsetLength * Uint32Array.BYTES_PER_ELEMENT;
    this.sortParameters = new Uint32Array(
      (this.SORT_PASSES + 1) * this.uniformBufferOffsetLength
    );
    for (var i = 0; i < this.SORT_PASSES + 1; i++) {
      const offset = i * this.uniformBufferOffsetLength;
      this.sortParameters[offset] = this.inputLength;
      this.sortParameters[offset + 1] = Math.max(i - 1, 0) * this.RADIX_BITS;
      this.sortParameters[offset + 2] =
        i == 0 ? this.reduceWorkgroupCount : this.passWorkgroupCount;
      this.sortParameters[offset + 3] = 0;
    }
    this.sortBumpSize = 4 * 4; // size: (4usize * std::mem::size_of::<u32>())
    this.histSize = this.SORT_PASSES * this.RADIX * 4; // (((SORT_PASSES * RADIX) as usize) * std::mem::size_of::<u32>())
    this.passHistSize = this.passWorkgroupCount * this.histSize; // ((pass_thread_blocks * (RADIX * SORT_PASSES) as usize) * std::mem::size_of::<u32>())
  }
  compute() {
    this.finalizeRuntimeParameters();
    const bufferTypes = [
      [
        "read-only-storage",
        "storage",
        "read-only-storage",
        "storage",
        "storage",
        "storage",
        "storage",
      ],
      ["uniform"],
    ];
    const bindings0Even = [
      "keysInOut",
      "keysTemp",
      "payloadInOut",
      "payloadTemp",
      "sortBump",
      "hist",
      "passHist",
    ];
    const bindings0Odd = [
      "keysTemp",
      "keysInOut",
      "payloadTemp",
      "payloadInOut",
      "sortBump",
      "hist",
      "passHist",
    ];
    const sortParameterBinding = new Array(this.SORT_PASSES + 1);
    for (var i = 0; i < sortParameterBinding.length; i++) {
      sortParameterBinding[i] = {
        buffer: "sortParameters",
        offset: i * this.uniformBufferOffsetSize,
        size: this.uniformBufferSize,
      };
    }
    return [
      new AllocateBuffer({
        label: "sortParameters",
        size: this.sortParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      new WriteGPUBuffer({
        label: "sortParameters",
        cpuSource: this.sortParameters,
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
        bufferTypes,
        bindings: [bindings0Even, [sortParameterBinding[0]]],
        label: `OneSweep sort (${this.type}) global_hist [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return [this.reduceWorkgroupCount];
        },
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "onesweep_scan",
        bufferTypes,
        bindings: [bindings0Even, [sortParameterBinding[0]]],
        label: `OneSweep sort (${this.type}) onesweep_scan [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return [this.SORT_PASSES];
        },
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "onesweep_pass",
        bufferTypes,
        bindings: [bindings0Even, [sortParameterBinding[1]]],
        label: `OneSweep sort (${this.type}) onesweep_pass shift 0 [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        logLaunchParameters: false,
        getDispatchGeometry: () => {
          return [this.passWorkgroupCount];
        },
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "onesweep_pass",
        bufferTypes,
        bindings: [bindings0Odd, [sortParameterBinding[2]]],
        label: `OneSweep sort (${this.type}) onesweep_pass digit 1 [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        logLaunchParameters: false,
        getDispatchGeometry: () => {
          return [this.passWorkgroupCount];
        },
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "onesweep_pass",
        bufferTypes,
        bindings: [bindings0Even, [sortParameterBinding[3]]],
        label: `OneSweep sort (${this.type}) onesweep_pass digit 2 [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        logLaunchParameters: false,
        getDispatchGeometry: () => {
          return [this.passWorkgroupCount];
        },
      }),
      new Kernel({
        kernel: this.sortOneSweepWGSL,
        entryPoint: "onesweep_pass",
        bufferTypes,
        bindings: [bindings0Odd, [sortParameterBinding[4]]],
        label: `OneSweep sort (${this.type}) onesweep_pass digit 3 [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: false,
        logLaunchParameters: false,
        getDispatchGeometry: () => {
          return [this.passWorkgroupCount];
        },
      }),
    ];
  }
  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc =
      args.inputKeys?.cpuBuffer ??
      args.inputKeys ??
      this.getBuffer("keysInOut").cpuBuffer;
    const memdest =
      args.outputKeys?.cpuBuffer ??
      args.outputKeys ??
      this.getBuffer("keysInOut").cpuBuffer;
    let referenceOutput;
    switch (args?.outputKeys?.label) {
      case "hist":
      case "passHist": {
        const hist = new Uint32Array(
          this.histSize / Uint32Array.BYTES_PER_ELEMENT
        );
        for (let i = 0; i < memsrc.length; i++) {
          for (let j = 0; j < this.SORT_PASSES; j++) {
            const histOffset = j * this.RADIX;
            const bucket =
              (memsrc[i] >>> (j * this.RADIX_BITS)) & (this.RADIX - 1);
            hist[histOffset + bucket]++;
          }
        }
        if (args?.outputKeys?.label === "hist") {
          referenceOutput = hist;
          break;
        }
        const passHist = new Uint32Array(
          this.passHistSize / Uint32Array.BYTES_PER_ELEMENT
        );
        const FLAG_INCLUSIVE = 2 << 30;
        /* divide into 4 pieces (4 passes), write only first RADIX (256) elements of each */
        for (let j = 0; j < this.SORT_PASSES; j++) {
          const histOffset = j * this.RADIX;
          const passHistOffset = j * (passHist.length / this.SORT_PASSES);
          let acc = 0;
          for (let bucket = 0; bucket < this.RADIX; bucket++) {
            passHist[passHistOffset + bucket] = acc | FLAG_INCLUSIVE;
            acc += hist[histOffset + bucket];
          }
        }
        if (args?.outputKeys?.label === "passHist") {
          referenceOutput = passHist;
          break;
        }
        break;
      }
      case undefined:
      default:
        referenceOutput = memsrc.slice().sort();
        break;
    }
    const isValidComparison = (args = {}) => {
      /* skip some comparisons if validation is not meaningful */
      switch (args.label) {
        case "passHist": {
          /** divide pastHist into SORT_PASSES equal parts, only the first RADIX
           * elements of each part are meaningful */
          const passHistLength =
            this.passHistSize / Uint32Array.BYTES_PER_ELEMENT;
          for (let j = 0; j < this.SORT_PASSES; j++) {
            const passHistOffset = j * (passHistLength / this.SORT_PASSES);
            if (
              args.i >= passHistOffset &&
              args.i < passHistOffset + this.RADIX
            ) {
              return true;
            }
          }
          return false;
        }
        default:
          return true;
      }
    };
    function validates(args) {
      return args.cpu == args.gpu;
    }
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors == 0) {
        break;
      }
      const compare = {
        cpu: referenceOutput[i],
        gpu: memdest[i],
        datatype: this.datatype,
        i,
        label: args?.outputKeys?.label,
      };
      if (isValidComparison(compare) && !validates(compare)) {
        returnString += `\nElement ${i}: expected ${
          referenceOutput[i]
        }, instead saw ${memdest[i]} (diff: ${Math.abs(
          (referenceOutput[i] - memdest[i]) / referenceOutput[i]
        )}).`;
        if (this.getBuffer("debugBuffer")) {
          returnString += ` debug[${i}] = ${
            this.getBuffer("debugBuffer").cpuBuffer[i]
          }.`;
        }
        if (this.getBuffer("debug2Buffer")) {
          returnString += ` debug2[${i}] = ${
            this.getBuffer("debug2Buffer").cpuBuffer[i]
          }.`;
        }
        allowedErrors--;
      }
    }
    if (returnString !== "") {
      console.log(
        this.label,
        this.type,
        this,
        "with input",
        memsrc,
        "should validate to (reference)",
        args?.outputKeys?.label ?? "",
        referenceOutput,
        "and actually validates to (GPU output)",
        memdest,
        this.getBuffer("debugBuffer") ? "\ndebugBuffer" : "",
        this.getBuffer("debugBuffer")
          ? this.getBuffer("debugBuffer").cpuBuffer
          : "",
        this.getBuffer("debug2Buffer") ? "\ndebug2Buffer" : "",
        this.getBuffer("debug2Buffer")
          ? this.getBuffer("debug2Buffer").cpuBuffer
          : ""
      );
    }
    return returnString;
  };
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
  inputLength: [2 ** 27],
  // inputLength: [2048, 4096],
  // inputLength: range(10, 27).map((i) => 2 ** i),
  datatype: ["u32"],
  type: ["keysonly" /* "keyvalue", */],
  disableSubgroups: [false],
};

export const SortOneSweepRegressionSuite = new BaseTestSuite({
  category: "sort",
  testSuite: "onesweep",
  initializeCPUBuffer: "constant",
  trials: 10,
  params: SortOneSweepRegressionParams,
  primitive: OneSweepSort,
});
