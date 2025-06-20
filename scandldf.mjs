/** Partially based on:
 * Thomas Smith (2024-25)
 * MIT License
 * https://github.com/b0nes164/GPUPrefixSums
 * https://github.com/b0nes164/Decoupled-Fallback-Paper
 */

import { range } from "./util.mjs";
import { Kernel, AllocateBuffer } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import {
  BinOpAdd,
  BinOpMax,
  BinOpMin,
  BinOpAddF32,
  BinOpMaxU32,
} from "./binop.mjs";
import { datatypeToBytes } from "./util.mjs";
import { BaseScan } from "./scan.mjs";

export class DLDFScan extends BaseScan {
  constructor(args) {
    super(args);

    /* this scan implementation has an additional buffer beyond BaseScan */
    /* Possibly: BaseScan should just list this buffer, even if it's not used */
    this.additionalKnownBuffers = ["scanParameters"]; // add "debugBuffer" if necessary
    for (const knownBuffer of this.additionalKnownBuffers) {
      this.knownBuffers.push(knownBuffer);
    }

    /** set a label that properly enumerates the kernel parameterization
     * this is not perfect though; it's also parameterized by values set
     * in finalizeRuntimeParameters (and those aren't available at this time)
     */
    if (!("label" in args)) {
      this.label += this.makeParamString([
        "type",
        "datatype",
        "binop",
        "useSubgroups",
      ]);
    }
  }

  scandldfWGSL = () => {
    let kernel = /* wgsl */ `
/* enables subgroups ONLY IF it was requested when creating device */
/* WGSL requires this declaration is first in the shader */
${this.fnDeclarations.enableSubgroupsIfAppropriate()}
struct ScanParameters
{
  size: u32,
  vec_size: u32,
  work_tiles: u32,
  simulate_mask: u32,
};

@group(0) @binding(0)
var<storage, read> inputBuffer: array<vec4<${this.datatype}>>;

@group(0) @binding(1)
/* for scan output: array<vec4<${this.datatype}>>;
 * for reduce output: array<${this.datatype}>;
 */
var<storage, read_write> outputBuffer: ${
      this.type === "exclusive" || this.type === "inclusive"
        ? "array<vec4<"
        : "array<"
    }${this.datatype}${
      this.type === "exclusive" || this.type === "inclusive" ? ">>" : ">"
    };

@group(0) @binding(2)
var<uniform> scanParameters: ScanParameters;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> spine: array<array<atomic<u32>, 2>>;
/** The reason why we don't use a struct is because a WGSL vector cannot accept
 * atomic types, nor can you make an atomic vector. You CAN dynamically index
 * into vectors.
 */

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM: u32 = ${this.workgroupSize};
const SPLIT_MEMBERS = 2u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_PARTIALS_SIZE = 2u * BLOCK_DIM / MIN_SUBGROUP_SIZE; // 2 per subgroup (double for conflict avoidance)

const VEC4_SPT = 4u; /* each thread handles VEC4_SPT vec4s */
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT; /* how many vec4s in the tile */

const FLAG_NOT_READY = 0u;
const FLAG_READY = 0x40000000u;
const FLAG_INCLUSIVE = 0x80000000u;
const FLAG_MASK = 0xC0000000u;
const VALUE_MASK = 0xffffu;
const ALL_READY = 3u; // this is (1 << SPLIT_MEMBERS) - 1

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_control: u32;
var<workgroup> wg_broadcast_tile_id: u32;
var<workgroup> wg_broadcast_prev_red: ${this.datatype};
var<workgroup> wg_partials: array<${this.datatype}, MAX_PARTIALS_SIZE>;
var<workgroup> wg_fallback: array<${this.datatype}, MAX_PARTIALS_SIZE>;
/** If we're making subgroup calls and we don't have subgroup hardware,
 * this sets up necessary declarations (workgroup memory, subgroup vars) */
${this.fnDeclarations.subgroupEmulation()}

@diagnostic(off, subgroup_uniformity)
fn unsafeShuffle(x: u32, source: u32) -> u32 {
  return subgroupShuffle(x, source);
}

//lop off of the upper ballot bits;
//we never need them across all subgroup sizes
@diagnostic(off, subgroup_uniformity)
fn unsafeBallot(pred: bool) -> u32 {
  /* sgid isn't used if hardware subgroup support */
  return subgroupBallot(pred).x;
}

/* I have "mine", a piece (u32) of a data element.
 * I need to recombine it with the piece(s) from other threads ("theirs").
 * Currently this is hardcoded for 2 pieces
 * This is the inverse of the below "split" function */
fn join(mine: u32, tid: u32) -> ${this.datatype} {
  let xor = tid ^ 1;
  let theirs: u32 = unsafeShuffle(mine, xor);
  return bitcast<${
    this.datatype
  }>((mine << (16u * tid)) | (theirs << (16u * xor)));
}

/* I need to store x, which is a data element of type $datatype.
 * Return the piece of x (as a uint) that I will actually store
 * Currently this is hardcoded for 2 pieces
 * This is the inverse of the above "join" function */
fn split(x: ${this.datatype}, tid: u32) -> u32 {
  return (bitcast<u32>(x) >> (tid * 16u)) & VALUE_MASK;
}

/* defines "binop", the operation associated with the scan monoid */
${this.binop.wgslfn}
/* the following declarations use subgroups ONLY IF enabled */
${this.fnDeclarations.commonDefinitions()}
${this.fnDeclarations.vec4InclusiveScan()}
${this.fnDeclarations.vec4InclusiveToExclusive()}
${this.fnDeclarations.vec4Reduce()}
${this.fnDeclarations.vec4ScalarBinopV4()}
${this.fnDeclarations.subgroupZero()}
${this.fnDeclarations.subgroupInclusiveOpScan()}
${this.fnDeclarations.subgroupReduce()}
${this.fnDeclarations.subgroupShuffle()}
${this.fnDeclarations.subgroupBallot()}
${this.fnDeclarations.wgReduce({ wgTempIsArgument: true })}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.initializeSubgroupVars()}

  // sid is subgroup ID, "which subgroup am I within this workgroup"
  let sid = builtinsNonuniform.lidx / sgsz; // Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

  // acquire partition index, initialize previous reduction var, set the lock
  if (builtinsNonuniform.lidx == 0u) {
    wg_broadcast_tile_id = atomicAdd(&scan_bump, 1u);
    /* this next initialization is important for block 0 because that block never
     * enters lookback and thus this broadcast value is never otherwise set */
    wg_broadcast_prev_red = ${this.binop.identity};
    wg_control = LOCKED;
  }
  let tile_id = workgroupUniformLoad(&wg_broadcast_tile_id);
  // s_offset: within this workgroup, at what index do I start loading?
  let s_offset = sgid + sid * sgsz * VEC4_SPT;
`;
    if (this.type === "exclusive" || this.type === "inclusive") {
      kernel += /* wgsl */ `
  var t_scan = array<vec4<${this.datatype}>, VEC4_SPT>();
  {
    /* This code block reduces VEC4_SPT vec4s per thread across a subgroup. This is
     *     subgroup_size * vec4 * VEC4_SPT items. Each t_scan[k] stripe is
     *     subgroup_size * vec4 items.
     * (1) Per thread: Fill t_scan with inclusive 4-wide scans of input vec4s
     *     Note thread i reads items i, i+sgsz, i+2*sgsz, etc.
     */
    var i = s_offset + tile_id * VEC_TILE_SIZE;
    if (tile_id < scanParameters.work_tiles - 1u) { // not the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        t_scan[k] = vec4InclusiveScan(inputBuffer[i]);
        i += sgsz;
      }
    }

    if (tile_id == scanParameters.work_tiles - 1u) { // the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        if (i < scanParameters.vec_size) {
          t_scan[k] = vec4InclusiveScan(inputBuffer[i]);
        }
        i += sgsz;
      }
    }
    /* t_scan[k].w contains reduction of its vec4 */

    /* (2) Per subgroup: Scan across entire subgroup */

    var prev: ${this.datatype} = ${this.binop.identity};
    let lane_mask = sgsz - 1u;
    let circular_shift = (sgid + lane_mask) & lane_mask;
    /* circular_shift: source is preceding thread in my subgroup, wrapping for thread 0 */
    for (var k = 0u; k < VEC4_SPT; k += 1u) {
      /* (a) scan across reduction of each vec4, feeding in input element "prev" */
      let sgScan =
        subgroupInclusiveOpScan(binop(select(prev,
                                             ${this.binop.identity},
                                             sgid != 0u),
                                      t_scan[k].w /* reduction of my vec4 */ ),
                                sgid, sgsz);


      /* (b) shuffle the scan result from thread x to thread x+1, wrapping
       * after the shuffle is completed, this does two things:
       *    (i) for sgid > 0, it communicates the reduction of all prior elements
       *        (from previous lanes) in this subgroup
       *   (ii) for sgid == 0, it contains the reduction of *all* lanes, which
       *        then gets passed into the next scan
       */
      let t = bitcast<${
        this.datatype
      }>(subgroupShuffle(bitcast<u32>(sgScan), circular_shift));

      /* (c) apply that scan to our current thread's vec4. Recall that our current
       *     thread's vec4 in t_scan contains the inclusive scan of the vec4.
       *     If we want an inclusive scan overall, great, we don't have to do anything.
       *     If we instead want an exclusive scan per vec4, we just recompute it
       *     from the inclusive scan per vec4 here.
       *     After this operation, t_scan[k] now contains an {exclusive, inclusive}
       *     scan of all elements in this subgroup.
       *     If we're only computing reduction, we don't have to update t_scan at all.
       */
      ${
        this.type === "exclusive"
          ? "t_scan[k] = vec4InclusiveToExclusive(t_scan[k]);"
          : ""
      }
      ${
        this.type === "exclusive" || this.type == "inclusive"
          ? "t_scan[k] = vec4ScalarBinopV4(select(prev, t, sgid != 0u), t_scan[k]);"
          : ""
      }
      /* (d) save the reduction of the entire subgroup into t for next k */
      prev = t; /* note: only valid/interesting for sgid == 0 */
    }

    if (sgid == 0u) {
      wg_partials[sid] = prev;
    }
    /* Outputs of this code block:
     * - wg_partials[sid] (subgroup reduction of vec4 per thread)
     * - t_scan[0:VEC4_SPT] (scan of vec4 across subgroup)
     *   (t_scan is not used if we are only reducing)
     */
  }`;
    }

    if (this.type === "reduce") {
      /* reduce is much much simpler. Reduce across each thread's vector (vec4Reduce),
       * then across the subgroup (subgroupReduce), then serially across the VEC4_SPT
       * vectors within the subgroup; put result in wg_partials[subgroupID].  */
      kernel += /* wgsl */ `
  {
    var subgroupReduction: ${this.datatype} = ${this.binop.identity};
    var i = s_offset + tile_id * VEC_TILE_SIZE;
    if (tile_id < scanParameters.work_tiles - 1u) { // not the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        subgroupReduction = binop(subgroupReduction,
                                  subgroupReduce(vec4Reduce(inputBuffer[i])));
        i += sgsz;
      }
    }

    if (tile_id == scanParameters.work_tiles - 1u) { // the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        subgroupReduction = binop(subgroupReduction,
                                  subgroupReduce(select(${this.binop.identity},
                                                               vec4Reduce(inputBuffer[i]),
                                                               i < scanParameters.vec_size)));
        i += sgsz;
      }
    }
    if (sgid == 0u) {
      wg_partials[sid] = subgroupReduction;
    }
  }`;
    }
    kernel += /* wgsl */ `
    workgroupBarrier();

  /* Non-divergent subgroup agnostic inclusive scan across subgroup partial reductions */
  let lane_log = u32(countTrailingZeros(sgsz)); /* log_2(sgsz) */
  let local_spine: u32 = BLOCK_DIM >> lane_log; /* BLOCK_DIM / subgroup size; how
                                                 * many partial reductions in this tile? */
  let aligned_size_base = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    /* fix for aligned_size_base == 1 (needed when subgroup_size == BLOCK_DIM) */
  let aligned_size = select(aligned_size_base, BLOCK_DIM, aligned_size_base == 1);
  {
    var offset = 0u;
    var top_offset = 0u;
    let lane_pred = sgid == sgsz - 1u;
    for (var j = sgsz; j <= aligned_size; j <<= lane_log) {
      let step = local_spine >> offset;
      let pred = builtinsNonuniform.lidx < step;
      let t = subgroupInclusiveOpScan(select(${this.binop.identity},
                                             wg_partials[builtinsNonuniform.lidx + top_offset],
                                             pred),
                                      sgid, sgsz);
      if (pred) {
        wg_partials[builtinsNonuniform.lidx + top_offset] = t;
        if (lane_pred) {
          wg_partials[sid + step + top_offset] = t;
        }
      }
      workgroupBarrier();

      if (j != sgsz) {
        let rshift = j >> lane_log;
        let index = builtinsNonuniform.lidx + rshift;
        if (index < local_spine && (index & (j - 1u)) >= rshift) {
          wg_partials[index] = binop(wg_partials[(index >> offset) + top_offset - 1u], wg_partials[index]);
        }
      }
      top_offset += step;
      offset += lane_log;
    }
  }
  /** output of this code block: populated wg_partials
   * local_spine is the number of subgroups in my workgroup
   * wg_partials[sid] contains reduction of subgroups [0, sid]
   * - if we're reducing, we don't care about this
   * wg_partials[local_spine - 1u] contains reduction of my entire tile
   * - scan and reduction both care about this
   */
  workgroupBarrier();

  /* Post my local reduction to the spine; now visible to the whole device */
  if (builtinsNonuniform.lidx < SPLIT_MEMBERS /* && (tile_id & params.simulate_mask) != 0u */) {
    let t = split(wg_partials[local_spine - 1u], builtinsNonuniform.lidx) | select(FLAG_READY, FLAG_INCLUSIVE, tile_id == 0u);
    atomicStore(&spine[tile_id][builtinsNonuniform.lidx], t);
  }

    /* reduce looks clean up to here */ ////
  /* Begin lookback. Only a single subgroup per workgroup does lookback. */
  if (tile_id != 0u) {
    var prev_red: ${this.datatype} = ${this.binop.identity};
    var lookback_id = tile_id - 1u;
    var control_flag = workgroupUniformLoad(&wg_control);
    while (control_flag == LOCKED) {
      var sg0: bool = isSubgroupZero(builtinsNonuniform.lidx, sgsz);
      if (sg0) { /* activate only subgroup 0 */
        var spin_count = 0u;
        while (spin_count < MAX_SPIN_COUNT) {
          /* fetch the value from tile lookback_id into SPLIT_MEMBERS threads */
          var flag_payload: u32 = select(0u,
                                         atomicLoad(&spine[lookback_id][builtinsNonuniform.lidx]),
                                         builtinsNonuniform.lidx < SPLIT_MEMBERS);

          /* is there useful data there across all participating threads?
           * "useful" means either a local reduction (READY) or an inclusive one (INCLUSIVE) */
          if (unsafeBallot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == ALL_READY) {
            /* Yes, useful data! Is it INCLUSIVE? */
            var seenInclusive = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
            if (seenInclusive != 0u) {
              /* is at least one of the lookback words inclusive? If so, the rest
               * are on their way, let's just wait. */
              /* "This can also block :^)"" ---TS */
              /* This rests on the assumption that the execution width of the load == store.
               * If for whatever reason, this is not true, it risks deadlock without FPG. */
              while (seenInclusive != ALL_READY) {
                /* keep fetching until all participating threads are INCLUSIVE */
                flag_payload = select(0u,
                                      atomicLoad(&spine[lookback_id][builtinsNonuniform.lidx]),
                                      builtinsNonuniform.lidx < SPLIT_MEMBERS);
                seenInclusive = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
              }
              /* flag_payload now contains an inclusive value from lookback_id, put it
               * back together & merge it into prev_red */
              prev_red = binop(join(flag_payload & VALUE_MASK, builtinsNonuniform.lidx), prev_red);
              /* merge that value with my local reduction and store it to the spine */
              if (builtinsNonuniform.lidx < SPLIT_MEMBERS) {
                let t = split(binop(prev_red, wg_partials[local_spine - 1u]),
                              builtinsNonuniform.lidx) |
                        FLAG_INCLUSIVE;

                atomicStore(&spine[tile_id][builtinsNonuniform.lidx], t);
              }
              /* lookback complete. reduction of all previous tiles is in prev_red. */
              if (builtinsNonuniform.lidx == 0u) {
                wg_control = UNLOCKED;
                wg_broadcast_prev_red = prev_red;
              }
              break;
            } else {
              /* Useful, but only READY, not INCLUSIVE.
               * Accumulate the value and go back another tile. */
              prev_red = binop(join(flag_payload & VALUE_MASK, builtinsNonuniform.lidx), prev_red);
              spin_count = 0u;
              lookback_id -= 1u;
            }
          } else {
            spin_count += 1u;
          }
        } /* end while spin_count */

        if (builtinsNonuniform.lidx == 0 && spin_count == MAX_SPIN_COUNT) {
          wg_broadcast_tile_id = lookback_id;
        }
      } /* end activate subgroup 0 */

      /* We are in one of two states here:
       * (1) We completed lookback, in which case control_flag is UNLOCKED.
       *     wg_broadcast_prev_red has the reduction of all previous tiles.
       *     We skip the next code block.
       * (2) We exceeded the spin count, in which case we have to fall back.
       *     control_flag will be LOCKED. wg_broadcast_tile_id has the
       *     stalled tile. We enter the next code block.
       */
      control_flag = workgroupUniformLoad(&wg_control); // this is also a workgroup barrier
      if (control_flag == LOCKED) {
        /* begin fallback */
        let fallback_id = wg_broadcast_tile_id;
        var t_red: ${this.datatype} = ${this.binop.identity};
        var i = s_offset + fallback_id * VEC_TILE_SIZE;
        for (var k = 0u; k < VEC4_SPT; k += 1u) {
          t_red = binop(t_red, vec4Reduce(inputBuffer[i])); /* reduce the 4 members of inputBuffer[i] */
          i += sgsz;
        }
        /* reduce t_red across entire subgroup */
        var f_red: ${this.datatype} = wgReduce(t_red, &wg_fallback, builtinsUniform, builtinsNonuniform);

        var sg0: bool = isSubgroupZero(builtinsNonuniform.lidx, sgsz);
        if (sg0) { /* activate only subgroup 0 */
          let f_split = split(f_red, builtinsNonuniform.lidx) | select(FLAG_READY, FLAG_INCLUSIVE, fallback_id == 0u);
          var f_payload: u32 = 0u;
          if (builtinsNonuniform.lidx < SPLIT_MEMBERS) {
            f_payload = atomicMax(&spine[fallback_id][builtinsNonuniform.lidx], f_split);
          }
          let incl_found = unsafeBallot((f_payload & FLAG_MASK) == FLAG_INCLUSIVE) == ALL_READY;
          if (incl_found) {
            prev_red = binop(join(f_payload & VALUE_MASK, builtinsNonuniform.lidx), prev_red);
          } else {
            prev_red = binop(f_red, prev_red);
          }

          if (fallback_id == 0u || incl_found) {
            if (builtinsNonuniform.lidx < SPLIT_MEMBERS) {
              let t = split(binop(prev_red, wg_partials[local_spine - 1u]), builtinsNonuniform.lidx) | FLAG_INCLUSIVE;
              atomicStore(&spine[tile_id][builtinsNonuniform.lidx], t);
            }
            if (builtinsNonuniform.lidx == 0u) {
              wg_control = UNLOCKED;
              wg_broadcast_prev_red = prev_red;
            }
          } else {
            lookback_id -= 1u;
          }
        }
        control_flag = workgroupUniformLoad(&wg_control);
      } /* end fallback */
    } /* end control flag still locked */
  }`;

    if (this.type === "exclusive" || this.type === "inclusive") {
      /* For scan computations:
       * At this point, t_scan[k] holds a per-subgroup scan.
       * This code block adds the [reduction of all prior blocks +
       * all prior subgroups within this block == prev] within
       * the block, serially over t_scan, and then writes them back to
       * the output
       */
      kernel += /* wgsl */ `
        var i = s_offset + tile_id * VEC_TILE_SIZE;
        let prev = binop(wg_broadcast_prev_red,
                         select(${this.binop.identity},
                                wg_partials[sid - 1u],
                                sid != 0u)); //wg_broadcast_tile_id is 0 for tile_id 0
        if (tile_id < scanParameters.work_tiles - 1u) { // not the last tile
          for(var k = 0u; k < VEC4_SPT; k += 1u) {
            outputBuffer[i] = vec4ScalarBinopV4(prev, t_scan[k]);
            i += sgsz;
          }
        }

        if (tile_id == scanParameters.work_tiles - 1u) { // this is the last tile
          for(var k = 0u; k < VEC4_SPT; k += 1u) {
            if (i < scanParameters.vec_size) {
              outputBuffer[i] = vec4ScalarBinopV4(prev, t_scan[k]);
            }
            i += sgsz;
          }
        }`;
    } else if (this.type === "reduce") {
      /** For reduction computations:
       * Much simpler. Only thread 0 of the last tile must add the reduction
       * of all previous tiles to its tile reduction and write that to the output.
       */
      kernel += /* wgsl */ `
        if (tile_id == scanParameters.work_tiles - 1u) { // this is the last tile
          if (builtinsNonuniform.lidx == 0u) {
            outputBuffer[0] = binop(wg_broadcast_prev_red, wg_partials[local_spine - 1u]);
          }
        }`;
    }
    kernel += "\n}";
    return kernel;
  };

  finalizeRuntimeParameters() {
    this.MISC_SIZE = 5; // Max scratch memory we use to track various stats
    this.PART_SIZE = 4096; // MUST match the partition size specified in shaders.
    this.MAX_READBACK_SIZE = 8192; // Max size of our readback buffer
    this.workgroupSize = 256;
    this.SUBGROUP_MIN_SIZE =
      this.device.adapterInfo.subgroupMinSize ?? this.workgroupSize;
    const inputSize = this.getBuffer("inputBuffer").size; // bytes
    const inputLength = inputSize / 4; /* 4 is size of datatype */
    this.workgroupCount = Math.ceil(inputLength / this.PART_SIZE);
    this.vec_size = Math.ceil(inputLength / 4); /* 4 is sizeof vec4 */
    this.work_tiles = this.workgroupCount;
    this.scanBumpSize = datatypeToBytes(this.datatype);
    // one vec4 per workgroup in spine
    this.spineSize = 4 * this.workgroupCount * datatypeToBytes(this.datatype);
    this.miscSize = 5 * datatypeToBytes(this.datatype);

    // scanParameters is: size: u32, vec_size: u32, work_tiles: u32
    this.scanParameters = new Uint32Array([
      inputLength, // this isn't used in the shader currently
      this.vec_size,
      this.workgroupCount,
      this.simulate_mask,
    ]);
  }
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "scanParameters",
        size: this.scanParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        populateWith: this.scanParameters,
      }),
      new AllocateBuffer({
        label: "scanBump",
        size: this.scanBumpSize,
      }),
      new AllocateBuffer({
        label: "spine",
        size: this.spineSize,
      }),
      new AllocateBuffer({
        label: "misc",
        size: this.miscSize,
      }),
      new Kernel({
        kernel: this.scandldfWGSL,
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

const DLDFScanParams = {
  inputLength: range(8, 28).map((i) => 2 ** i),
};

export const DLDFScanPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "timing" },
  test_br: "gpuinfo.description",
  caption: "CPU timing (performance.now), GPU timing (timestamps)",
};

export const DLDFGPUTimePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "gputime", label: "GPU runtime (ns)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "GPU timing (timestamps)",
  filter: (row) => row.timing === "GPU",
};

export const DLDFGPUBWPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "GPU timing (timestamps)",
  filter: (row) => row.timing === "GPU",
};

export const DLDFCPUTimePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "cputime", label: "CPU runtime (ns)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "CPU timing (performance.now)",
  filter: (row) => row.timing === "CPU",
};

export const DLDFDottedCPUTimePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "cputime", label: "CPU runtime (ns)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "CPU timing (performance.now)",
  mark: "dot",
  filter: (row) => row.timing === "CPU",
};

export const DLDFDottedGPUTimePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "gputime", label: "GPU runtime (ns)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "GPU timing (timestamps)",
  mark: "dot",
  filter: (row) => row.timing === "GPU",
};

export const DLDFCPUBWPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "webgpucache" },
  test_br: "gpuinfo.description",
  caption: "CPU timing (performance.now)",
  filter: (row) => row.timing === "CPU",
};

export const DLDFScanTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF scan",
  trials: 1,
  params: DLDFScanParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: DLDFScan,
  primitiveArgs: {
    datatype: "u32",
    type: "inclusive",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
  plots: [DLDFScanPlot],
});

export const DLDFReduceTestSuite = new BaseTestSuite({
  category: "reduce",
  testSuite: "DLDF reduce",
  trials: 1,
  params: DLDFScanParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: DLDFScan,
  primitiveArgs: {
    datatype: "f32",
    type: "reduce",
    binop: BinOpAddF32,
    gputimestamps: true,
  },
  plots: [DLDFScanPlot],
});

const DLDFRegressionParams = {
  inputLength: range(10, 23).map((i) => 2 ** i),
  type: ["reduce", "inclusive", "exclusive"],
  datatype: ["f32", "u32"],
  binopbase: [BinOpAdd, BinOpMax, BinOpMin],
  disableSubgroups: [false /*, true*/],
};

const DLDFLengthOnlyRegressionParams = {
  // webgpucache: ["enable", "disable"] /* put this first so it varies slowest */,
  inputLength: range(10, 25).map((i) => 2 ** i),
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

const DLDFLotsOfLengthsWithCacheRegressionParams = {
  webgpucache: ["enable", "disable"] /* put this first so it varies slowest */,
  inputLength: range(0, 100).map((i) => 2 ** 20 + 16384 * i),
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

const DLDF2LengthsWithCacheRegressionParams = {
  webgpucache: ["enable", "disable"] /* put this first so it varies slowest */,
  inputLength: range(0, 2).map((i) => 2 ** 20 + 16384 * i),
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

const DLDFLengthOnlyRegressionParams22 = {
  webgpucache: ["enable", "disable"] /* put this first so it varies slowest */,
  inputLength: [2 ** 22],
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

const DLDFLengthOnlyRegressionParams25 = {
  webgpucache: ["enable", "disable"] /* put this first so it varies slowest */,
  inputLength: [2 ** 25],
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

const DLDFMiniParams = {
  inputLength: [2 ** 20],
  type: ["inclusive", "exclusive"],
  datatype: ["f32", "u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false],
};

const DLDFSingletonParams = {
  inputLength: [2 ** 20],
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false],
};

export const DLDFScanAccuracyRegressionSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 20,
  params: DLDFRegressionParams,
  primitive: DLDFScan,
  plots: [DLDFScanPlot],
});

export const DLDFCachePerfTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 1,
  params: DLDFLengthOnlyRegressionParams,
  primitive: DLDFScan,
  plots: [
    DLDFCPUTimePlot,
    DLDFCPUBWPlot,
    DLDFGPUTimePlot,
    DLDFGPUBWPlot,
    DLDFScanPlot,
  ],
});

export const DLDFDottedCachePerfTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 1,
  params: DLDFLotsOfLengthsWithCacheRegressionParams,
  primitive: DLDFScan,
  plots: [DLDFDottedCPUTimePlot, DLDFDottedGPUTimePlot],
});

export const DLDFDottedCachePerf2TestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 1,
  params: DLDF2LengthsWithCacheRegressionParams,
  primitive: DLDFScan,
  plots: [DLDFDottedCPUTimePlot, DLDFDottedGPUTimePlot],
});

export const DLDFScanMiniSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 2,
  params: DLDFMiniParams,
  primitive: DLDFScan,
});

const DLDFailureParams = {
  inputLength: [2 ** 25],
  type: ["exclusive"],
  datatype: ["u32"],
  binopbase: [BinOpAdd],
  disableSubgroups: [false /*, true*/],
};

export const DLDFFailureSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 10,
  params: DLDFailureParams,
  primitive: DLDFScan,
});

export const DLDFSingletonWithTimingSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 1,
  params: DLDFSingletonParams,
  primitive: DLDFScan,
});

export const DLDFPerfSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF",
  trials: 1,
  params: DLDFLengthOnlyRegressionParams,
  primitive: DLDFScan,
  plots: [DLDFScanPlot],
});
