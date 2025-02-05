import { range, arrayProd } from "./util.mjs";
import { Kernel, AllocateBuffer } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddF32, BinOpAddU32, BinOpMaxU32 } from "./binop.mjs";
import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";
import { BaseScan, scanWGSizePlot, scanWGCountPlot } from "./scan.mjs";

export class DLDFScan extends BaseScan {
  constructor(args) {
    super(args);

    /* this scan implementation has an additional buffer beyond BaseScan */
    /* Possibly: BaseScan should just list this buffer, even if it's not used */
    this.additionalKnownBuffers = ["scanParameters"];
    for (const knownBuffer of this.additionalKnownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.knownBuffers.push(knownBuffer);
      }
    }
  }

  scandldfWGSL = () => {
    return /* wgsl */ `
enable subgroups;
struct ScanParameters
{
  size: u32,
  vec_size: u32,
  work_tiles: u32,
};

@group(0) @binding(0)
var<storage, read> inputBuffer: array<vec4<${this.datatype}>>;

@group(0) @binding(1)
var<storage, read_write> outputBuffer: array<vec4<${this.datatype}>>;

@group(0) @binding(2)
var<uniform> scanParameters: ScanParameters;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> spine: array<array<atomic<u32>, 2>>;
/** The reason why we don't use a struct is because we want to
 * be able to dynamically index into each member of the split
 * representation, and WGSL doesn't let you do that with a struct.
 * So instead, we make the spine an array of arrays of size 2.
 * */

@group(0) @binding(4)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM: u32 = ${this.workgroupSize};
const SPLIT_MEMBERS = 2u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_PARTIALS_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE * 2u; //Double for conflict avoidance

const VEC4_SPT = 4u; /* each thread handles VEC4_SPT vec4s */
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT; /* how many vec4s in the tile */

const FLAG_NOT_READY = 0u;
const FLAG_READY = 0x40000000u;
const FLAG_INCLUSIVE = 0x80000000u;
const FLAG_MASK = 0xC0000000u;
const VALUE_MASK = 0xffffu;
const ALL_READY = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_control: u32;
var<workgroup> wg_broadcast_tile_id: u32;
var<workgroup> wg_broadcast_prev_red: ${this.datatype};
var<workgroup> wg_partials: array<${this.datatype}, MAX_PARTIALS_SIZE>;
var<workgroup> wg_fallback: array<${this.datatype}, MAX_PARTIALS_SIZE>;

@diagnostic(off, subgroup_uniformity)
fn unsafeShuffle(x: u32, source: u32) -> u32 {
  return subgroupShuffle(x, source);
}

//lop off of the upper ballot bits;
//we never need them across all subgroup sizes
@diagnostic(off, subgroup_uniformity)
fn unsafeBallot(pred: bool) -> u32 {
  return subgroupBallot(pred).x;
}

/* I have "mine", a piece (u32) of a data element.
 * I need to recombine it with the piece(s) from other threads.
 * Currently this is hardcoded for 2 pieces
 * This is the inverse of the below "split" function
 */
fn join(mine: u32, tid: u32) -> ${this.datatype} {
  let xor = tid ^ 1;
  let theirs = unsafeShuffle(mine, xor);
  return bitcast<${this.datatype}>((mine << (16u * tid)) | (theirs << (16u * xor)));
}

/* I need to store x, which is a data element of type $datatype.
 * Return the piece of x (as a uint) that I will actually store
 * Currently this is hardcoded for 2 pieces
 * This is the inverse of the above "join" function */
fn split(x: ${this.datatype}, tid: u32) -> u32 {
  return (bitcast<u32>(x) >> (tid * 16u)) & VALUE_MASK;
}

${this.binop.wgslfn}
${this.fnDeclarations.vec4Functions}
${this.fnDeclarations.subgroupInclusiveOpScan}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
  @builtin(local_invocation_id) threadid: vec3<u32>,
  @builtin(subgroup_invocation_id) laneid: u32,
  @builtin(subgroup_size) lane_count: u32) {

  // sid is subgroup ID, "which subgroup am I within this workgroup"
  let sid = threadid.x / lane_count; // Caution 1D workgroup ONLY! Ok, but technically not in HLSL spec

  // acquire partition index, set the lock
  if (threadid.x == 0u) {
    wg_broadcast_tile_id = atomicAdd(&scan_bump, 1u);
    wg_control = LOCKED;
  }
  let tile_id = workgroupUniformLoad(&wg_broadcast_tile_id);
  // s_offset: within this workgroup, at what index do I start loading?
  let s_offset = laneid + sid * lane_count * VEC4_SPT;

  var t_scan = array<vec4<${this.datatype}>, VEC4_SPT>();
  {
    var i = s_offset + tile_id * VEC_TILE_SIZE;
    if (tile_id < scanParameters.work_tiles - 1u) { // not the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        t_scan[k] = vec4InclusiveScan(inputBuffer[i]);
        i += lane_count;
      }
    }

    if (tile_id == scanParameters.work_tiles - 1u) { // the last tile
      for (var k = 0u; k < VEC4_SPT; k += 1u) {
        if (i < scanParameters.vec_size) {
          t_scan[k] = vec4InclusiveScan(inputBuffer[i]);
        }
        i += lane_count;
      }
    }

    var prev: ${this.datatype} = ${this.binop.identity};
    let lane_mask = lane_count - 1u;
    let circular_shift = (laneid + lane_mask) & lane_mask;
    for(var k = 0u; k < VEC4_SPT; k += 1u) {
      let t = subgroupShuffle(
                subgroupInclusiveOpScan(select(prev, ${this.binop.identity}, laneid != 0u) + t_scan[k].w, laneid, lane_count),
                circular_shift
              );
      t_scan[k] += select(prev, t, laneid != 0u);
      prev = t;
    }

    if (laneid == 0u) {
      wg_partials[sid] = prev;
    }
  }
  workgroupBarrier();

  //Non-divergent subgroup agnostic inclusive scan across subgroup partial reductions
  let lane_log = u32(countTrailingZeros(lane_count));
  let local_spine: u32 = BLOCK_DIM >> lane_log;
  let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
  {
    var offset = 0u;
    var top_offset = 0u;
    let lane_pred = laneid == lane_count - 1u;
    for(var j = lane_count; j <= aligned_size; j <<= lane_log) {
      let step = local_spine >> offset;
      let pred = threadid.x < step;
      let t = subgroupInclusiveAdd(select(${this.binop.identity}, wg_partials[threadid.x + top_offset], pred));
      if (pred) {
        wg_partials[threadid.x + top_offset] = t;
        if (lane_pred) {
          wg_partials[sid + step + top_offset] = t;
        }
      }
      workgroupBarrier();

      if (j != lane_count) {
        let rshift = j >> lane_log;
        let index = threadid.x + rshift;
        if (index < local_spine && (index & (j - 1u)) >= rshift) {
          wg_partials[index] += wg_partials[(index >> offset) + top_offset - 1u];
        }
      }
      top_offset += step;
      offset += lane_log;
    }
  }
  workgroupBarrier();

  //Device broadcast
  if (threadid.x < SPLIT_MEMBERS) {
    let t = split(wg_partials[local_spine - 1u], threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, tile_id == 0u);
    atomicStore(&spine[tile_id][threadid.x], t);
  }

  //lookback, single subgroup
  if (tile_id != 0u) {
    var prev_red: ${this.datatype} = ${this.binop.identity};
    var lookback_id = tile_id - 1u;
    var control_flag = workgroupUniformLoad(&wg_control);
    while (control_flag == LOCKED) {
      if (threadid.x < lane_count) { /* activate only subgroup 0 */
        var spin_count = 0u;
        while (spin_count < MAX_SPIN_COUNT) {
          /* fetch the value from the lookback into SPLIT_MEMBERS threads */
          var flag_payload = select(0u, atomicLoad(&spine[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
          /* is there useful data there across all participating threads? "useful" means either a local reduction
           * (READY) or an inclusive one (INCLUSIVE) */
          if (unsafeBallot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == ALL_READY) {
            /* Yes, useful information. Is it INCLUSIVE? */
            var incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
            if (incl_bal != 0u) {
              // Did we find any inclusive? Alright, the rest are guaranteed to be on their way, lets just wait.
              // This can also block :^)
              while (incl_bal != ALL_READY) {
                flag_payload = select(0u, atomicLoad(&spine[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
                incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
              }
              prev_red += join(flag_payload & VALUE_MASK, threadid.x);
              if (threadid.x < SPLIT_MEMBERS) {
                /* this + needs to be binop */
                let t = split(prev_red + wg_partials[local_spine - 1u], threadid.x) | FLAG_INCLUSIVE;
                atomicStore(&spine[tile_id][threadid.x], t);
              }
              if (threadid.x == 0u) {
                wg_control = UNLOCKED;
                wg_broadcast_prev_red = prev_red;
              }
              break;
            } else {
              /* Useful, but only READY, not INCLUSIVE.
               * Accumulate the value and go back another tile. */
              prev_red += join(flag_payload & VALUE_MASK, threadid.x);
              spin_count = 0u;
              lookback_id -= 1u;
            }
          } else {
            spin_count += 1u;
          }
        }

        if (threadid.x == 0 && spin_count == MAX_SPIN_COUNT) {
          wg_broadcast_tile_id = lookback_id;
        }
      }

      //Fallback if still locked
      control_flag = workgroupUniformLoad(&wg_control);
      if (control_flag == LOCKED) {
        let fallback_id = wg_broadcast_tile_id;
        {
          var t_red: ${this.datatype} = 0;
          var i = s_offset + fallback_id * VEC_TILE_SIZE;
          for(var k = 0u; k < VEC4_SPT; k += 1u) {
            /* TODO: this will be replaced by binop */
            /* for now, treat 1 as multiplicative identity for f32/i32 */
            t_red += dot(inputBuffer[i], vec4<${this.datatype}>(1, 1, 1, 1));
            i += lane_count;
          }

          let s_red = subgroupAdd(t_red);
          if (laneid == 0u) {
            wg_fallback[sid] = s_red;
          }
        }
        workgroupBarrier();

        // Non-divergent subgroup agnostic reduction across subgroup partial reductions
        var f_red: ${this.datatype} = ${this.binop.identity};
        {
          var offset = 0u;
          var top_offset = 0u;
          let lane_pred = laneid == lane_count - 1u;
          for(var j = lane_count; j <= aligned_size; j <<= lane_log) {
            let step = local_spine >> offset;
            let pred = threadid.x < step;
            f_red = subgroupAdd(select(0, wg_fallback[threadid.x + top_offset], pred));
            if (pred && lane_pred) {
              wg_fallback[sid + step + top_offset] = f_red;
            }
            workgroupBarrier();
            top_offset += step;
            offset += lane_log;
          }
        }

        if (threadid.x < lane_count) {
          let f_split = split(f_red, threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, fallback_id == 0u);
          var f_payload = 0u;
          if (threadid.x < SPLIT_MEMBERS) {
            f_payload = atomicMax(&spine[fallback_id][threadid.x], f_split);
          }
          let incl_found = unsafeBallot((f_payload & FLAG_MASK) == FLAG_INCLUSIVE) == ALL_READY;
          if (incl_found) {
            prev_red += join(f_payload & VALUE_MASK, threadid.x);
          } else {
            prev_red += f_red;
          }

          if (fallback_id == 0u || incl_found) {
            if (threadid.x < SPLIT_MEMBERS) {
              let t = split(prev_red + wg_partials[local_spine - 1u], threadid.x) | FLAG_INCLUSIVE;
              atomicStore(&spine[tile_id][threadid.x], t);
            }
            if (threadid.x == 0u) {
              wg_control = UNLOCKED;
              wg_broadcast_prev_red = prev_red;
            }
          } else {
            lookback_id -= 1u;
          }
        }
        control_flag = workgroupUniformLoad(&wg_control);
      }
    }
  }

  {
    /* all writeback to output array is below */
    var i = s_offset + tile_id * VEC_TILE_SIZE;
    let prev = wg_broadcast_prev_red + select(0, wg_partials[sid - 1u], sid != 0u); //wg_broadcast_tile_id is 0 for tile_id 0
    if (tile_id < scanParameters.work_tiles - 1u) { // not the last tile
      for(var k = 0u; k < VEC4_SPT; k += 1u) {
        outputBuffer[i] = t_scan[k] + prev;
        i += lane_count;
      }
    }

    if (tile_id == scanParameters.work_tiles - 1u) { // this is the last tile
      for(var k = 0u; k < VEC4_SPT; k += 1u) {
        if (i < scanParameters.vec_size) {
          outputBuffer[i] = t_scan[k] + prev;
        }
        i += lane_count;
      }
    }
  }
}`;
  };

  finalizeRuntimeParameters() {
    this.MISC_SIZE = 5; // Max scratch memory we use to track various stats
    this.PART_SIZE = 4096; // MUST match the partition size specified in shaders.
    this.MAX_READBACK_SIZE = 8192; // Max size of our readback buffer
    this.workgroupSize = 256;
    const inputSize = this.getBuffer("inputBuffer").size; // bytes
    const inputCount = inputSize / 4; /* 4 is size of datatype */
    this.workgroupCount = Math.ceil(inputCount / this.PART_SIZE);
    this.vec_size = Math.ceil(inputCount / 4); /* 4 is sizeof vec4 */
    this.work_tiles = this.workgroupCount;
    this.scanBumpSize = datatypeToBytes(this.datatype);
    // 4 words per element
    this.spineSize = 4 * this.workgroupCount * datatypeToBytes(this.datatype);
    this.miscSize = 5 * datatypeToBytes(this.datatype);

    // scanParameters is: size: u32, vec_size: u32, work_tiles: u32
    this.scanParameters = new Uint32Array([
      inputCount, // this isn't used currently
      Math.ceil(inputCount / 4),
      Math.ceil(inputCount / this.PART_SIZE),
    ]);
  }
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "scanParameters",
        size: this.scanParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
        label: "Thomas Smith's scan with decoupled lookback/decoupled fallback",
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

const DLDFScanParamsSingleton = {
  inputLength: range(8, 8).map((i) => 2 ** i),
};

export const DLDFScanPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "timing" },
  test_br: "gpuinfo.description",
  caption: "CPU timing (performance.now), GPU timing (timestamps)",
};

export const DLDFScanTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "DLDF scan",
  trials: 1,
  params: DLDFScanParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: DLDFScan,
  primitiveConfig: {
    datatype: "u32",
    type: "inclusive",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
  plots: [DLDFScanPlot],
});
