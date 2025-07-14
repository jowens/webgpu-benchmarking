import { BasePrimitive } from "./primitive.mjs";
import { range } from "./util.mjs";
import { Kernel, AllocateBuffer, WriteGPUBuffer } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";

export class BaseMatrixTranspose extends BasePrimitive {
  constructor(args) {
    super(args);

    /* buffer registration should be done in subclass */

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }
}

class WorkQueueMatrixTranspose extends BaseMatrixTranspose {
  constructor(args) {
    super(args);

    this.knownBuffers = [
      "inputBuffer",
      "outputBuffer",
      "workQueue",
      "workCount",
    ];

    for (const knownBuffer of this.knownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.registerBuffer({
          label: knownBuffer,
          buffer: args[knownBuffer],
          device: this.device,
        });
        delete this[knownBuffer]; // let's make sure it's in one place only
      }
    }
  }

  get bytesTransferred() {
    return (
      this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
    );
  }

  /** parameters:
   * - matrixWidthHeight: input matrix has edge length of this
   *   Uniform param
   * - baseCaseWidthHeight: size when we stop recursing and just do the transpose
   *   Currently a uniform param; could be a compile-time param though
   * - workgroupCount (should fill the machine)
   */

  /** Work queue encoding
   * Size of matrix is mxm
   * Size of baseCaseWidthHeight is bxb
   * - So actually valid points are (x,y) = ([0 - m/b), [0 - m/b)]
   * We have to encode x, y, size
   * x and y each require log(m/b) bits
   * size is always a power of 2 and thus requires log(log(m/b)) bits
   * Let's say 13 bits for x and y, 6 bits for size
   * [x13, y13, sz6] = 32 bits
   *                          (for b = 1)
   * x = x13 * b              (can represent up to 8192^2 = 67M elements)
   * y = y13 * b
   * sz = 2 ** (sz6 + log(b)) (can represent up to size of 2^63)
   * Need to make sure that there is no valid piece of work that encodes to 0
   */

  workToU32({ x, y, sz }) {
    const x13 = x / this.baseCaseWidthHeight;
    const y13 = y / this.baseCaseWidthHeight;
    const sz6 = 2 ** (Math.log2(sz) - Math.log2(this.baseCaseWidthHeight));
    return (x13 << 19) | (y13 << 6) | sz6;
  }

  workQueueMatrixTransposeWGSL = () => {
    let kernel = /* wgsl */ `
    ${this.fnDeclarations.enableSubgroupsIfAppropriate()}
    ${this.fnDeclarations.commonDefinitions()}
    ${this.fnDeclarations.subgroupZero()}

    struct TransposeParameters {
      matrixWidthHeight: u32, /* width === height */
      baseCaseWidthHeight: u32,
      workUnits: u32,
    };

    struct WorkUnit {
      x: u32,
      y: u32,
      sz: u32
    };

    const BASE_CASE_WIDTH_HEIGHT: u32 = ${this.baseCaseWidthHeight};
    const BASE_CASE_WIDTH_HEIGHT_LOG2: u32 = ${Math.log2(
      this.baseCaseWidthHeight
    )};

    fn u32ToWork(in: u32) -> WorkUnit {
      var work: WorkUnit;
      work.x = (in >> 19) * BASE_CASE_WIDTH_HEIGHT;
      work.y = ((in >> 6) & 0x1fff) * BASE_CASE_WIDTH_HEIGHT;
      work.sz = (in & 0x3f);
      return work;
    }

    fn workToU32(work: WorkUnit) -> u32 {
      let x13: u32 = work.x / BASE_CASE_WIDTH_HEIGHT;
      let y13: u32 = work.y / BASE_CASE_WIDTH_HEIGHT;
      let sz6: u32 = 1u << (u32(log2(f32(work.sz))) - BASE_CASE_WIDTH_HEIGHT_LOG2);
      return (x13 << 19) | (y13 << 6) | sz6;
    }

    @group(0) @binding(0)
    var<storage, read_write> outputBuffer: array<${this.datatype}>;

    @group(0) @binding(1)
    var<storage, read> inputBuffer: array<${this.datatype}>;

    /* u32: points to INPUT (encodes [ x, y, size ]) */
    @group(0) @binding(2)
    var<storage, read_write> workQueue: array<atomic<u32>>;

    @group(0) @binding(3)
    var<storage, read_write> workCount: atomic<u32>;

    @group(1) @binding(0)
    var<uniform> transposeParameters : TransposeParameters;

    var<workgroup> wg_matrix: array<u32, ${
      this.baseCaseWidthHeight * this.baseCaseWidthHeight
    }>;

    fn performTranspose(workUnit: u32) {
      /* uses wg_matrix */
      /** can be more efficient if we pad the matrix
       * in workgroup memory to avoid bank conflicts */

    }

    @compute @workgroup_size(${this.workgroupCount}, 1, 1)
    fn workQueueMatrixTranspose(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
      var matrixWidthHeight = transposeParameters.matrixWidthHeight;
      var baseCaseWidthHeight = transposeParameters.baseCaseWidthHeight;
      var workUnits = transposeParameters.workUnits;

      for (var i = 0; i < 100000; i++) { /* so we don't go into an infinite loop */
      // loop { /* this is the loop we want when we're production-ready */
        let currentWorkCount = atomicLoad(&workCount);
        if (currentWorkCount == workUnits) {
          /* we're done! */
          return;
        }
        /* Insert (add work)
         * - Find an empty slot
         * - Atomic weak compare-exchange myWork with that slot
         * - Success means you get back a zero
         * - Failure: Start over
         */

        /* Consume (remove work)
         * - Find a full slot (via atomic load)
         * - Atomic weak compare-exchange zero with that slot
         * - Success means you get back a non-zero
         * - Failure: EITHER
         *   - Loop over atomic exchanges
         *   - Loop over find-a-full-slot
         */

      }

    } /* end kernel workQueueMatrixTranspose */`;
    return kernel;
  };

  finalizeRuntimeParameters() {
    this.workgroupCount = this.workgroupCount ?? 256;
    this.baseCaseWidthHeight = this.baseCaseWidthHeight ?? 32;
    this.workQueueSize = this.workQueueSize ?? 2 ** 18;

    const inputLength = this.getBuffer("inputBuffer").length;
    this.matrixWidthHeight = Math.sqrt(inputLength);
    this.transposeParameters = new Uint32Array(3);
    this.transposeParameters[0] = this.matrixWidthHeight;
    this.transposeParameters[1] = this.baseCaseWidthHeight;
    this.transposeParameters[2] =
      (this.matrixWidthHeight / this.baseCaseWidthHeight) ** 2;

    this.workQueue = new Uint32Array(
      this.workQueueSize / Uint32Array.BYTES_PER_ELEMENT
    );
    this.workQueue[0] = this.workToU32({
      x: 0,
      y: 0,
      sz: this.matrixWidthHeight,
    }); /* the entire matrix */
    this.workQueue[1] = 0;
    this.workQueue[2] = this.matrixWidthHeight;

    this.workCountLength = 1;
    this.workCountSize = this.workCountLength * Uint32Array.BYTES_PER_ELEMENT;
  }
  compute() {
    this.finalizeRuntimeParameters();
    const bufferTypes = [
      ["storage", "read-only-storage", "storage", "storage"],
      ["uniform"],
    ];
    const bindings = [
      ["outputBuffer", "inputBuffer", "workQueue", "workCount"],
      ["transposeParameters"],
    ];
    return [
      new AllocateBuffer({
        label: "transposeParameters",
        size: this.transposeParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      new WriteGPUBuffer({
        label: "transposeParameters",
        cpuSource: this.transposeParameters,
      }),
      new AllocateBuffer({
        label: "workQueue",
        size: this.workQueueSize,
      }),
      new WriteGPUBuffer({
        label: "workQueue",
        cpuSource: this.workQueue,
      }),
      new AllocateBuffer({
        label: "workCount",
        size: this.workCountSize,
      }),
      new Kernel({
        kernel: this.workQueueMatrixTransposeWGSL,
        bufferTypes,
        bindings,
        label: `work queue matrix transpose (${this.transposeParameters[0]} x ${this.transposeParameters[0]}) [subgroups: ${this.useSubgroups}]`,
        logKernelCodeToConsole: true,
        getDispatchGeometry: () => {
          return [this.workgroupCount];
        },
      }),
    ];
  }
  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc =
      args.inputBuffer?.cpuBuffer ??
      args.inputBuffer ??
      this.getBuffer("inputBuffer").cpuBuffer;
    const memdest =
      args.outputBuffer?.cpuBuffer ??
      args.outputBuffer ??
      this.getBuffer("outputBuffer").cpuBuffer;
    if (memsrc === memdest) {
      console.warn(
        "Warning: source and destination for transpose are the same buffer",
        "src:",
        memsrc,
        "dest",
        memdest
      );
    }
    const referenceOutput = new Uint32Array(memsrc.length);
    for (let i = 0; i < memsrc.length; i++) {
      const x = i % this.matrixWidthHeight;
      const y = i / this.matrixWidthHeight;
      referenceOutput[y + x * this.matrixWidthHeight] = memsrc[i];
    }
    console.info(
      memsrc.length,
      this.matrixWidthHeight,
      memsrc,
      memdest,
      referenceOutput
    );

    /* arrow function to allow use of this.type within it */
    const validates = (args) => {
      return args.cpu === args.gpu;
    };
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors === 0) {
        break;
      }
      const compare = {
        cpu: referenceOutput[i],
        gpu: memdest[i],
        datatype: this.datatype,
        i,
        label: args?.outputKeys?.label,
      };
      if (!validates(compare)) {
        returnString += `\nElement ${i}: expected ${compare.cpu}, instead saw ${compare.gpu} `;
        returnString += `(diff: ${Math.abs(
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
        this.direction,
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

const TransposeParams = {
  inputLength: [2 ** 12],
  datatype: ["u32"],
  type: ["keysonly" /* "keyvalue", */],
  disableSubgroups: [false],
  workgroupCount: [1],
};

export const TransposeRegressionSuite = new BaseTestSuite({
  category: "transpose",
  testSuite: "workqueue",
  trials: 2,
  initializeCPUBuffer: "fisher-yates",
  params: TransposeParams,
  primitive: WorkQueueMatrixTranspose,
});
