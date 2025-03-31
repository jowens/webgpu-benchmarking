import { range, arrayProd } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  // InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32 } from "./binop.mjs";
import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";

// exports: TestSuites, Primitives

export class BaseScan extends BasePrimitive {
  constructor(args) {
    super(args);

    // default scan is exclusive
    this.type = args.type ?? "exclusive";
    if (
      this.type != "inclusive" &&
      this.type != "exclusive" &&
      this.type != "reduce"
    ) {
      throw new Error(
        `${this.constructor.name}: scan type (currently ${this.type}) must be {inclusive, exclusive, reduce}.`
      );
    }

    for (const required of ["datatype", "binop"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
    }

    if (this.datatype != this.binop.datatype) {
      throw new Error(
        `${this.constructor.name}: datatype (${this.datatype}) is incompatible with binop datatype (${this.binop.datatype}).`
      );
    }

    this.knownBuffers = ["inputBuffer", "outputBuffer"];

    for (const knownBuffer of this.knownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
        delete this[knownBuffer]; // let's make sure it's in one place only
      }
    }

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }

  get bytesTransferred() {
    return (
      this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
    );
  }

  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
    const memdest =
      args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    let referenceOutput;
    try {
      referenceOutput = new (datatypeToTypedArray(this.datatype))(
        memdest.length
      );
    } catch (error) {
      console.error(error, "Tried to allocate array of length", memdest.length);
    }
    for (let i = 0; i < memsrc.length; i++) {
      switch (this.type) {
        case "exclusive":
          referenceOutput[i] =
            i == 0
              ? this.binop.identity
              : this.binop.op(referenceOutput[i - 1], memsrc[i - 1]);
          break;
        case "inclusive":
          referenceOutput[i] = this.binop.op(
            i == 0 ? this.binop.identity : referenceOutput[i - 1],
            memsrc[i]
          );
          break;
        case "reduce":
          referenceOutput[0] = this.binop.op(
            i == 0 ? this.binop.identity : referenceOutput[0],
            memsrc[i]
          );
          break;
      }
    }
    function validates(args) {
      return args.cpu == args.gpu;
    }
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors == 0) {
        break;
      }
      if (
        !validates({
          cpu: referenceOutput[i],
          gpu: memdest[i],
          datatype: this.datatype,
        })
      ) {
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
        "with input",
        memsrc,
        "should validate to",
        referenceOutput,
        "and actually validates to",
        memdest,
        this.getBuffer("debugBuffer") ? "\ndebugBuffer" : "",
        this.getBuffer("debugBuffer")
          ? this.getBuffer("debugBuffer").cpuBuffer
          : "",
        this.getBuffer("debug2Buffer") ? "\ndebug2Buffer" : "",
        this.getBuffer("debug2Buffer")
          ? this.getBuffer("debug2Buffer").cpuBuffer
          : "",
        this.binop.constructor.name,
        this.binop.datatype,
        "identity is",
        this.binop.identity,
        "length is",
        memsrc.length,
        "memsrc[",
        memsrc.length - 1,
        "] is",
        memsrc[memsrc.length - 1]
      );
    }
    return returnString;
  };
}

export const scanWGSizePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fx: { field: "timing" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export const scanWGCountPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fx: { field: "timing" },
  stroke: { field: "workgroupCount" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup count",
};

/* needs to be a function if we do string interpolation */
// eslint-disable-next-line no-unused-vars
function scanWGCountFnPlot() {
  return {
    x: { field: "inputBytes", label: "Input array size (B)" },
    /* y.field is just showing off that we can have an embedded function */
    /* { field: "bandwidth", ...} would do the same */
    y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "workgroupCount" },
    text_br: (d) => `${d.gpuinfo.description}`,
    caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
  };
}

// eslint-disable-next-line no-unused-vars
const scanWGSizeBinOpPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fy: { field: "binop" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export class WGScan extends BaseScan {
  constructor(args) {
    super(args);
  }

  finalizeRuntimeParameters() {
    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;
    this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.min(
      Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize),
      this.maxGSLWorkgroupCount
    );
    this.numPartials = this.workgroupCount;
  }
  scanKernelDefinition = () => {
    /** this needs to be an arrow function so "this" is the Primitive
     *  that declares it
     */

    /** this definition could be inline when the kernel is specified,
     * but since we call it twice, we move it here
     */
    const scanType = this.type;
    /* exclusive -> Exclusive, inclusive -> Inclusive */
    const scanTypeCap = scanType.charAt(0).toUpperCase() + scanType.slice(1);
    return /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> outputBuffer: array<${
        this.datatype
      }>;
      /* input */
      @group(0) @binding(1) var<storage, read> inputBuffer: array<${
        this.datatype
      }>;

      ${BasePrimitive.fnDeclarations.commonDefinitions}

      /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
      var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

      ${this.binop.wgslfn}

      ${BasePrimitive.fnDeclarations.roundUpDivU32}

      ${BasePrimitive.fnDeclarations.workgroupScan(this)}

      @compute @workgroup_size(${this.workgroupSize}) fn workgroupScanKernel(
         builtins: Builtins) {
          var scan: ${this.datatype} = workgroup${scanTypeCap}Scan(builtins);
          outputBuffer[builtins.gid.x] = scan;
        }`;
  };
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new Kernel({
        kernel: this.scanKernelDefinition,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["outputBuffer", "inputBuffer"]],
        label: "workgroup scan",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [1];
        },
      }),
    ];
  }
}

export class HierarchicalScan extends BaseScan {
  /**
   * 3 steps, each a kernel:
   * 1. Reduce each workgroup, write into partials (spine)
   * 2. Scan partials (spine)
   * 3. Scan each workgroup, adding in scanned partials
   *
   * (Currently) limited in size to what we can scan in step 2 w/ a single kernel
   * Could be alleviated by continuing to recurse
   */
  constructor(args) {
    super(args);
  }

  finalizeRuntimeParameters() {
    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;
    this.numThreadsPerWorkgroup = arrayProd(this.workgroupSize);

    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.ceil(
      this.getBuffer("inputBuffer").size /
        (this.numThreadsPerWorkgroup * datatypeToBytes(this.datatype))
    );
    this.numPartials = this.workgroupCount;
  }
  reducePerWorkgroupKernel = () => {
    return /* wgsl */ `
    enable subgroups;
    /* output */
    @group(0) @binding(0) var<storage, read_write> partials: array<${this.datatype}>;
    /* input */
    @group(0) @binding(1) var<storage, read> inputBuffer: array<${this.datatype}>;

    /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
    var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

    ${this.fnDeclarations.commonDefinitions}
    ${this.binop.wgslfn}
    ${this.fnDeclarations.roundUpDivU32}
    ${this.fnDeclarations.workgroupReduce}

    @compute @workgroup_size(${this.workgroupSize}) fn reducePerWorkgroupKernel(
      builtins: Builtins
    ) {
        ${this.fnDeclarations.computeLinearizedGridParameters}
        var reduction: ${this.datatype} = workgroupReduce(&inputBuffer, &wgTemp, builtins);
        if (builtins.lid == 0) {
          partials[wgid] = reduction;
        }
      }
    `;
  };

  scanOneWorkgroupKernel = () => {
    return /* wgsl */ `
    enable subgroups;
    /* input + output */
    @group(0) @binding(0) var<storage, read_write> partials: array<${this.datatype}>;

    ${this.fnDeclarations.commonDefinitions}

    /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
    var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

    ${this.binop.wgslfn}
    ${this.fnDeclarations.roundUpDivU32}
    ${this.fnDeclarations.oneWorkgroupExclusiveScan}

    @compute @workgroup_size(${this.workgroupSize}) fn scanOneWorkgroupKernel(
      builtinsUniform: BuiltinsUniform,
      builtinsNonuniform: BuiltinsNonuniform
    ) {
        oneWorkgroupExclusiveScan(builtinsUniform, builtinsNonuniform, &partials);
      }
    `;
  };

  scanAndAddPartialsKernel = () => {
    /** this needs to be an arrow function so "this" is the Primitive
     *  that declares it
     */
    const scanType = this.type;
    /* exclusive -> Exclusive, inclusive -> Inclusive */
    const scanTypeCap = scanType.charAt(0).toUpperCase() + scanType.slice(1);
    return /* wgsl */ `
      enable subgroups;
      /* output */
      @group(0) @binding(0) var<storage, read_write> outputBuffer: array<${this.datatype}>;
      /* input */
      @group(0) @binding(1) var<storage, read> inputBuffer: array<${this.datatype}>;
      @group(0) @binding(2) var<storage, read> partials: array<${this.datatype}>;

      /* TODO: the "32" in the next line should be workgroupSize / subgroupSize */
      var<workgroup> wgTemp: array<${this.datatype}, 32>; // zero initialized

      ${this.fnDeclarations.commonDefinitions}
      ${this.binop.wgslfn}
      ${this.fnDeclarations.roundUpDivU32}
      ${this.fnDeclarations.workgroupScan}

      @compute @workgroup_size(${this.workgroupSize}) fn scanAndAddPartialsKernel(
        builtins: Builtins
      ) {
          ${this.fnDeclarations.computeLinearizedGridParameters}
          var scan: ${this.datatype} = workgroup${scanTypeCap}Scan(builtins, &outputBuffer, &inputBuffer, &partials, &wgTemp);
          outputBuffer[gid] = scan;
        }`;
  };
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new AllocateBuffer({
        label: "partials",
        size: this.numPartials * 4,
      }),
      new Kernel({
        kernel: this.reducePerWorkgroupKernel,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["partials", "inputBuffer"]],
        label: "reduce each workgroup into partials",
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return this.getSimpleDispatchGeometry();
        },
      }),
      new Kernel({
        kernel: this.scanOneWorkgroupKernel,
        bufferTypes: [["storage"]],
        bindings: [["partials"]],
        label: "single-workgroup exclusive scan",
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return [1];
        },
      }),
      new Kernel({
        kernel: this.scanAndAddPartialsKernel,
        bufferTypes: [["storage", "read-only-storage", "read-only-storage"]],
        bindings: [["outputBuffer", "inputBuffer", "partials"]],
        label: "add partials to scan of each workgroup",
        logKernelCodeToConsole: false,
        getDispatchGeometry: () => {
          return this.getSimpleDispatchGeometry();
        },
      }),
    ];
  }
}

const ScanParams = {
  inputLength: range(8, 24).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

// eslint-disable-next-line no-unused-vars
const ScanParamsSingleton = {
  inputLength: [2 ** 7],
  maxGSLWorkgroupCount: [2 ** 7],
  workgroupSize: [2 ** 7],
};

export const HierarchicalScanTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "hierarchical scan",
  trials: 10,
  params: ScanParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: HierarchicalScan,
  primitiveArgs: {
    datatype: "u32",
    binop: BinOpAddU32,
    gputimestamps: true,
  },
  plots: [
    { ...scanWGSizePlot, ...{ fy: { field: "workgroupSize" } } },
    { ...scanWGCountPlot, ...{ fy: { field: "workgroupCount" } } },
  ],
});
