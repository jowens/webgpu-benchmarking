import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  // InitializeMemoryBlock,
  AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpMaxU32 } from "./binop.mjs";
import { datatypeToTypedArray } from "./util.mjs";

// exports: TestSuites, Primitives

class BaseScan extends BasePrimitive {
  constructor(args) {
    super(args);

    // default scan is exclusive
    this.type = args.type ?? "exclusive";
    if (this.type != "inclusive" && this.type != "exclusive") {
      throw new Error(
        `${this.constructor.name}: scan type (currently ${this.type}) must be inclusive or exclusive`
      );
    }

    for (const required of ["datatype", "binop"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
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

  bytesTransferred() {
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
    const referenceOutput = new (datatypeToTypedArray(this.datatype))(
      memdest.length
    );
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
      }
    }
    console.log(
      this.label,
      this.type,
      "should validate to",
      referenceOutput,
      "and actually validates to",
      memdest,
      "\n",
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
    function validates(cpu, gpu, datatype) {
      switch (datatype) {
        case "f32":
          if (cpu == 0) {
            return gpu == 0; // don't divide by zero
          } else {
            return Math.abs(cpu - gpu) / cpu < 0.001;
          }
        default:
          return cpu == gpu;
      }
    }
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors == 0) {
        break;
      }
      if (!validates(referenceOutput[i], memdest[i], this.datatype)) {
        returnString += `Element ${i}: expected ${referenceOutput[i]}, instead saw ${memdest[i]}.\n`;
        allowedErrors--;
      }
    }
    return returnString;
  };
}

const scanWGSizePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

const scanWGCountPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
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
      var<workgroup> temp: array<${this.datatype}, 32>; // zero initialized

      ${this.binop.wgslfn}

      fn roundUpDivU32(a : u32, b : u32) -> u32 {
        return (a + b - 1) / b;
      }

      ${BasePrimitive.fnDeclarations.workgroupScan(this, {
        inputBuffer: "inputBuffer",
        temp: "temp",
      })}

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
  constructor(args) {
    super(args);
  }

  finalizeRuntimeParameters() {
    /* Set reasonable defaults for tunable parameters */
    this.workgroupSize = this.workgroupSize ?? 256;

    /* Compute settings based on tunable parameters */
    this.workgroupCount = Math.ceil(
      this.getBuffer("inputBuffer").size / this.workgroupSize
    );
    this.numPartials = this.workgroupCount;
  }
  reducePerWorkgroupDefinition = () => {
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
    var<workgroup> temp: array<${this.datatype}, 32>; // zero initialized

    ${this.binop.wgslfn}

    fn roundUpDivU32(a : u32, b : u32) -> u32 {
      return (a + b - 1) / b;
    }

    ${BasePrimitive.fnDeclarations.workgroupReduce(this, {
      inputBuffer: "inputBuffer",
      temp: "temp",
    })}

    @compute @workgroup_size(${this.workgroupSize}) fn workgroupReduceKernel(
      Builtins builtins
    ) {
        var reduce: ${this.datatype} = workgroupReduce(id, nwg, lid, sgsz);
        if (lid == 0) {
          outputBuffer[wgid.x] = reduce;
        }
      }
    `;
  };

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
      var<workgroup> temp: array<${this.datatype}, 32>; // zero initialized

      ${this.binop.wgslfn}

      fn roundUpDivU32(a : u32, b : u32) -> u32 {
        return (a + b - 1) / b;
      }

      ${BasePrimitive.fnDeclarations.workgroupScan(this, {
        inputBuffer: "inputBuffer",
        temp: "temp",
      })}

      @compute @workgroup_size(${this.workgroupSize}) fn workgroupScanKernel(
        Builtins builtins
      ) {
          var scan: ${
            this.datatype
          } = workgroup${scanTypeCap}Scan(id, nwg, lid, sgsz);
          outputBuffer[id.x] = scan;
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
        kernel: this.reducePerWorkgroupDefinition,
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["outputBuffer", "inputBuffer"]],
        label: "workgroup scan",
        getDispatchGeometry: () => {
          /* this is a grid-stride loop, so limit the dispatch */
          return [1];
        },
      }),
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

const ScanParams = {
  inputSize: range(8, 26).map((i) => 2 ** i) /* slowest */,
  maxGSLWorkgroupCount: range(2, 8).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

// eslint-disable-next-line no-unused-vars
const ScanParamsSingleton = {
  inputSize: [2 ** 7],
  maxGSLWorkgroupCount: [2 ** 7],
  workgroupSize: [2 ** 7],
};

export const NoAtomicScanTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "no-atomic persistent-kernel u32 sum reduction",
  trials: 10,
  params: ScanParams,
  uniqueRuns: ["inputSize", "workgroupCount", "workgroupSize"],
  primitive: WGScan,
  primitiveConfig: {
    datatype: "u32",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
  plots: [
    { ...scanWGSizePlot, ...{ fy: { field: "workgroupCount" } } },
    { ...scanWGCountPlot, ...{ fy: { field: "workgroupSize" } } },
  ],
});
