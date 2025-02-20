import { range } from "./util.mjs";
import { BasePrimitive, Kernel } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import {
  BinOpNopU32,
  BinOpAddF32,
  BinOpAddU32,
  BinOpMaxU32,
  BinOpMaxF32,
  BinOpMinU32,
  BinOpMinF32,
} from "./binop.mjs";
import { datatypeToTypedArray } from "./util.mjs";

export class SubgroupRegression extends BasePrimitive {
  constructor(args) {
    super(args);
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
    this.knownBuffers = ["inputBuffer", "outputBuffer", "debugBuffer"];
    this.args = args;
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
    const sgsz = this.getBuffer("debugBuffer").cpuBuffer[0];
    const referenceOutput = new (datatypeToTypedArray(this.datatype))(
      memdest.length
    );
    /* compute reference output - this populates referenceOutput */
    this.args.computeReference({ referenceOutput, memsrc, sgsz });

    function validates(cpu, gpu, datatype) {
      switch (datatype) {
        case "f32":
          if (cpu == 0) {
            return gpu == 0; // don't divide by zero
          } else {
            return Math.abs((cpu - gpu) / cpu) < 0.001;
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
        returnString += `Element ${i}: expected ${referenceOutput[i]}, instead saw ${memdest[i]}.`;
        if (args.debugBuffer) {
          returnString += ` debug[${i}] = ${args.debugBuffer[i]}.`;
        }
        returnString += "\n";
        allowedErrors--;
      }
    }
    if (returnString !== "") {
      /* we saw an error */
      console.log(
        this.label,
        "should validate to",
        referenceOutput,
        "and actually validates to",
        memdest,
        "\n"
      );
    }
    return returnString;
  };
  kernel = () => {
    return /* wgsl */ `
${this.fnDeclarations.enableSubgroupsIfAppropriate}

@group(0) @binding(0)
var<storage, read_write> outputBuffer: array<${this.datatype}>;

@group(0) @binding(1)
var<storage, read> inputBuffer: array<${this.datatype}>;

@group(0) @binding(2)
var<storage, read_write> debugBuffer: array<u32>;

${this.fnDeclarations.subgroupEmulation}
${this.fnDeclarations.commonDefinitions}
${this.fnDeclarations.subgroupShuffle}
/* some functions only work if binop is defined in the primitive */
${this.binop ? this.binop.wgslfn : ""}
${this.binop ? this.fnDeclarations.subgroupReduce : ""}

@compute @workgroup_size(${this.workgroupSize}, 1, 1)
fn main(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.initializeSubgroupVars}
  ${this.fnDeclarations.computeLinearizedGridParametersSplit}
  ${this.args.wgslOp}
  // example of wgslOp:
  // outputBuffer[gid] = subgroupShuffle(inputBuffer[gid], (gid ^ 1) & (sgsz - 1));
  if (gid == 0) {
    /* validation requires knowing the subgroup size */
    debugBuffer[0] = sgsz;
  }
  return;
}`;
  };
  finalizeRuntimeParameters() {
    const inputSize = this.getBuffer("inputBuffer").size; // bytes
    const inputLength = inputSize / 4; /* 4 is size of datatype */
    this.workgroupCount = Math.ceil(inputLength / this.workgroupSize);
  }
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new Kernel({
        kernel: this.kernel,
        bufferTypes: [["storage", "read-only-storage", "storage"]],
        bindings: [["outputBuffer", "inputBuffer", "debugBuffer"]],
        logKernelCodeToConsole: false,
        logCompilationInfo: true,
        logLaunchParameters: false,
      }),
    ];
  }
}

const SubgroupParams = {
  inputLength: range(8, 10).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
  datatype: ["f32", "u32"],
  disableSubgroups: [true, false],
};

const SubgroupBinOpParams = {
  inputLength: range(8, 10).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
  binop: [
    BinOpAddF32,
    BinOpAddU32,
    BinOpMaxU32,
    BinOpMaxF32,
    BinOpMinU32,
    BinOpMinF32,
  ],
  disableSubgroups: [true, false],
};

const seeds = [
  {
    /* swap with your neighbor, even <-> odd */
    testSuite: "subgroupShuffle neighbor",
    primitive: SubgroupRegression,
    primitiveConfig: {
      wgslOp:
        "outputBuffer[gid] = subgroupShuffle(inputBuffer[gid], (gid ^ 1) & (sgsz - 1));",
      computeReference: ({ referenceOutput, memsrc, sgsz }) => {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i++) {
          referenceOutput[i] = memsrc[i ^ 1];
        }
      },
    },
  },
  {
    /* rotate +1, within a subgroup */
    testSuite: "subgroupShuffle rotate +1",
    primitive: SubgroupRegression,
    primitiveConfig: {
      wgslOp:
        "outputBuffer[gid] = subgroupShuffle(inputBuffer[gid], (gid + 1) & (sgsz - 1));",
      computeReference: ({ referenceOutput, memsrc, sgsz }) => {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i++) {
          const subgroupBaseIdx = i & ~(sgsz - 1); /* top bits */
          const subgroupIdx = (i + 1) & (sgsz - 1); /* bottom bits */
          referenceOutput[i] = memsrc[subgroupBaseIdx + subgroupIdx];
        }
      },
    },
  },
  {
    /* sum-reduce */
    testSuite: "subgroupReduce",
    primitive: SubgroupRegression,
    params: SubgroupBinOpParams,
    primitiveConfig: {
      wgslOp: "outputBuffer[gid] = subgroupReduce(inputBuffer[gid]);",
      computeReference: function ({ referenceOutput, memsrc, sgsz }) {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i += sgsz) {
          let red = this.binop.identity;
          for (let j = 0; j < sgsz; j++) {
            red = this.binop.op(red, memsrc[i + j]);
          }
          for (let j = 0; j < sgsz; j++) {
            referenceOutput[i + j] = red;
          }
        }
      },
    },
  },
];

function tsGen(params) {
  return new BaseTestSuite({
    category: params.category ?? "subgroups",
    ...("testSuite" in params && { testSuite: params.testSuite }),
    trials: params.trials ?? 0,
    params: params.params ?? SubgroupParams,
    primitive: params.primitive ?? SubgroupRegression,
    primitiveConfig: params.primitiveConfig,
  });
}

export const subgroupTestSuites = seeds.map(tsGen);
