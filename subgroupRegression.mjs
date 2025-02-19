import { range } from "./util.mjs";
import { BasePrimitive, Kernel } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import {
  BinOpAddF32,
  BinOpAddU32,
  BinOpMaxU32,
  BinOpMinF32,
} from "./binop.mjs";
import { datatypeToTypedArray } from "./util.mjs";

export class SubgroupShuffleRegression extends BasePrimitive {
  constructor(args) {
    super(args);
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
    this.knownBuffers = ["inputBuffer", "outputBuffer"];
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
    const referenceOutput = new (datatypeToTypedArray(this.datatype))(
      memdest.length
    );
    /* compute reference output */
    for (let i = 0; i < memsrc.length; i++) {
      referenceOutput[i] = memsrc[i ^ 1];
    }
    console.log(
      this.label,
      this.type,
      "should validate to",
      referenceOutput,
      "and actually validates to",
      memdest,
      "\n"
    );
    function validates(cpu, gpu, datatype) {
      return cpu == gpu;
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
    return returnString;
  };
  kernel = () => {
    return /* wgsl */ `
${this.fnDeclarations.enableSubgroupsIfAppropriate}

@group(0) @binding(0)
var<storage, read_write> outputBuffer: array<${this.datatype}>;

@group(0) @binding(1)
var<storage, read> inputBuffer: array<${this.datatype}>;

${this.fnDeclarations.subgroupEmulation}
${this.fnDeclarations.commonDefinitions}
${this.fnDeclarations.subgroupShuffle}

@compute @workgroup_size(${this.workgroupSize}, 1, 1)
fn main(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.initializeSubgroupVars}
  ${this.fnDeclarations.computeLinearizedGridParametersSplit}
  outputBuffer[gid] = subgroupShuffle(inputBuffer[gid], gid ^ 1);
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
        bufferTypes: [["storage", "read-only-storage"]],
        bindings: [["outputBuffer", "inputBuffer"]],
        logKernelCodeToConsole: false,
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

export const SubgroupShuffleTestSuite = new BaseTestSuite({
  category: "subgroups",
  testSuite: "subgroupShuffle",
  trials: 1,
  params: SubgroupParams,
  primitive: SubgroupShuffleRegression,
});
