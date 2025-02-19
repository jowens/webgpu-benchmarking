import { range } from "./util.mjs";
import { BasePrimitive, Kernel } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import {
  BinOpAddF32,
  BinOpAddU32,
  BinOpMaxU32,
  BinOpMinF32,
} from "./binop.mjs";
import { datatypeToBytes } from "./util.mjs";

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
  kernel = () => {
    return /* wgsl */ `
${this.fnDeclarations.enableSubgroupsIfAppropriate}

@group(0) @binding(0)
var<storage, read_write> outputBuffer: array<${this.datatype}>;

@group(0) @binding(1)
var<storage, read> inputBuffer: array<${this.datatype}>;

${this.fnDeclarations.subgroupEmulation}
/* defines "binop", the operation associated with the scan monoid */
${this.binop.wgslfn}
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
        logKernelCodeToConsole: true,
      }),
    ];
  }
}

const SubgroupParams = {
  inputLength: range(8, 10).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
  datatype: ["f32"],
};

export const SubgroupShuffleTestSuite = new BaseTestSuite({
  category: "subgroups",
  testSuite: "subgroupShuffle",
  trials: 1,
  params: SubgroupParams,
  primitive: SubgroupShuffleRegression,
  primitiveConfig: {
    datatype: "u32",
    type: "inclusive",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
});
