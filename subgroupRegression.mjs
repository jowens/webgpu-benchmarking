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
  }
  kernel = () => {
    return /* wgsl */ `
${this.fnDeclarations.enableSubgroupsIfAppropriate}
@group(0) @binding(0)
var<storage, read> inputBuffer: array<${this.datatype}>;

@group(0) @binding(1)
var<storage, read> outputBuffer: array<${this.datatype}>;

${this.fnDeclarations.subgroupEmulation}
/* defines "binop", the operation associated with the scan monoid */
${this.binop.wgslfn}
${this.fnDeclarations.commonDefinitions}
${this.fnDeclarations.computeLinearizedGridParameters}
${this.fnDeclarations.subgroupBallot}

@compute @workgroup_size(${this.workgroupSize}, 1, 1)
fn main(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.initializeSubgroupVars}
  ${this.fnDeclarations.computeLinearizedGridParameters}
  outputBuffer[gid] = subgroupShuffle(inputBuffer[gid], gid ^ 1);
}`;
  };
  finalizeRuntimeParameters() {}
  compute() {
    this.finalizeRuntimeParameters();
    return [
      new Kernel({
        kernel: this.kernel,
        bufferTypes: [["read-only-storage", "storage"]],
        bindings: [["inputBuffer", "outputBuffer"]],
      }),
    ];
  }
}

const SubgroupParams = {
  inputLength: range(8, 28).map((i) => 2 ** i),
};

export const SubgroupShuffleTestSuite = new BaseTestSuite({
  category: "subgroups",
  testSuite: "subgroupShuffle",
  trials: 1,
  params: SubgroupParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: SubgroupShuffleRegression,
  primitiveConfig: {
    datatype: "u32",
    type: "inclusive",
    binop: BinOpMaxU32,
    gputimestamps: true,
  },
});
