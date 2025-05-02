import { range } from "./util.mjs";
import { BasePrimitive, Kernel } from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAdd, BinOpMax, BinOpMin } from "./binop.mjs";
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
    const referenceOutput = new (datatypeToTypedArray(
      this.getBuffer("outputBuffer").datatype
    ))(memdest.length);
    /* compute reference output - this populates referenceOutput */
    this.args.computeReference({ referenceOutput, memsrc, sgsz });

    function validates(args) {
      return args.cpu === args.gpu;
    }
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors === 0) {
        break;
      }
      if (
        !validates({
          cpu: referenceOutput[i],
          gpu: memdest[i],
          datatype: this.datatype,
        })
      ) {
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
        "with input",
        memsrc,
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
${this.fnDeclarations.enableSubgroupsIfAppropriate()}

@group(0) @binding(0)
var<storage, read_write> outputBuffer: array<${
      this.getBuffer("outputBuffer").datatype
    }>;

@group(0) @binding(1)
var<storage, read> inputBuffer: array<${
      this.getBuffer("inputBuffer").datatype
    }>;

@group(0) @binding(2)
var<storage, read_write> debugBuffer: array<u32>;

${this.fnDeclarations.subgroupEmulation()}
${this.fnDeclarations.commonDefinitions()}
${this.fnDeclarations.subgroupShuffle()}
${this.fnDeclarations.subgroupBallot()}
/* some functions only work if binop is defined in the primitive */
${this.binop ? this.binop.wgslfn : ""}
${this.binop ? this.fnDeclarations.subgroupReduce() : ""}
${this.binop ? this.fnDeclarations.subgroupInclusiveOpScan() : ""}

@compute @workgroup_size(${this.workgroupSize}, 1, 1)
fn main(builtinsUniform: BuiltinsUniform,
        builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.initializeSubgroupVars()}
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  ${
    typeof this.args.wgslOp === "function"
      ? this.args.wgslOp(this)
      : this.args.wgslOp
  }
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
  inputLength: range(8, 20).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
  datatype: ["f32", "u32"],
  binopbase: [BinOpAdd, BinOpMax, BinOpMin],
  disableSubgroups: [true, false],
};

const seeds = [
  {
    /* swap with your neighbor, even <-> odd */
    testSuite: "subgroupShuffle neighbor",
    primitive: SubgroupRegression,
    primitiveArgs: {
      wgslOp: (env) => {
        return /* wgsl */ `outputBuffer[gid] = bitcast<${env.datatype}>(subgroupShuffle(bitcast<u32>(inputBuffer[gid]), (gid ^ 1) & (sgsz - 1)));`;
      },
      computeReference: ({ referenceOutput, memsrc /*, sgsz */ }) => {
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
    primitiveArgs: {
      wgslOp: (env) => {
        return /* wgsl */ `outputBuffer[gid] = bitcast<${env.datatype}>(subgroupShuffle(bitcast<u32>(inputBuffer[gid]), (gid + 1) & (sgsz - 1)));`;
      },
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
    /* rotate -1, within a subgroup */
    testSuite: "subgroupShuffle rotate -1",
    primitive: SubgroupRegression,
    primitiveArgs: {
      wgslOp: (env) => {
        return /* wgsl */ `outputBuffer[gid] = bitcast<${env.datatype}>(subgroupShuffle(bitcast<u32>(inputBuffer[gid]), (gid + sgsz - 1) & (sgsz - 1)));`;
      },
      computeReference: ({ referenceOutput, memsrc, sgsz }) => {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i++) {
          const subgroupBaseIdx = i & ~(sgsz - 1); /* top bits */
          const subgroupIdx = (i - 1) & (sgsz - 1); /* bottom bits */
          referenceOutput[i] = memsrc[subgroupBaseIdx + subgroupIdx];
        }
      },
    },
  },
  {
    /* reduce */
    testSuite: "subgroupReduce",
    primitive: SubgroupRegression,
    params: SubgroupBinOpParams,
    primitiveArgs: {
      wgslOp: /* wgsl */ `outputBuffer[gid] = subgroupReduce(inputBuffer[gid]);`,
      computeReference: function ({ referenceOutput, memsrc, sgsz }) {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i += sgsz) {
          let red = this.binop.identity;
          /* first reduce across the subgroup ... */
          for (let j = 0; j < sgsz && i + j < memsrc.length; j++) {
            red = this.binop.op(red, memsrc[i + j]);
          }
          /* ... then write it to the entire subgroup */
          for (let j = 0; j < sgsz && i + j < memsrc.length; j++) {
            referenceOutput[i + j] = red;
          }
        }
      },
    },
  },
  {
    /* ballot */
    testSuite: "subgroupBallot",
    primitive: SubgroupRegression,
    primitiveArgs: {
      wgslOp: /* wgsl */ `outputBuffer[gid] = subgroupBallot((bitcast<u32>(inputBuffer[gid]) & 1) != 0);`,
      computeReference: function ({ referenceOutput, memsrc, sgsz }) {
        /* compute reference output */
        let out = new Uint32Array(4);
        /* f32ToU32 objects allow a bitcast to U32 */
        const f32ToU32Buffer = new ArrayBuffer(4);
        const f32ToU32DataView = new DataView(f32ToU32Buffer);
        const cappedSgsz = Math.min(sgsz, 128);
        for (let i = 0; i < memsrc.length; i += sgsz) {
          out[0] = out[1] = out[2] = out[3] = 0;
          /* first ballot across the subgroup (capping after 128 elements) ... */
          for (let j = 0; j < cappedSgsz && i + j < memsrc.length; j++) {
            let u32Value;
            switch (this.datatype) {
              case "f32":
                f32ToU32DataView.setFloat32(0, memsrc[i + j], true);
                u32Value = f32ToU32DataView.getUint32(0, true);
                break;
              case "u32":
                u32Value = memsrc[i + j];
                break;
            }
            out[Math.floor(j / 32)] |= (u32Value & 1) << j % 32;
          }
          /* ... then write it to the entire subgroup */
          for (let j = 0; j < sgsz && i + j < memsrc.length; j++) {
            for (let k = 0; k < 4; k++) {
              referenceOutput[4 * (i + j) + k] = out[k];
            }
          }
        }
      },
    },
  },
  {
    /* subgroupInclusiveOpScan */
    testSuite: "subgroupInclusiveOpScan",
    primitive: SubgroupRegression,
    params: SubgroupBinOpParams,
    primitiveArgs: {
      wgslOp: /* wgsl */ `outputBuffer[gid] = subgroupInclusiveOpScan(inputBuffer[gid], sgid, sgsz);`,
      computeReference: function ({ referenceOutput, memsrc, sgsz }) {
        /* compute reference output */
        for (let i = 0; i < memsrc.length; i += sgsz) {
          let acc = this.binop.identity;
          for (let j = 0; j < sgsz && i + j < memsrc.length; j++) {
            acc = referenceOutput[i + j] = this.binop.op(acc, memsrc[i + j]);
          }
        }
      },
    },
  },
];

function regressionGen(params) {
  return new BaseTestSuite({
    category: params.category ?? "subgroups",
    ...("testSuite" in params && { testSuite: params.testSuite }),
    trials: params.trials ?? 0,
    params: params.params ?? SubgroupParams,
    primitive: params.primitive ?? SubgroupRegression,
    primitiveArgs: params.primitiveArgs,
  });
}

export const subgroupAccuracyRegressionSuites = seeds.map(regressionGen);
