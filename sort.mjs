import { BasePrimitive } from "./primitive.mjs";

export class BaseSort extends BasePrimitive {
  constructor(args) {
    super(args);

    this.knownBuffers = ["inputKeys", "outputKeys"];

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
    return this.getBuffer("inputKeys").size + this.getBuffer("outputKeys").size;
  }

  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc = args.inputKeys ?? this.getBuffer("inputKeys").cpuBuffer;
    const memdest = args.outputKeys ?? this.getBuffer("outputKeys").cpuBuffer;
    const referenceOutput = memsrc.slice().sort();
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
