import { BasePrimitive } from "./primitive.mjs";

export class BaseSort extends BasePrimitive {
  constructor(args) {
    super(args);

    this.knownBuffers = ["keysIn", "keysOut", "payloadIn", "payloadOut"];

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
    return this.getBuffer("keysIn").size + this.getBuffer("keysOut").size;
  }

  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc =
      args.inputKeys?.cpuBuffer ??
      args.inputKeys ??
      this.getBuffer("keysIn").cpuBuffer;
    const memdest =
      args.outputKeys?.cpuBuffer ??
      args.outputKeys ??
      this.getBuffer("keysOut").cpuBuffer;
    let referenceOutput;
    switch (args?.outputKeys?.label) {
      case "hist":
      case "passHist": {
        const hist = new Uint32Array(1024);
        for (let i = 0; i < memsrc.length; i++) {
          for (let j = 0; j < 4; j++) {
            const histOffset = j * 256;
            const bucket = (memsrc[i] >>> (j * 8)) & 0xff;
            hist[histOffset + bucket]++;
          }
        }
        if (args?.outputKeys?.label === "hist") {
          referenceOutput = hist;
          break;
        }
        const passHist = new Uint32Array(memdest.length);
        const FLAG_INCLUSIVE = 2 << 30;
        /* divide into 4 pieces (4 passes), write only first RADIX (256) elements of each */
        for (let j = 0; j < 4; j++) {
          const histOffset = j * 256;
          const passHistOffset = j * (memdest.length / 4);
          let acc = 0;
          for (let bucket = 0; bucket < 256; bucket++) {
            passHist[passHistOffset + bucket] = acc | FLAG_INCLUSIVE;
            acc += hist[histOffset + bucket];
          }
        }
        if (args?.outputKeys?.label === "passHist") {
          referenceOutput = passHist;
          break;
        }
        break;
      }
      case undefined:
      default:
        referenceOutput = memsrc.slice().sort();
        break;
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
          : "",
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
