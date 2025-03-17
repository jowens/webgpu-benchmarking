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
}
