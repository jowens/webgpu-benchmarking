import { BasePrimitive } from "./primitive.mjs";

export class BaseSort extends BasePrimitive {
  constructor(args) {
    super(args);

    /* buffer registration should be done in subclass */

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }
}
