import { datatypeToTypedArray } from "./util.mjs";

export function toGPUBufferBinding(obj) {
  switch (obj.constructor) {
    case GPUBufferBinding:
      return obj;
    case GPUBuffer:
    default:
      return GPUBufferBinding(obj);
  }
}

export function getBufferSize(obj) {
  if (obj.size) {
    return obj.size;
  } else if (obj.buffer.size) {
    return obj.buffer.size;
  } else {
    return -1;
  }
}

class Buffer {
  constructor(args) {}
}

export class InputBuffer extends Buffer {
  constructor(args) {
    super(args);
  }
}

export class OutputBuffer extends Buffer {
  constructor(args) {
    super(args);
  }
}
