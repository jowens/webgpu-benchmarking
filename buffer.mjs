import { datatypeToTypedArray } from "./util.mjs";

/** Buffer class
 * Goal: handles all buffer-related tasks
 * Can be a GPUBuffer only, or also have a CPU backing buffer
 * Needs to be able to export necessary info for binding
 *
 * Members:
 * - gpuBuffer (GPUBuffer or GPUBufferBinding)
 * - cpuBuffer (optional)
 */

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
  constructor(args) {
    Object.assign(this, args);
  }
}

export class ReadWriteBuffer extends Buffer {
  constructor(args) {
    super(args);
    this.layoutObjectType = "storage";
  }
}

export class ReadOnlyBuffer extends Buffer {
  constructor(args) {
    super(args);
    this.layoutObjectType = "read-only-storage";
  }
}
