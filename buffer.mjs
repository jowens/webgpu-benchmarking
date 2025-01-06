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

export class Buffer {
  #buffer; // this is a GPUBufferBinding
  constructor(args) {
    Object.assign(this, args);
  }

  set buffer(b) {
    if (b?.buffer) {
      /* this is already a GPUBufferBinding */
      this.#buffer = b;
    } else {
      /* this is a GPUBuffer or undefined, make it a GPUBufferBinding */
      this.#buffer = { buffer: b };
    }
  }
  get buffer() {
    return this.#buffer;
  }

  get size() {
    return this.#buffer.size ?? this.#buffer.buffer.size;
  }
}

class ReadWriteBuffer extends Buffer {
  constructor(args) {
    super(args);
    this.layoutObjectType = "storage";
  }
}

class ReadOnlyBuffer extends Buffer {
  constructor(args) {
    super(args);
    this.layoutObjectType = "read-only-storage";
  }
}
