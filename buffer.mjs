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

export class Buffer {
  // "gpuBuffer" is a GPUBufferBinding
  constructor(args) {
    Object.assign(this, args);
    /* generally expect args to contain datatype and size */
    if (!("label" in args)) {
      this.label = `Buffer (datatype: ${this.datatype}; size: ${this.size})`;
    }
  }

  set buffer(b) {
    if (b?.buffer) {
      /* this is already a GPUBufferBinding */
      this.gpuBuffer = b;
    } else {
      /* this is a GPUBuffer or undefined, make it a GPUBufferBinding */
      this.gpuBuffer = { buffer: b };
    }
  }
  get buffer() {
    return this.gpuBuffer;
  }

  get size() {
    return this.gpuBuffer?.size ?? this.gpuBuffer?.buffer?.size;
  }
}

class BufferWithCPU extends Buffer {
  /* this class uses this.cpuBuffer */
  constructor(args) {
    /** don't pass "size" to parent constructor, the gpuBuffer constructor
     * has no notion of that, an arg of size is only used for the CPU buffer
     */
    if (!("device" in args)) {
      console.error("A CreateXBuffer must be initialized with a device");
    }
    if (!("label" in args)) {
      console.error(
        "A CreateXBuffer must be initialized with a label, likely one of the knownBuffers associated with the primitive"
      );
    }
    const { size, ...argsNotSize } = args;
    super(argsNotSize);
    this.cpuBuffer = new (datatypeToTypedArray(this.datatype))(size);
  }
}

export class CreateInputBufferWithCPU extends BufferWithCPU {
  constructor(args) {
    super(args);

    // since we're input, fill the buffer with useful data
    for (let i = 0; i < args.size; i++) {
      if (this.datatype == "f32") {
        this.cpuBuffer[i] = args?.randomizeInput
          ? Math.random() * 2.0 - 1.0
          : i & (2 ** 22 - 1);
        // Rand: [-1,1]; non-rand: roughly, range of 32b significand
      } else if (this.datatype == "u32") {
        this.cpuBuffer[i] = i == 0 ? 0 : this.cpuBuffer[i - 1] + 1; // trying to get u32s
      }
      // otherwise, initialize nothing
    }
    this.gpuBuffer = this.device.createBuffer({
      label: this.label,
      size: this.cpuBuffer.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.gpuBuffer, 0, this.cpuBuffer);
  }
}

export class CreateOutputBufferWithCPU extends BufferWithCPU {
  constructor(args) {
    super(args);
    this.gpuBuffer = this.device.createBuffer({
      label: this.label,
      size: this.cpuBuffer.byteLength,
      /** Output-only buffers may not need COPY_DST but we might
       * initialize them with a buffer copy, so be safe and set it
       *
       * This might be a performance issue if COPY_DST costs something
       */
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    this.mappableGPUBuffer = this.device.createBuffer({
      label: "mappable memory destination buffer",
      size: this.cpuBuffer.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }
}
