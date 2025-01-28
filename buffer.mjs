import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";

/** Buffer class
 * Goal: handles all buffer-related tasks
 * Required: GPUBuffer, which handles all queries (e.g., size)
 * Can be a GPUBuffer only, or also have a CPU backing buffer
 * Needs to be able to export necessary info for binding
 *
 * Members:
 * - #gpuBuffer (GPUBufferBinding)
 * - #cpuBuffer (optional)
 *
 * size: number of elements (not number of bytes)
 *
 * TODO: initialize call could take different strings to
 *   do different initalizations (all 1s, random, distribution, etc.)
 */

export class Buffer {
  #gpuBuffer; /* this ALWAYS stores a GPUBufferBinding */
  #mappableGPUBuffer;
  #cpuBuffer;
  constructor(args) {
    this.args = { ...args };
    /* generally expect args to contain datatype and size */

    this.label =
      args.label ?? `Buffer (datatype: ${this.datatype}; size: ${this.size})`;

    /* we can optionally create (and optionally initialize) a CPUBuffer too */
    /* we do this first in case we need to initialize the GPUBuffer */
    if (this.args.createCPUBuffer) {
      if (!("size" in args)) {
        console.error(
          `Buffer: if createCPUBuffer is true, must also specify size`
        );
      }
      if (!("datatype" in args)) {
        console.error(
          `Buffer: if createCPUBuffer is true, must also specify datatype`
        );
      }
      this.#cpuBuffer = new (datatypeToTypedArray(this.datatype))(
        this.args.size
      );
      if (this.args.initializeCPUBuffer) {
        // since we're input, fill the buffer with useful data
        for (let i = 0; i < this.args.size; i++) {
          if (this.datatype == "f32") {
            this.#cpuBuffer[i] =
              this.args.initializeCPUBuffer === "randomize"
                ? Math.random() * 2.0 - 1.0
                : i & (2 ** 22 - 1);
            // Rand: [-1,1]; non-rand: roughly, range of 32b significand
          } else if (this.datatype == "u32") {
            this.#cpuBuffer[i] = i == 0 ? 0 : this.#cpuBuffer[i - 1] + 1; // trying to get u32s
          }
          // otherwise, initialize nothing
        }
      }
    }

    if (this.args.createGPUBuffer) {
      if ("buffer" in this.args) {
        console.error(
          "Buffer: don't pass in a buffer AND specify createGPUBuffer"
        );
      }
      if (!("device" in this.args)) {
        console.error("Buffer: must specify device if createGPUBuffer is true");
      }
      if (!("size" in this.args)) {
        console.error(
          `Buffer: if createGPUBuffer is true, must also specify size`
        );
      }
      if (!("datatype" in this.args)) {
        console.error(
          `Buffer: if createGPUBuffer is true, must also specify datatype`
        );
      }
      this.buffer = this.device.createBuffer({
        label: this.label,
        size: this.args.size * datatypeToBytes(this.args.datatype),
        /** Output-only buffers may not need COPY_DST but we might
         * initialize them with a buffer copy, so be safe and set it
         *
         * This might be a performance issue if COPY_DST costs something
         */
        usage:
          this.args.usage ??
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
      });
      if (this.args.initializeGPUBuffer) {
        if (this.#cpuBuffer == undefined) {
          console.error("Buffer: initializeGPUBuffer requires a CPUBuffer");
        }
        this.device.queue.writeBuffer(
          this.buffer.buffer,
          this.buffer.offset ?? 0,
          this.#cpuBuffer
        );
      }
      if (this.args.createMappableGPUBuffer) {
        this.#mappableGPUBuffer = this.device.createBuffer({
          label: "mappable |" + this.label,
          size: this.size,
          usage:
            this.args.mappableGPUBufferUsage ??
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }
    } else {
      this.buffer = args.buffer;
    }
    if (this.#gpuBuffer === undefined) {
      console.warn(
        "Buffer constructor:",
        this.label,
        "GPU buffer is undefined"
      );
    }
  }

  async copyGPUToCPU() {
    // copy buffer into mappable buffer ...
    const copyEncoder = this.device.createCommandEncoder({
      label: "encoder: GPU output buffer data -> mappable buffers",
    });
    copyEncoder.copyBufferToBuffer(
      this.buffer.buffer,
      this.buffer.offset ?? 0,
      this.#mappableGPUBuffer,
      0,
      this.#mappableGPUBuffer.size
    );
    const copyCommandBuffer = copyEncoder.finish();
    this.device.queue.submit([copyCommandBuffer]);

    // ... then back to host
    await this.#mappableGPUBuffer.mapAsync(GPUMapMode.READ);
    this.#cpuBuffer = new (datatypeToTypedArray(this.datatype))(
      this.#mappableGPUBuffer.getMappedRange().slice()
    );
    this.#mappableGPUBuffer.unmap();
  }

  set buffer(b) {
    if (b?.buffer) {
      /* this is already a GPUBufferBinding */
      this.#gpuBuffer = b;
    } else {
      /* this is a GPUBuffer or undefined, make it a GPUBufferBinding */
      this.#gpuBuffer = { buffer: b };
    }
  }
  get buffer() {
    return this.#gpuBuffer;
  }
  get cpuBuffer() {
    return this.#cpuBuffer;
  }
  get datatype() {
    return this.args.datatype;
  }
  get size() {
    /* returns size in bytes */
    return (
      this.#gpuBuffer?.size ?? this.#gpuBuffer?.buffer?.size ?? this.args.size
    );
  }
  get items() {
    /* returns number of items */
    return this.size / datatypeToBytes(this.datatype);
  }
  get device() {
    return this.args.device;
  }
}

class BufferWithCPU extends Buffer {
  /* this class uses this.cpuBuffer */
  constructor(args) {
    /** don't pass "size" to parent constructor, the gpuBuffer constructor
     * has no notion of that, an arg of size is only used for the CPU buffer
     */

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
