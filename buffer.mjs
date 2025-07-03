import { bitreverse, datatypeToTypedArray, datatypeToBytes } from "./util.mjs";

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
 * In general with buffers we try to use:
 * - _size_ to indicate a byte count
 * - _length_ to indicate an element count
 * Complication is that GPU buffers prefer size and CPU buffers
 *   prefer length, so there's getter code that converts between
 *   the two.
 */

export class Buffer {
  #gpuBuffer; /* this ALWAYS stores a GPUBufferBinding */
  #mappableGPUBuffer;
  #cpuBuffer;
  #cpuBufferBackup; /* if we overwrite #cpuBuffer */
  #cpuBufferIsDirty; /* compared to cpuBufferBackup */
  #size;
  #length;
  #device;
  constructor(args) {
    this.args = { ...args };
    /* generally expect args to contain datatype and size or length */

    if (args.size !== undefined && args.length !== undefined) {
      console.error("Buffer: Do not specify both length and size, pick one");
    }

    this.label =
      args.label ?? `Buffer (datatype: ${this.datatype}; size: ${this.size})`;

    /* copy known arguments into Buffer class for use in init calls */
    const knownArgs = ["datatype", "size", "length", "device"];

    for (const knownArg of knownArgs) {
      if (args[knownArg]) {
        switch (knownArg) {
          case "size":
            this.#size = args[knownArg];
            break;
          case "length":
            this.#length = args[knownArg];
            break;
          case "device":
            this.#device = args[knownArg];
            break;
          default:
            this[knownArg] = args[knownArg];
            break;
        }
      }
    }

    /* we can optionally create (and optionally initialize) a CPUBuffer too */
    /* we do this first in case we need to initialize the GPUBuffer */
    /* we have made these allocations separate function calls in case we
     * need to make them outside of the constructor, so we copy whatever
     * we need from args into the buffer instance */
    if (this.args.createCPUBuffer) {
      this.createCPUBuffer(this.args);
    }

    if (this.args.createGPUBuffer) {
      this.createGPUBuffer(this.args);
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

  createCPUBuffer(args = {}) {
    const length = args.length ?? this.length;
    if (length === undefined) {
      console.error("Buffer::allocateCPUBuffer: must specify length");
    }
    const datatype = args.datatype ?? this.datatype;
    if (datatype === undefined) {
      console.error("Buffer::allocateCPUBuffer: must specify datatype");
    }
    this.#cpuBuffer = new (datatypeToTypedArray(this.datatype))(this.length);
    const is64Bit = datatype === "u64";
    this.#cpuBufferIsDirty = true;
    if (args.initializeCPUBuffer) {
      for (let i = 0; i < length; i++) {
        let val;
        switch (datatype) {
          case "f32":
            switch (args.initializeCPUBuffer) {
              case "randomizeMinusOneToOne":
                /* [-1, 1] */
                val = Math.random() * 2.0 - 1.0;
                break;
              case "randomizeAbsUnder1024":
                // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random#getting_a_random_integer_between_two_values
                /* [-1024, 1024], ints only */
                val = Math.floor(Math.random() * 2049.0 - 1024);
                break;
              case "fisher-yates":
              default:
                /* roughly, range of 32b significand */
                val = i & (2 ** 22 - 1);
                break;
            }
            this.#cpuBuffer[i] = val;
            break;
          case "u32":
          case "i32":
          case "u64":
            switch (args.initializeCPUBuffer) {
              case "xor-beef":
                val = i ^ 0xbeef;
                break;
              case "randomizeAbsUnder1024":
                // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random#getting_a_random_integer_between_two_values
                /* [-1024, 1024], ints only */
                val = Math.floor(Math.random() * 2049.0 - 1024);
                break;
              case "constant":
                val = 42;
                break;
              case "bitreverse":
                val = bitreverse(i);
                break;
              case "randomBytes": {
                let temp = BigInt(0);
                for (let j = 0; j < (is64Bit ? 8 : 4); j++) {
                  const randByte = BigInt(Math.floor(Math.random() * 256.0));
                  temp = (temp << 8n) | randByte;
                }
                if (is64Bit) {
                  val = temp;
                } else {
                  val = Number(temp);
                }
                break;
              }
              case "fisher-yates":
              default:
                if (is64Bit) {
                  val = BigInt(i);
                } else {
                  val = i == 0 ? 0 : this.#cpuBuffer[i - 1] + 1; // trying to get u32s
                }
                break;
            }
            break;
        }
        this.#cpuBuffer[i] = is64Bit ? BigInt(val) : val;
      }
      /* now post-process the array */
      switch (args.initializeCPUBuffer) {
        case "fisher-yates": {
          const shuffleArray = function (array) {
            for (let i = array.length - 1; i > 0; i--) {
              // Generate a random index from 0 to i inclusive
              const j = Math.floor(Math.random() * (i + 1));

              // Swap elements at i and j
              [array[i], array[j]] = [array[j], array[i]];
            }
          };
          shuffleArray(this.#cpuBuffer);
          break;
        }
        default:
          break;
      }
      /* we have now populated #cpuBuffer */
      if (args.storeCPUBackup) {
        this.#cpuBufferBackup = this.#cpuBuffer.slice();
      }
      this.#cpuBufferIsDirty = false;
    }
  }

  createGPUBuffer(args = {}) {
    if ("buffer" in args) {
      console.error(
        "Buffer::allocateGPUBuffer: don't pass in a buffer AND specify createGPUBuffer"
      );
    }
    const device = args.device ?? this.device;
    if (device === undefined) {
      console.error("Buffer::allocateGPUBuffer: must specify device");
    }
    const size = args.size ?? this.size;
    if (size === undefined) {
      console.error("Buffer::allocateGPUBuffer must specify size");
    }
    const datatype = args.datatype ?? this.datatype;
    if (datatype === undefined) {
      console.error("Buffer::allocateGPUBuffer must specify datatype");
    }
    const usage = args.usage ?? this.usage;
    if (this.buffer) {
      console.info("Buffer::createGPUBuffer: Destroying", this.buffer.label);
      this.buffer.destroy();
    }
    this.device.pushErrorScope("out-of-memory");
    this.buffer = this.device.createBuffer({
      label: this.label,
      size: size,
      /** Output-only buffers may not need COPY_DST but we might
       * initialize them with a buffer copy, so be safe and set it
       *
       * This might be a performance issue if COPY_DST costs something
       */
      usage:
        usage ??
        GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
    });
    this.device.popErrorScope().then((error) => {
      if (error) {
        // error is a GPUOutOfMemoryError object instance
        this.buffer = null;
        console.error(
          `Buffer::createGPUBuffer: Out of memory, buffer too large. Error: ${error.message}`
        );
      }
    });
    if (args.initializeGPUBuffer) {
      this.copyCPUToGPU();
    }
    if (args.createMappableGPUBuffer) {
      this.createMappableGPUBuffer(size);
    }
  }

  createMappableGPUBuffer(size) {
    if (this.#mappableGPUBuffer) {
      this.#mappableGPUBuffer.destroy();
    }
    this.device.pushErrorScope("out-of-memory");
    this.#mappableGPUBuffer = this.device.createBuffer({
      label: "mappable | " + this.label,
      size: size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.device.popErrorScope().then((error) => {
      if (error) {
        // error is a GPUOutOfMemoryError object instance
        this.buffer = null;
        console.error(
          `Buffer::createMappableGPUBuffer: Out of memory, buffer too large. Error: ${error.message}`
        );
      }
    });
  }

  copyCPUToGPU() {
    if (this.#cpuBuffer == undefined) {
      console.error("Buffer::copyCPUToGPU requires a CPUBuffer");
    }
    this.device.queue.writeBuffer(
      this.buffer.buffer,
      this.buffer.offset ?? 0,
      this.#cpuBuffer
    );
  }

  async copyGPUToCPU() {
    // copy buffer into mappable buffer ...
    const copyEncoder = this.device.createCommandEncoder({
      label: `encoder: GPU buffer ${this.label} data -> mappable buffers`,
    });
    if (this.#mappableGPUBuffer === undefined) {
      this.createMappableGPUBuffer(this.size);
    }
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
    try {
      this.#cpuBuffer = new (datatypeToTypedArray(this.datatype))(
        this.#mappableGPUBuffer.getMappedRange().slice()
      );
    } catch (error) {
      console.error(
        error,
        "Buffer::copyGPUToCPU: tried to allocate and copy array of length",
        this.#mappableGPUBuffer.getMappedRange().slice().length
      );
    }

    this.#mappableGPUBuffer.unmap();
  }

  copyCPUBackupToCPU() {
    if (this.args.storeCPUBackup && this.#cpuBufferIsDirty) {
      this.#cpuBuffer = this.#cpuBufferBackup.slice();
      this.#cpuBufferIsDirty = false;
    }
  }

  destroy() {
    if (this.#gpuBuffer && this.#gpuBuffer.buffer) {
      this.#gpuBuffer.buffer.destroy();
    }
    if (this.#mappableGPUBuffer) {
      this.#mappableGPUBuffer.destroy();
    }
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
  get cpuBufferBackup() {
    return this.#cpuBufferBackup;
  }
  /* worried about an infinite loop in the below */
  get size() {
    /* returns size in bytes */
    return (
      this.#gpuBuffer?.size ??
      this.#gpuBuffer?.buffer?.size ??
      this.#size ??
      this.#length * datatypeToBytes(this.datatype)
    );
  }
  get length() {
    /* returns number of items */
    if (this.size) {
      return this.size / datatypeToBytes(this.datatype);
    } else {
      return this.#length;
    }
  }
  get device() {
    return this.#device;
  }
}
