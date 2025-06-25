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
 *
 * TODO remove next line when fixed
 * size: number of elements (not number of bytes)
 *
 * TODO: initialize call could take different strings to
 *   do different initalizations (all 1s, random, distribution, etc.)
 */

export class Buffer {
  #gpuBuffer; /* this ALWAYS stores a GPUBufferBinding */
  #mappableGPUBuffer;
  #cpuBuffer;
  #cpuBufferBackup; /* if we overwrite #cpuBuffer */
  #cpuBufferIsDirty; /* compared to cpuBufferBackup */
  constructor(args) {
    this.args = { ...args };
    /* generally expect args to contain datatype and size or length */

    this.label =
      args.label ?? `Buffer (datatype: ${this.datatype}; size: ${this.size})`;

    if (args.datatype) {
      this.datatype = args.datatype;
    }

    /* we can optionally create (and optionally initialize) a CPUBuffer too */
    /* we do this first in case we need to initialize the GPUBuffer */
    if (this.args.createCPUBuffer) {
      if (!("size" in args) && !("length" in args)) {
        console.error(
          "Buffer: if createCPUBuffer is true, must also specify size or length"
        );
      }
      if ("size" in args && "length" in args) {
        console.error(
          "Buffer: please only specify one of size (bytes) or length (elements)"
        );
      }
      if (!("datatype" in args)) {
        console.error(
          `Buffer: if createCPUBuffer is true, must also specify datatype`
        );
      }
      this.#cpuBuffer = new (datatypeToTypedArray(this.datatype))(this.length);
      const is64Bit = this.datatype === "u64";
      this.#cpuBufferIsDirty = true;
      if (this.args.initializeCPUBuffer) {
        for (let i = 0; i < this.length; i++) {
          let val;
          switch (this.datatype) {
            case "f32":
              switch (this.args.initializeCPUBuffer) {
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
              switch (this.args.initializeCPUBuffer) {
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
        switch (this.args.initializeCPUBuffer) {
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

    if (this.args.createGPUBuffer) {
      if ("buffer" in this.args) {
        console.error(
          "Buffer: don't pass in a buffer AND specify createGPUBuffer"
        );
      }
      if (!("device" in this.args)) {
        console.error("Buffer: must specify device if createGPUBuffer is true");
      }
      if (!("size" in args) && !("length" in args)) {
        console.error(
          "Buffer: if createGPUBuffer is true, must also specify size or length"
        );
      }
      if ("size" in args && "length" in args) {
        console.error(
          "Buffer: please only specify one of size (bytes) or length (elements)"
        );
      }
      if (!("datatype" in this.args)) {
        console.error(
          `Buffer: if createGPUBuffer is true, must also specify datatype`
        );
      }
      this.buffer = this.device.createBuffer({
        label: this.label,
        size: this.size,
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
        this.copyCPUToGPU();
      }
      if (this.args.createMappableGPUBuffer) {
        this.#mappableGPUBuffer = this.device.createBuffer({
          label: "mappable | " + this.label,
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
      this.args?.size ??
      this.args?.length * datatypeToBytes(this.datatype)
    );
  }
  get length() {
    /* returns number of items */
    if (this.size) {
      return this.size / datatypeToBytes(this.datatype);
    } else {
      return this.args.length;
    }
  }
  get device() {
    return this.args.device;
  }
}
