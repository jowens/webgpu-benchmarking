import { datatypeToTypedArray } from "./util.mjs";

// this is only for benchmarking purposes, not designed for production
class TestBuffer {
  constructor(args) {
    Object.assign(this, args);
    if (!("device" in args)) {
      console.error("A TestBuffer must be initialized with a device");
    }
    /* expect args to contain datatype and size */
    if (!("label" in args)) {
      this.label = `Buffer (datatype: ${args.datatype}; size: ${args.size})`;
    }
    this.cpuBuffer = new (datatypeToTypedArray(this.datatype))(this.size);
  }
}

export class TestInputBuffer extends TestBuffer {
  constructor(args) {
    super(args);
    this.type = "input";

    // since we're input, fill the buffer with useful data
    for (let i = 0; i < this.size; i++) {
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
export class TestOutputBuffer extends TestBuffer {
  constructor(args) {
    super(args);
    this.type = "output";

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
export class TestUniformsBuffer extends TestBuffer {
  constructor(device, config) {
    super(device, config);
    this.type = "uniforms";
    // unimplemented yet
  }
}
