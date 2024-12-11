// this is only for benchmarking purposes, not designed for production
class TestBuffer {
  constructor(device, config) {
    Object.assign(this, config);
    /* expect config to contain datatype and size */
    this.device = device;
    if (!("label" in config)) {
      this.label = `Buffer (datatype: ${config.datatype}; size: ${config.size})`;
    }
    this.cpuBuffer = new (this.getArrayType())(this.size);
  }
  getArrayType() {
    switch (this.datatype) {
      case "f32":
        return Float32Array;
      case "i32":
        return Int32Array;
      case "u32":
        return Uint32Array;
    }
    return undefined;
  }
}
export class TestInputBuffer extends TestBuffer {
  constructor(device, config) {
    super(device, config);
    this.type = "input";

    // since we're input, fill the buffer with useful data
    for (let i = 0; i < this.memsrcSize; i++) {
      if (this.datatype == "f32") {
        this.cpuBuffer[i] = config?.randomizeInput
          ? Math.random() * 2.0 - 1.0
          : i & (2 ** 22 - 1);
        // Rand: [-1,1]; non-rand: roughly, range of 32b significand
      } else if (this.datatype == "u32") {
        this.cpuBuffer[i] = i == 0 ? 0 : memsrcu32[i - 1] + 1; // trying to get u32s
      }
      // otherwise, initialize nothing
    }
    this.gpuBuffer = device.createBuffer({
      label: this.label,
      size: this.cpuBuffer.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.gpuBuffer, 0, this.cpuBuffer);
  }
}
export class TestOutputBuffer extends TestBuffer {
  constructor(device, config) {
    super(device, config);
    this.type = "output";

    this.gpuBuffer = device.createBuffer({
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

    this.mappableGPUBuffer = device.createBuffer({
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
