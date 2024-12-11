import { TestInputBuffer, TestOutputBuffer } from "./testbuffer.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";

export class BasePrimitive {
  constructor(args) {
    // expect that args are:
    // { device: device,
    //   params: { param1: val1, param2: val2 },
    //   someConfigurationSetting: thatSetting,
    //   gputimestamps: true,
    //   uniforms: uniformbuffer0,
    //   inputs: [inputbuffer0, inputbuffer1],
    //   outputs: outputbuffer0,
    // }
    if (this.constructor === BasePrimitive) {
      throw new Error(
        "Cannot instantiate abstract class BasePrimitive directly."
      );
    }

    /* required arguments, and handled first */
    for (const field of ["device"]) {
      if (!(field in args)) {
        throw new Error(
          `Primitive ${this.constructor.name} requires a "${field}" argument.`
        );
      }
      /* set this first so that it can be used below */
      this[field] = args[field];
    }
    /* arguments that could be objects or arrays */
    for (const field of ["uniforms", "inputs", "outputs"]) {
      this[`#{field}`] = undefined;
    }

    // now let's walk through all the fields
    for (const [field, value] of Object.entries(args)) {
      switch (field) {
        case "params":
          /* paste these directly into the primitive (flattened) */
          Object.assign(this, args.params);
          break;
        case "gputimestamps":
          /* only set this if BOTH it's requested AND it's enabled in the device */
          this.gputimestamps =
            this.device.features.has("timestamp-query") && value;
          break;
        case "device":
          /* do nothing, handled above */
          break;
        case "uniforms":
        case "inputs":
        case "outputs": // fall-through deliberate
        default:
          this[field] = value;
          break;
      }
    }
  }
  #uniforms;
  #inputs;
  #outputs;
  set uniforms(val) {
    const arr = Array.isArray(val) ? val : [val];
    this.#uniforms = arr;
  }
  get uniforms() {
    return this.#uniforms;
  }
  set inputs(val) {
    const arr = Array.isArray(val) ? val : [val];
    this.#inputs = arr;
  }
  get inputs() {
    return this.#inputs;
  }
  set outputs(val) {
    const arr = Array.isArray(val) ? val : [val];
    this.#outputs = arr;
  }
  get outputs() {
    return this.#outputs;
  }
  #timingHelper;

  kernel() {
    /* call this from a subclass instead */
    throw new Error("Cannot call kernel() from abstract class BasePrimitive.");
  }
  getDispatchGeometry() {
    /* call this from a subclass instead */
    throw new Error(
      "Cannot call getDispatchGeometry() from abstract class BasePrimitive."
    );
  }
  getGPUBufferFromBinding(binding) {
    /**
     * Input is some sort of buffer object. Currently recognized:
     * - GPUBuffer
     * - *TestBuffer
     * Returns a GPUBuffer
     */
    let outputBuffer;
    switch (binding.constructor) {
      case GPUBuffer:
        outputBuffer = binding;
        break;
      case TestInputBuffer: // fallthrough deliberate
      case TestOutputBuffer:
        outputBuffer = binding.gpuBuffer;
        break;
      default:
        console.error(
          `Primitive:getGPUBinding: Unknown datatype for buffer: ${typeof binding}`
        );
        break;
    }
    return outputBuffer;
  }
  async execute() {
    const encoder = this.device.createCommandEncoder({
      label: `${this.constructor.name} primitive encoder`,
    });

    /** loop through each of the actions listed in this.compute(),
     * instantiating whatever WebGPU constructs are necessary to run them
     *
     * TODO: memoize these constructs
     */

    /* begin timestamp prep */
    let kernelCount = 0; // how many kernels are there total?
    if (this.gputimestamps) {
      for (const action of this.compute()) {
        if (action.constructor == Kernel) {
          kernelCount++;
        }
      }
    }
    this.#timingHelper = new TimingHelper(this.device, kernelCount);

    for (const action of this.compute()) {
      switch (action.constructor) {
        case Kernel:
          const computeModule = this.device.createShaderModule({
            label: `module: ${this.constructor.name}`,
            code: action.kernel(),
          });

          const kernelPipeline = this.device.createComputePipeline({
            label: `${this.constructor.name} compute pipeline`,
            layout: "auto",
            compute: {
              module: computeModule,
            },
          });

          const bindings = [];
          let bindingIdx = 0;
          for (const buffer of ["uniforms", "outputs", "inputs"]) {
            if (this[buffer] !== undefined) {
              for (const binding of this[buffer]) {
                bindings.push({
                  binding: bindingIdx++,
                  resource: { buffer: this.getGPUBufferFromBinding(binding) },
                });
              }
            }
          }

          const kernelBindGroup = this.device.createBindGroup({
            label: `bindGroup for ${this.constructor.name} kernel`,
            layout: kernelPipeline.getBindGroupLayout(0),
            entries: bindings,
          });

          const kernelDescriptor = {
            label: `${this.constructor.name} compute pass`,
          };

          const kernelPass = this.gputimestamps
            ? this.#timingHelper.beginComputePass(encoder, kernelDescriptor)
            : encoder.beginComputePass(kernelDescriptor);

          kernelPass.setPipeline(kernelPipeline);
          kernelPass.setBindGroup(0, kernelBindGroup);
          kernelPass.dispatchWorkgroups(...this.getDispatchGeometry());
          kernelPass.end();
          break;
        case InitializeMemoryBlock:
          let DatatypeArray;
          switch (action.datatype) {
            case "f32":
              DatatypeArray = Float32Array;
              break;
            case "i32":
              DatatypeArray = Int32Array;
              break;
            case "u32":
            default:
              DatatypeArray = Uint32Array;
              break;
          }
          /* initialize entire array to action.value ... */
          const initBlock = DatatypeArray.from(
            { length: action.buffer.size / DatatypeArray.BYTES_PER_ELEMENT },
            () => action.value
          );

          /* ... then write it into the buffer */
          this.device.queue.writeBuffer(
            this.getGPUBufferFromBinding(action.buffer),
            0 /* offset */,
            initBlock
          );
          break;
      }
    }

    const commandBuffer = encoder.finish();
    this.device.queue.submit([commandBuffer]);
    await this.device.queue.onSubmittedWorkDone();
  }
  async getResult() {
    return this.#timingHelper.getResult();
  }
}

export class Kernel {
  constructor(k) {
    this.kernel = k;
  }
}

export class InitializeMemoryBlock {
  constructor(buffer, value, datatype) {
    this.buffer = buffer;
    this.value = value;
    this.datatype = datatype;
  }
}
