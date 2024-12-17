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
      this[`#${field}`] = undefined;
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

    this.bindings = [];
  }
  /**
   * #{uniforms, inputs, outputs} are ALWAYS arrays of buffers. Setting
   *     a single buffer converts to an array of size 1.
   */
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
  getSimpleDispatchGeometry() {
    // todo this is too simple
    let dispatchGeometry = [this.workgroupCount, 1];
    while (
      dispatchGeometry[0] > this.device.limits.maxComputeWorkgroupsPerDimension
    ) {
      /* too big */
      dispatchGeometry[0] = Math.ceil(dispatchGeometry[0] / 2);
      dispatchGeometry[1] *= 2;
    }
    return dispatchGeometry;
  }
  getGPUBufferFromBinding(binding) {
    /**
     * Input is some sort of buffer object. Currently recognized:
     * - GPUBuffer
     * - *TestBuffer
     * Returns a GPUBuffer
     */
    let gpuBuffer;
    switch (binding.constructor) {
      case GPUBuffer:
        gpuBuffer = binding;
        break;
      case TestInputBuffer: // fallthrough deliberate
      case TestOutputBuffer:
        gpuBuffer = binding.gpuBuffer;
        break;
      default:
        console.error(
          `Primitive:getGPUBufferFromBinding: Unknown datatype for buffer: ${typeof binding}`
        );
        break;
    }
    return gpuBuffer;
  }
  getCPUBufferFromBinding(binding) {
    /**
     * Input is some sort of buffer object. Currently recognized:
     * - TypedArray
     * - *TestBuffer
     * Returns a TypedArray
     */
    let cpuBuffer;
    switch (binding.constructor) {
      case TestInputBuffer: // fallthrough deliberate
      case TestOutputBuffer:
        cpuBuffer = binding.cpuBuffer;
        break;
      case Uint32Array:
      case Int32Array:
      case Float32Array:
        cpuBuffer = binding;
      default:
        console.error(
          `Primitive:getCPUBufferFromBinding: Unknown datatype for buffer: ${typeof binding}`
        );
        break;
    }
    return cpuBuffer;
  }
  /**
   * Ensures output is an object with members "in", "out" where
   *     each member is an array of TypedArrays
   * output: { "in": [TypedArray], "out": [TypedArray] }
   * if input is not in that format, bindingsToTypedArrays converts it
   * it should convert inputs of:
   *     { "in": [somethingBufferish], "out": [somethingBufferish] }
   *     or
   *     { "in": TypedArray, "out": TypedArray }
   */
  bindingsToTypedArrays(bindings) {
    // https://stackoverflow.com/questions/65824084/how-to-tell-if-an-object-is-a-typed-array
    const TypedArray = Object.getPrototypeOf(Uint8Array);
    const bindingsOut = {};
    for (const type of ["in", "out"]) {
      if (bindings[type] instanceof TypedArray) {
        /* already a typed array! */
        bindingsOut[type] = [bindings[type]];
      } else {
        /* array of something we need to convert */
        bindingsOut[type] = bindings[type].map(this.getCPUBufferFromBinding);
      }
    }
    return bindingsOut;
  }
  async execute(args) {
    /** loop through each of the actions listed in this.compute(),
     * instantiating whatever WebGPU constructs are necessary to run them
     *
     * TODO: memoize these constructs
     */

    /* begin timestamp prep - count kernels, allocate 2 timestamps/kernel */
    let kernelCount = 0; // how many kernels are there total?
    if (this.gputimestamps) {
      for (const action of this.compute()) {
        if (action.constructor == Kernel && action.enabled()) {
          kernelCount++;
        }
      }
    }

    /* populate the u/o/i declared bindings before we process any new ones */

    let bindingIdx = 0;
    for (const buffer of ["uniforms", "outputs", "inputs"]) {
      if (this[buffer] !== undefined) {
        for (const binding of this[buffer]) {
          this.bindings.push({
            binding: bindingIdx++,
            resource: { buffer: this.getGPUBufferFromBinding(binding) },
          });
        }
      }
    }

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }, // {"uniform", "storage", "read-only-storage"}
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    for (let trial = 0; trial < args.trials ?? 1; trial++) {
      const encoder = this.device.createCommandEncoder({
        label: `${this.constructor.name} primitive encoder`,
      });
      this.#timingHelper = new TimingHelper(this.device, kernelCount);
      for (const action of this.compute()) {
        switch (action.constructor) {
          case Kernel:
            if (!action.enabled()) {
              /* don't run this kernel at all */
              break;
            }
            /* action.kernel can be a string or a function */
            let kernelString;
            switch (typeof action.kernel) {
              case "string":
                kernelString = action.kernel;
                break;
              case "function":
                /* have never used kernelArgs, but let's support it anyway */
                kernelString = action.kernel(action.kernelArgs);
                break;
              default:
                throw new Error(
                  "Primitive::Kernel: kernel must be a function or a string"
                );
                break;
            }
            if (action.debugPrintKernel) {
              console.log(kernelString);
            }

            const computeModule = this.device.createShaderModule({
              label: `module: ${this.constructor.name}`,
              code: kernelString,
            });

            const kernelPipeline = this.device.createComputePipeline({
              label: `${this.constructor.name} compute pipeline`,
              layout: pipelineLayout, //"auto",
              compute: {
                module: computeModule,
              },
            });

            const kernelBindGroup = this.device.createBindGroup({
              label: `bindGroup for ${this.constructor.name} kernel`,
              layout: kernelPipeline.getBindGroupLayout(0),
              entries: this.bindings,
            });

            const kernelDescriptor = {
              label: `${this.constructor.name} compute pass`,
            };

            const kernelPass = this.gputimestamps
              ? this.#timingHelper.beginComputePass(encoder, kernelDescriptor)
              : encoder.beginComputePass(kernelDescriptor);

            kernelPass.setPipeline(kernelPipeline);
            kernelPass.setBindGroup(0, kernelBindGroup);
            /* For binding geometry:
             *     Look in kernel first, then to primitive if nothing in kernel
             * There was some binding wonkiness with using ?? to pick the gDG call
             * so that's why there's an if statement instead
             *
             * todo: dispatchGeometry should be able to be an array or number, not
             *     just a function
             * */
            let dispatchGeometry;
            if (action.getDispatchGeometry) {
              dispatchGeometry = action.getDispatchGeometry();
            } else {
              dispatchGeometry = this.getDispatchGeometry();
            }
            kernelPass.dispatchWorkgroups(...dispatchGeometry);

            console.info(`memsrcSize: ${this.memsrcSize}
workgroupCount: ${this.workgroupCount}
workgroupSize: ${this.workgroupSize}
maxGSLWorkgroupCount: ${this.maxGSLWorkgroupCount}
dispatchGeometry: ${dispatchGeometry}`);
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
            if (typeof action.buffer === "string") {
              /** if we specify a buffer by a string,
               * go find the actual buffer associated with that string.
               * pick first one we find, in bindings order */
              for (const binding of this.bindings) {
                if (binding.resource.buffer.label == action.buffer) {
                  action.buffer = binding.resource.buffer;
                  break;
                }
              }
              if (typeof action.buffer === "string") {
                /* we didn't find a buffer; the string didn't match any of them */
                throw new Error(
                  `Primitive::InitializeMemoryBlock: Could not find buffer named ${action.buffer}.`
                );
              }
            }
            /* initialize entire CPU-side array to action.value ... */
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
          case AllocateBuffer:
            console.log(this);
            const allocatedBuffer = this.device.createBuffer({
              label: action.label,
              size: action.size,
              usage:
                action.usage ??
                /* default: read AND write */
                GPUBufferUsage.STORAGE |
                  GPUBufferUsage.COPY_SRC |
                  GPUBufferUsage.COPY_DST,
            });
            this.bindings.push({
              binding: this.bindings.length,
              resource: { buffer: allocatedBuffer },
            });
            break;
        }
      }

      const commandBuffer = encoder.finish();
      this.device.queue.submit([commandBuffer]);
    }
    await this.device.queue.onSubmittedWorkDone();
  }
  async getResult() {
    return this.#timingHelper.getResult();
  }
}

export class Kernel {
  constructor(args) {
    this.label = "kernel"; // default
    if (typeof args == "function") {
      /* the function takes an optional arg object and returns the kernel string */
      this.kernel = args;
    } else {
      /* more complicated objects */
      /* one field should be "kernel" */
      if (!args.kernel || typeof args.kernel !== "function") {
        throw new Error(
          "Kernel::constructor: Requires a 'kernel' field that is a function that returns the kernel string"
        );
      }
      Object.assign(this, args);
    }
  }
  enabled() {
    /* default: enabled */
    return this.enable ?? true;
  }
}

export class InitializeMemoryBlock {
  constructor(args) {
    Object.assign(this, args);
  }
}

export class AllocateBuffer {
  constructor(args) {
    Object.assign(this, args);
  }
}
