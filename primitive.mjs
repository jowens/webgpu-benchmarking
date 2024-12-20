import { TestInputBuffer, TestOutputBuffer } from "./testbuffer.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";

export class BasePrimitive {
  constructor(args) {
    // expect that args are:
    // { device: device,
    //   params: { param1: val1, param2: val2 }, // TUNABLE params
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
    /* set label first */
    if (!("label" in args)) {
      this.label = this.constructor.name;
    }

    /* required arguments, and handled next */
    for (const requiredField of ["device"]) {
      if (!(requiredField in args)) {
        throw new Error(
          `Primitive ${this.label} requires a "${requiredField}" argument.`
        );
      }
      this[requiredField] = args[requiredField];
    }

    // now let's walk through all the args
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
        default:
          /* copy it into the primitive */
          this[field] = value;
          break;
      }
    }
  }

  #bufferDescription; // this holds all buffer state for the primitive
  #bindGroupLayout;
  #pipelineLayout;
  #timingHelper;

  set bufferDescription(obj) {
    // TODO: have a "defer" entry that doesn't rebuild this if set
    this.#bufferDescription = obj;
    /* now rebuild bindGroupLayout and pipelineLayout */
    const entries = [];
    for (const [binding, type] of Object.entries(this.#bufferDescription)) {
      // "binding" is a string, so turn it back into an int here
      entries.push({
        binding: parseInt(binding, 10),
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type },
      });
    }
    console.log("Entries in set bufferDescription:", entries);
    this.#bindGroupLayout = this.device.createBindGroupLayout({
      entries,
      label: `${this.label} bind group with ${entries.length} entries`,
    });
    this.#pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.#bindGroupLayout],
      label: `${this.label} pipeline layout (${this.#bindGroupLayout.label})`,
    });
    console.log(
      "#bindgrouplayout",
      this.#bindGroupLayout,
      "#pipelinelayout",
      this.#pipelineLayout
    );
  }
  get bufferDescription() {
    return this.#bufferDescription;
  }
  getBindGroupEntries() {
    const entries = [];
    for (const binding in this.bufferDescription) {
      entries.push({ binding, resource: { buffer: this.buffers[binding] } });
    }
    console.log("getBindGroupEntries", entries);
    return entries;
  }

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
  updateBufferInternals() {
    /* rebuild any internal structures with respect to buffers */
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
  async execute(args = {}) {
    /** loop through each of the actions listed in this.compute(),
     * instantiating whatever WebGPU constructs are necessary to run them
     *
     * TODO: memoize these constructs
     */

    /* set up defaults */
    if (!("trials" in args)) {
      args.trials = 1;
    }

    /* begin timestamp prep - count kernels, allocate 2 timestamps/kernel */
    console.log("gputimestamps:", this.gputimestamps);
    let kernelCount = 0; // how many kernels are there total?
    if (this.gputimestamps) {
      for (const action of this.compute()) {
        if (action.constructor == Kernel && action.enabled()) {
          kernelCount++;
        }
      }
    }
    console.log("Kernel count:", kernelCount);

    const encoder = this.device.createCommandEncoder({
      label: `${this.label} primitive encoder`,
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
            label: `module: ${this.label}`,
            code: kernelString,
          });

          console.log(this);

          const kernelPipeline = this.device.createComputePipeline({
            label: `${this.label} compute pipeline`,
            layout: this.#pipelineLayout,
            compute: {
              module: computeModule,
            },
          });

          const kernelBindGroup = this.device.createBindGroup({
            label: `bindGroup for ${this.label} kernel`,
            layout: kernelPipeline.getBindGroupLayout(0),
            entries: this.getBindGroupEntries(),
          });

          const kernelDescriptor = {
            label: `${this.label} compute pass`,
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
           * TODO: dispatchGeometry should be able to be an array or number, not
           *     just a function
           * */
          let dispatchGeometry;
          if (action.getDispatchGeometry) {
            dispatchGeometry = action.getDispatchGeometry();
          } else {
            dispatchGeometry = this.getDispatchGeometry();
          }
          /* trials might be >1 so make sure additional runs are idempotent */
          for (let trial = 0; trial < args.trials ?? 1; trial++) {
            console.log("DISPATCHING:", dispatchGeometry);
            kernelPass.dispatchWorkgroups(...dispatchGeometry);
          }

          console.info(`inputBytes: ${this.inputBytes}
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
            for (const buffer of this.buffers) {
              if (buffer.label == action.buffer) {
                action.buffer = buffer;
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
          const bufferLen = this.buffers.push(allocatedBuffer);
          /** and update this primitive's buffer description object;
           * this will trigger a rebuild of bindGroup and pipeline layouts */
          this.bufferDescription = {
            ...this.bufferDescription,
            [bufferLen - 1]: action.bufferType ?? "storage",
          };
          action.bufferType ?? "storage";
          break;
      }
    }

    /* TODO: Is this the right way to do timing? */
    const commandBuffer = encoder.finish();
    if (args?.enableCPUTiming) {
      await this.device.queue.onSubmittedWorkDone();
      this.cpuStartTime = performance.now();
    }
    this.device.queue.submit([commandBuffer]);
    if (args?.enableCPUTiming) {
      await this.device.queue.onSubmittedWorkDone();
      this.cpuEndTime = performance.now();
    }
  }
  async getResult() {
    return {
      gpuTotalTimeNS: this.#timingHelper.getResult(),
      cpuTotalTimeNS: (this.cpuEndTime - this.cpuStartTime) * 1000000.0,
    };
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
