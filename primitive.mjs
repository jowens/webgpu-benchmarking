import { Buffer } from "./buffer.mjs";
import { TestInputBuffer, TestOutputBuffer } from "./testbuffer.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";

export class BasePrimitive {
  static pipelineLayoutsCache = new Map();
  static __timingHelper; // initialized to undefined

  constructor(args) {
    // expect that args are:
    // { device: device,   // REQUIRED
    //   label: label,
    //   tuning: { param1: val1, param2: val2 }, // TUNABLE params
    //   someConfigurationSetting: thatSetting,
    //   uniforms: uniformbuffer0,
    //   inputBuffer0: inputbuffer0,
    //   inputBuffer1: inputbuffer1,
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
        case "device":
          /* do nothing, handled above */
          break;
        default:
          /* copy it into the primitive */
          this[field] = value;
          break;
      }
    }

    this.__buffers = {}; // this is essentially private
  }

  /** registerBuffer associates a name with a buffer and
   *    stores the buffer in the list of buffers
   * Modes of operation:
   * - registerBuffer(String name) means
   *   "associate the buffer this[name] with name"
   *   Meant to register named buffers (known to the primitive)
   * - registerBuffer(Buffer b) means
   *   "associate the buffer b with b.label"
   * - registerBuffer(Object o) means
   *   "associate a new Buffer(o) with o.label"
   *
   */
  registerBuffer(bufferObj) {
    switch (typeof bufferObj) {
      case "string":
        this.__buffers[bufferObj] = new Buffer({
          label: bufferObj,
          buffer: this[bufferObj],
        });
        break;
      default:
        switch (bufferObj.constructor.name) {
          case "Buffer":
            /* already created the buffer, don't remake it */
            this.__buffers[bufferObj.label] = bufferObj;
            break;
          default:
            this.__buffers[bufferObj.label] = new Buffer(bufferObj);
            break;
        }
    }
  }

  getBuffer(label) {
    return this.__buffers[label];
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
        break;
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

    /* do we need to register any new buffers specified in execute? */
    for (const knownBuffer of this.knownBuffers) {
      if (knownBuffer in args) {
        this.registerBuffer({
          label: knownBuffer,
          buffer: args[knownBuffer],
        });
      }
    }

    /* begin timestamp prep - count kernels, allocate 2 timestamps/kernel */
    let kernelCount = 0; // how many kernels are there total?
    if (args.enableGPUTiming) {
      for (const action of this.compute()) {
        if (action.constructor == Kernel && action.enabled()) {
          kernelCount++;
        }
      }
      if (BasePrimitive.__timingHelper) {
        if (BasePrimitive.__timingHelper.numKernels === kernelCount) {
          /* do nothing - we can reuse __timingHelper without change */
        } else {
          /* keep the same instance, but reset it */
          /* this ensures timing buffers are destroyed */
          BasePrimitive.__timingHelper.destroy();
          BasePrimitive.__timingHelper.reset(kernelCount);
        }
      } else {
        /* there's no timing helper */
        BasePrimitive.__timingHelper = new TimingHelper(
          this.device,
          kernelCount
        );
      }
    }
    const encoder = this.device.createCommandEncoder({
      label: `${this.label} primitive encoder`,
    });
    for (const action of this.compute()) {
      switch (action.constructor) {
        case Kernel: {
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
          }
          if (action.debugPrintKernel) {
            console.log(kernelString);
          }

          const computeModule = this.device.createShaderModule({
            label: `module: ${this.label}`,
            code: kernelString,
          });

          // build up bindGroupLayouts and pipelineLayout
          let pipelineLayout;
          if (action.bufferTypes in BasePrimitive.pipelineLayoutsCache) {
            /* cached, use it */
            pipelineLayout =
              BasePrimitive.pipelineLayoutsCache[action.bufferTypes];
          } else {
            /* first build up bindGroupLayouts, then create a pipeline layout */
            const bindGroupLayouts = [];
            for (const bufferTypesGroup of action.bufferTypes) {
              /* could also cache bind groups */
              const entries = [];
              bufferTypesGroup.forEach((element, index) => {
                if (element !== "") {
                  // not sure if this is right comparison for an empty elt?
                  entries.push({
                    binding: index,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: element },
                  });
                }
              });
              const bindGroupLayout = this.device.createBindGroupLayout({
                entries,
              });
              bindGroupLayouts.push(bindGroupLayout);
            }
            pipelineLayout = this.device.createPipelineLayout({
              bindGroupLayouts,
            });
            /* and cache it */
            BasePrimitive.pipelineLayoutsCache[action.bufferTypes] =
              pipelineLayout;
          }
          pipelineLayout.label = `${this.label} compute pipeline`;

          const kernelPipeline = this.device.createComputePipeline({
            label: `${this.label} compute pipeline`,
            layout: pipelineLayout,
            compute: {
              module: computeModule,
            },
          });

          if (action.bindings.size > 1) {
            console.error(
              "Primitive::execute::Kernel currently only supports one bind group",
              action.bindings
            );
          }

          const kernelBindGroup = this.device.createBindGroup({
            label: `bindGroup for ${this.label} kernel`,
            layout: kernelPipeline.getBindGroupLayout(0),
            /* the [0] below is because we only support 1 bind group */
            entries: action.bindings[0].map((element, index) => ({
              binding: index,
              resource: this.__buffers[element].buffer,
            })),
          });

          const kernelDescriptor = {
            label: `${this.label} compute pass`,
          };

          const kernelPass = args.enableGPUTiming
            ? BasePrimitive.__timingHelper.beginComputePass(
                encoder,
                kernelDescriptor
              )
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
          for (let trial = 0; trial < (args.trials ?? 1); trial++) {
            kernelPass.dispatchWorkgroups(...dispatchGeometry);
          }

          console.info(`${this.label} | ${action.label}
size of bindings: ${action.bindings[0].map(
            (e) => this.__buffers[e].buffer.buffer.size
          )}
workgroupCount: ${this.workgroupCount}
workgroupSize: ${this.workgroupSize}
maxGSLWorkgroupCount: ${this.maxGSLWorkgroupCount}
dispatchGeometry: ${dispatchGeometry}`);
          kernelPass.end();
          break;
        }
        case InitializeMemoryBlock: {
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
        }
        case AllocateBuffer: {
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
          this.registerBuffer({ label: action.label, buffer: allocatedBuffer });
          break;
        }
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
  async getTimingResult() {
    const gpuTotalTimeNS = await BasePrimitive.__timingHelper.getResult();
    const cpuTotalTimeNS = (this.cpuEndTime - this.cpuStartTime) * 1000000.0;
    return { gpuTotalTimeNS, cpuTotalTimeNS };
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
