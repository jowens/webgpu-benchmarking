import { Buffer } from "./buffer.mjs";
import { TestInputBuffer, TestOutputBuffer } from "./testbuffer.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";
import {
  wgslFunctions,
  wgslFunctionsWithoutSubgroupSupport,
} from "./wgslFunctions.mjs";
import { CountingMap, NonCachingMap } from "./mapvariants.mjs";
import { formatWGSL } from "./util.mjs";

class WebGPUObjectCache {
  constructor({
    enabled = [
      "pipelineLayouts",
      "bindGroupLayouts",
      "computeModules",
      "computePipelines",
    ] /* this list is the default */,
    // eslint-disable-next-line no-unused-vars
    ...args
  } = {}) {
    /** each object type has its own cache for efficiency, and for the ability
     * to selectively enable/disable that cache */
    this.pipelineLayouts = enabled.includes("pipelineLayouts")
      ? new CountingMap()
      : new NonCachingMap();
    this.bindGroupLayouts = enabled.includes("bindGroupLayouts")
      ? new CountingMap()
      : new NonCachingMap();
    this.computeModules = enabled.includes("computeModules")
      ? new CountingMap()
      : new NonCachingMap();
    this.computePipelines = enabled.includes("computePipelines")
      ? new CountingMap()
      : new NonCachingMap();
  }
  get stats() {
    return `Cache hits/misses:
Pipeline layouts: ${this.pipelineLayouts.hits}/${this.pipelineLayouts.misses}
Bind group layouts: ${this.bindGroupLayouts.hits}/${this.bindGroupLayouts.misses}
Compute modules: ${this.computeModules.hits}/${this.computeModules.misses}
Compute pipelines: ${this.computePipelines.hits}/${this.computePipelines.misses}`;
  }
}

function generateCacheKey(obj) {
  // Handle non-object types (primitives, null, undefined) directly.
  if (typeof obj !== "object" || obj === null) {
    return String(obj);
  }

  return JSON.stringify(obj, (key, value) => {
    // 1. Always completely ignore labels for caching purposes
    if (key === "label") {
      return undefined;
    }

    // 2. Check for pre-computed '__cacheKey' on nested objects
    // This check applies to the 'value' being processed,
    // which would be a nested object.
    if (
      typeof value === "object" &&
      value !== null &&
      Object.hasOwn(value, "__cacheKey") && // ESLint-friendly check
      typeof value.__cacheKey === "string" // Ensure it's a string, as expected for a key
    ) {
      // If a valid cacheKey is found on a nested object,
      // use that string as its representation in the parent's key.
      // We prepend a unique identifier (e.g., "__ref_") to avoid collisions
      // with actual string values that might coincidentally look like a cache key.
      return `__cached_ref_:${value.__cacheKey}`;
    }

    // For all other keys/values, return the original value.
    // JSON.stringify will then process this value as usual (e.g., ignoring functions/symbols).
    return value;
  });
}

export class BasePrimitive {
  // these static objects are shared by all Primitives
  static __deviceToWebGPUObjectCache = new WeakMap();
  // this is a WeakMap<device, CacheableWebGPUObjects>
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
    //   disableSubgroups: true [default: false],
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

    if (!BasePrimitive.__deviceToWebGPUObjectCache.has(this.device)) {
      /* never seen this device before */
      BasePrimitive.__deviceToWebGPUObjectCache.set(
        this.device,
        /** if we want to enable a certain set of caches,
         *  make the constructor argument to WebGPUObjectCache
         * { enabled: ["pipelineLayouts"] }
         */
        new WebGPUObjectCache()
      );
    }

    /** possible that we've specified binop but not datatype, in
     * which case set datatype from binop */
    if (!("datatype" in args) && "binop" in args) {
      args.datatype = args.binop.datatype;
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

    // not sure a Map gives me anything I don't get from {}
    this.__buffers = {}; // this is essentially private
    this.useSubgroups = this.device.features.has("subgroups");
    if (args.disableSubgroups) {
      this.useSubgroups = false;
    }
    if (this.useSubgroups) {
      this.fnDeclarations = new wgslFunctions(this);
      this.SUBGROUP_MIN_SIZE = this.device.adapterInfo.subgroupMinSize;
      this.SUBGROUP_MAX_SIZE = this.device.adapterInfo.subgroupMaxSize;
    } else {
      this.fnDeclarations = new wgslFunctionsWithoutSubgroupSupport(this);
    }
  }

  makeParamString(params) {
    let str = " [";
    for (const param of params) {
      str += ` ${param}: ${this[param]}`;
    }
    str += " ]";
    return str;
  }

  static cacheStats(device) {
    return BasePrimitive.__deviceToWebGPUObjectCache.get(device).stats;
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
   * Additionally: If the resulting buffer has no datatype,
   *   set it to the datatype of the primitive
   * Rationale: We can pass a raw GPUBuffer that only knows
   *   its byte length as an input buffer, but we need to
   *   compute kernel parameters (e.g., number of workgroups)
   *   based the number of inputs in that buffer, so we need
   *   to know its datatype
   *
   * It may be useful at some point to have this function
   * return the buffer it just registered.
   */
  registerBuffer(bufferObj) {
    switch (typeof bufferObj) {
      case "string":
        this.__buffers[bufferObj] = new Buffer({
          label: bufferObj,
          buffer: this[bufferObj],
          // this probably needs datatype: but I won't add that
          // until I know it's useful
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
            if (
              this.__buffers[bufferObj.label]?.datatype === undefined &&
              this?.datatype
            ) {
              /* assign a datatype from primitive if buffer doesn't have one */
              this.__buffers[bufferObj.label].datatype = this.datatype;
            }
            break;
        }
        break;
    }
  }

  getBuffer(label) {
    return this.__buffers[label];
  }

  getBufferResource(args) {
    /** Converts a name of a buffer to a buffer
     * Used for the "resource" argument of createBindGroup()
     * Works in the context of a GPUBufferBinding, which is either
     * - buffer_name => buffer
     * - { buffer: buffer_name, offset: o, size: s } => { buffer: buffer, offset: o, size: s }
     */
    switch (typeof args) {
      case "string":
        return this.getBuffer(args).buffer;
      default:
        if (args.buffer) {
          /** this will probably fail if args specify offset or size AND the
           * buffer in getBuffer ALSO has offset or size, but we haven't seen
           * a case like that yet and figure out the right behavior when we do */
          if (typeof args.buffer !== "string") {
            throw new Error(
              "Primitive::getBufferResource: 'buffer' argument must be a string naming a buffer"
            );
          }
          /** this.getBuffer(args.buffer) returns a Buffer object
           * this.getBuffer(args.buffer).buffer returns a GPUBufferBinding
           *   that object certainly has a buffer member, but also possibly has offset and size
           * Current behavior is to ignore the GPUBufferBinding's offset/size, and use
           *   offset/size if specified in args, but we have no use case for what the correct
           *   behavior should be. The return below is where it should change if we want
           *   different behavior.
           */
          return {
            ...args,
            buffer: this.getBuffer(args.buffer).buffer.buffer,
          };
        } else {
          throw new Error(
            "Primitive::getBufferResource: argument",
            args,
            "must be either a string that names a known buffer or a GPUBuffer object with a buffer member that is a string that names a known buffer"
          );
        }
    }
  }

  getDispatchGeometry() {
    /* call this from a subclass instead */
    throw new Error(
      "Cannot call getDispatchGeometry() from abstract class BasePrimitive."
    );
  }
  getSimpleDispatchGeometry(args = {}) {
    const workgroupCount = args.workgroupCount ?? this.workgroupCount;
    // todo this is too simple
    if (workgroupCount === undefined) {
      throw new Error(
        "Primitive:getSimpleDispatchGeometry(): Must specify either a workgroupCount in args or this.workgroupCount."
      );
    }
    let dispatchGeometry = [workgroupCount, 1];
    while (
      dispatchGeometry[0] > this.device.limits.maxComputeWorkgroupsPerDimension
    ) {
      /* too big */
      dispatchGeometry[0] = Math.ceil(dispatchGeometry[0] / 2);
      dispatchGeometry[1] *= 2;
    }
    return dispatchGeometry;
  }
  get bytesTransferred() {
    /* call this from a subclass instead */
    throw new Error(
      "Cannot call bytesTransferred() from abstract class BasePrimitive."
    );
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
  async execute(args = {}) {
    /** loop through each of the actions listed in this.compute(),
     * instantiating whatever WebGPU constructs are necessary to run them
     *
     * TODO: memoize these constructs
     */

    if (args.encoder && args.enableCPUTiming) {
      console.warn(
        "Primitive::execute: cannot pass in an encoder AND\nenable CPU timing, CPU timing will be disabled"
      );
    }

    /* set up defaults */
    if (!("trials" in args)) {
      args.trials = 1;
    }

    /* do we need to register any new buffers specified in execute? */
    if (this.knownBuffers) {
      for (const knownBuffer of this.knownBuffers) {
        if (knownBuffer in args) {
          this.registerBuffer({
            label: knownBuffer,
            buffer: args[knownBuffer],
          });
        }
      }
    } else {
      console.warn(
        "Primitive::execute: This primitive has no knownBuffers, please specify\nthem in the primitive class constructor."
      );
    }

    /* begin timestamp prep - count kernels, allocate 2 timestamps/kernel */
    let kernelCount = 0; // how many kernels are there total?
    if (args.enableGPUTiming) {
      for (const action of this.compute()) {
        if (action.constructor === Kernel && action.enabled()) {
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
    /* if we passed in an encoder, use that */
    const encoder =
      args.encoder ??
      this.device.createCommandEncoder({
        label: `${this.label} primitive encoder`,
      });
    if (!BasePrimitive.__deviceToWebGPUObjectCache.has(this.device)) {
      /* should never see this */
      console.error("Device not found in WebGPU object cache", this.device);
    }
    const webGPUObjectCache =
      /* this was created in constructor */
      BasePrimitive.__deviceToWebGPUObjectCache.get(this.device);
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
          if (action?.logKernelCodeToConsole) {
            console.log(
              action.label ? `/*** ${action.label} ***/\n` : "",
              formatWGSL(kernelString)
            );
          }

          const computeModuleCacheKey = generateCacheKey(kernelString);
          let computeModule = webGPUObjectCache.computeModules.get(
            computeModuleCacheKey
          );
          if (!computeModule) {
            computeModule = this.device.createShaderModule({
              code: kernelString,
            });
            computeModule.__cacheKey = computeModuleCacheKey;
            webGPUObjectCache.computeModules.set(
              computeModuleCacheKey,
              computeModule
            );
          }
          computeModule.label = `module: ${this.label}`;

          if (action?.logCompilationInfo) {
            /* careful: probably has performance implications b/c await */
            const shaderInfo = await computeModule.getCompilationInfo();
            if (shaderInfo.messages.length > 0) {
              console.log("Warnings for", action.label, shaderInfo.messages);
            }
            for (const message of shaderInfo.messages) {
              console.log(
                message.type,
                "at line",
                message.lineNum,
                message.message
              );
            }
          }

          // build up bindGroupLayouts and pipelineLayout
          const pipelineLayoutCacheKey = generateCacheKey(action.bufferTypes);
          let pipelineLayout = webGPUObjectCache.pipelineLayouts.get(
            pipelineLayoutCacheKey
          );
          if (!pipelineLayout) {
            /* first build up bindGroupLayouts, then create a pipeline layout */
            const bindGroupLayouts = [];
            for (const [
              bufferTypesGroupIndex,
              bufferTypesGroup,
            ] of action.bufferTypes.entries()) {
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
              const bindGroupLayoutCacheKey = generateCacheKey(entries);
              let bindGroupLayout = webGPUObjectCache.bindGroupLayouts.get(
                bindGroupLayoutCacheKey
              );
              if (!bindGroupLayout) {
                /* did not find it in cache, construct it */
                bindGroupLayout = this.device.createBindGroupLayout({
                  entries,
                });
                bindGroupLayout.__cacheKey = bindGroupLayoutCacheKey;
                webGPUObjectCache.bindGroupLayouts.set(
                  bindGroupLayoutCacheKey,
                  bindGroupLayout
                );
              }
              bindGroupLayout.label = `${this.label} bindGroupLayout[${bufferTypesGroupIndex}]`;
              bindGroupLayouts.push(bindGroupLayout);
            }
            pipelineLayout = this.device.createPipelineLayout({
              bindGroupLayouts,
            });
            pipelineLayout.__cacheKey = pipelineLayoutCacheKey;
            /* and cache it */
            webGPUObjectCache.pipelineLayouts.set(
              pipelineLayoutCacheKey,
              pipelineLayout
            );
          }
          pipelineLayout.label = `${this.label} compute pipeline`;

          const computePipelineDesc = {
            layout: pipelineLayout,
            compute: {
              module: computeModule,
              ...(action.entryPoint && { entryPoint: action.entryPoint }),
              // warning: next line has never been used/tested
              ...(action.constants && { constants: action.constants }),
            },
          };
          const computePipelineCacheKey = generateCacheKey(computePipelineDesc);
          let computePipeline = webGPUObjectCache.computePipelines.get(
            computePipelineCacheKey
          );
          if (!computePipeline) {
            computePipeline =
              this.device.createComputePipeline(computePipelineDesc);
            computePipeline.__cacheKey = computePipelineCacheKey;
            /* and cache it */
            webGPUObjectCache.computePipelines.set(
              computePipelineCacheKey,
              computePipeline
            );
          }
          computePipeline.label = `${this.label} compute pipeline`;

          if (action?.bindings?.length === undefined) {
            console.error(
              "Primitive::execute::Kernel: must specify a bindings argument",
              action.bindings
            );
          }

          for (const bindingGroup of action.bindings) {
            for (const binding of bindingGroup) {
              if (
                !(binding in this.__buffers || binding.buffer in this.__buffers)
              ) {
                console.error(
                  `Primitive ${this.label} has no registered buffer ${binding}.`,
                  this
                );
              }
            }
          }

          const kernelDescriptor = {
            label: `${this.label} compute pass`,
          };

          const kernelPass = args.enableGPUTiming
            ? BasePrimitive.__timingHelper.beginComputePass(
                encoder,
                kernelDescriptor
              )
            : encoder.beginComputePass(kernelDescriptor);

          kernelPass.setPipeline(computePipeline);

          for (const [bindGroupIndex, bindGroup] of action.bindings.entries()) {
            const entries = bindGroup.map((binding, bindingIndex) => ({
              binding: bindingIndex,
              resource: this.getBufferResource(binding),
            }));
            /** Don't think it's feasible to cache bind groups because we don't have a
             * canonical way to name a buffer */
            const kernelBindGroup = this.device.createBindGroup({
              label: `bindGroup ${bindGroupIndex} for ${this.label} ${
                action.entryPoint && action.entryPoint
              } kernel`,
              layout: computePipeline.getBindGroupLayout(bindGroupIndex),
              entries,
            });
            kernelPass.setBindGroup(bindGroupIndex, kernelBindGroup);
          }

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
            /* dispatchGeometry is a vector, so spread it to make it x,y,z */
            kernelPass.dispatchWorkgroups(...dispatchGeometry);
          }

          if (action.logLaunchParameters) {
            /** "size of bindings" must be updated to handle
             * - multiple binding groups
             * - any binding that is not just a buffer name ({name, offset, size})
             */
            console.info(`${this.label} | ${action.label}
size of bindings: ${action.bindings[0].map(
              (e) => this.getBuffer(e).buffer.buffer.size
            )}
workgroupCount: ${this.workgroupCount}
workgroupSize: ${this.workgroupSize}
maxGSLWorkgroupCount: ${this.maxGSLWorkgroupCount}
dispatchGeometry: ${dispatchGeometry}`);
          }
          kernelPass.end();
          break;
        }
        case InitializeMemoryBlock: {
          /* TODO: Rewrite this as a kernel, delete get{GPU,CPU}BufferFromBinding */
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
              if (buffer.label === action.buffer) {
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
          if (!("size" in action)) {
            console.error(
              `Primitive::AllocateBuffer: Buffer ${action.label} must include a size (args are `,
              action,
              ")"
            );
          }
          if (!Number.isFinite(action.size)) {
            console.error(
              `Primitive::AllocateBuffer: Buffer ${action.label} must specify a numerical size (args are `,
              action,
              ")"
            );
          }
          /** What if this buffer is already allocated?
           * We can tell if this is the case if it's already registered.
           * This might happen if we have an intermediate buffer that's not
           *   normally visible outside the primitive, but we want to get
           *   its values.
           * So: if the buffer is already allocated/registered AND it has the
           *   same size, then skip the new allocation.
           */
          const existingBuffer = this.getBuffer(action.label);
          if (existingBuffer && existingBuffer.size === action.size) {
            /* just use the existing buffer */
          } else {
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
            this.registerBuffer({
              label: action.label,
              buffer: allocatedBuffer,
            });
          }
          /* todo: combine this with WriteGPUBuffer below */
          if (action.populateWith) {
            this.device.queue.writeBuffer(
              this.getBuffer(action.label).buffer.buffer,
              0,
              action.populateWith
            );
          }
          break;
        }
        case WriteGPUBuffer: {
          const gpuBuffer = this.getBuffer(action.label).buffer;
          this.device.queue.writeBuffer(
            gpuBuffer.buffer,
            action.offset ?? 0,
            action.cpuSource
          );
          break;
        }
      }
    }

    if (args.encoder) {
      /* user passed in an encoder, return it, don't submit it */
      return encoder;
    } else {
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
    if (typeof args === "function") {
      /* the function takes an optional arg object and returns the kernel string */
      this.kernel = args;
    } else {
      /* more complicated objects */
      /* one field should be "kernel", and it better be a function or string */
      if (
        !args.kernel ||
        (typeof args.kernel !== "function" && typeof args.kernel !== "string")
      ) {
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

export class WriteGPUBuffer {
  constructor(args) {
    Object.assign(this, args);
  }
}
