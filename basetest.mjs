export class BasePrimitive {
  constructor(args) {
    // expect that args are:
    // { device: device,
    //   params: { param1: val1, param2: val2 },
    //   uniforms: uniformbuffer0,
    //   inputs: [inputbuffer0, inputbuffer1],
    //   outputs: outputbuffer0,
    // }
    if (this.constructor === BasePrimitive) {
      throw new Error(
        "Cannot instantiate abstract class BasePrimitive directly."
      );
    }
    Object.assign(this, args.params);
    // could do some defaults here

    let field;

    /* required arguments */
    for (field of ["device"]) {
      if (!(field in args)) {
        throw new Error(
          `Primitive ${this.constructor} requires a "${field}" argument.`
        );
      }
      this[field] = args[field];
    }

    /* arguments that could be objects or arrays */
    for (field of ["uniforms", "inputs", "outputs"]) {
      if (field in args) {
        const arr = Array.isArray(args[field]) ? args[field] : [args[field]];
        this[field] = arr;
      }
    }
  }
  kernel() {
    /* call this from a subclass instead */
    throw new Error("Cannot call kernel() from abstract class BasePrimitive.");
  }
  getDispatch() {
    /* call this from a subclass instead */
    throw new Error(
      "Cannot call getDispatch() from abstract class BasePrimitive."
    );
  }
  async execute() {
    const encoder = this.device.createCommandEncoder({
      label: `${this.constructor.name} primitive encoder`,
    });

    for (const action of this.compute) {
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
            if (buffer in this) {
              for (const binding of this[buffer]) {
                bindings.push({
                  binding: bindingIdx++,
                  resource: { buffer: binding },
                });
              }
            }
          }

          const kernelBindGroup = this.device.createBindGroup({
            label: `bindGroup for ${this.constructor.name} kernel`,
            layout: kernelPipeline.getBindGroupLayout(0),
            entries: bindings,
          });

          const kernelPass = encoder.beginComputePass(encoder, {
            label: `${this.constructor.name} compute pass`,
          });

          kernelPass.setPipeline(kernelPipeline);
          kernelPass.setBindGroup(0, kernelBindGroup);
          kernelPass.dispatchWorkgroups(...this.getDispatch());
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
            action.buffer,
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
}

export class BaseTestSuite {
  constructor(config) {
    Object.assign(this, config);
    this.config = config; // just to keep this encapsulated
    // depending on what's in config, instantiate:
    // primitive, processResults, plots, summarize
    //   each are a class
    // params
    //   is an object
    if ("plots" in config) {
      /** this complexity is to make sure that a plot
       * that uses a template literal has it evaluated
       * in the context of this test suite. By default,
       * it's lexically scoped, which is useless. call()
       * makes sure it's evaluated in this class's context.
       */
      function f(plot, env) {
        if (typeof plot === "function") {
          return plot.call(env);
        } else {
          // presumably a plain object
          return plot;
        }
      }
      // TODO: i do not understand why 'map' does not work
      this.processedPlots = [];
      for (let plot of this.plots) {
        this.processedPlots.push(f(plot, this));
      }
    }
  }
  getTest(params) {
    // factory
    const prim = new this.primitive(params);
    /* copy all string fields from TestSuite -> Test */
    for (const key of Object.keys(this)) {
      if (typeof this[key] == "string") {
        prim[key] = this[key];
      }
    }
    return prim;
  }
  getPlots() {
    return this.processedPlots;
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
