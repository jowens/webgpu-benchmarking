export class BasePrimitive {
  constructor(args) {
    console.log(args);
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
          `Primitive ${this.constructor.name} requires a "${field}" argument.`
        );
      }
      this[field] = args[field];
    }

    /* arguments that could be objects or arrays */
    for (field of ["uniforms", "inputs", "outputs"]) {
      this[`#{field}`] = undefined;
      if (field in args) {
        this[field] = args[field]; // should use setter
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
