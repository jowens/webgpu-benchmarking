import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
// util functions in util.js, imported by HTML wrapper

const adapter = await navigator.gpu?.requestAdapter();
const hasSubgroups = adapter.features.has("subgroups");
const canTimestamp = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredFeatures: [
    ...(canTimestamp ? ["timestamp-query"] : []),
    ...(hasSubgroups ? ["subgroups"] : []),
  ],
});

if (!device) {
  fail("Fatal error: Device does not support WebGPU.");
}

const range = (min, max) =>
  [...Array(max - min + 1).keys()].map((i) => i + min);

const membwTest = {
  name: "membw",
  type: "memory",
  description:
    "Copies input array to output array. One thread is assigned per 32b input element.",
  parameters: {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 25).map((i) => 2 ** i),
  },
  trials: 10,
  kernel: (workgroupSize) => /* wgsl */ `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${workgroupSize}) fn memcpyKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        /* needs to be a grid-stride loop */
        let i = id.y * nwg.x * ${workgroupSize} + id.x;
        memDest[i] = memSrc[i] + 1.0;
    }`,
  validate: (input, output) => {
    return input + 1.0 == output;
  },
  bytesTransferred: (memInput, memOutput) => {
    return memInput.byteLength + memOutput.byteLength;
  },
  plots: [
    {
      x: { x: "memsrcSize", label: "Copied array size (B)" },
      y: { y: "bandwidth", label: "Achieved bandwidth (GB/s)" },
      stroke: { stroke: "workgroupSize" },
    },
  ],
};

const maddTest = {
  name: "madd",
  type: "compute",
  description:
    "Computes N multiply-adds per input element. One thread is responsible for one 32b input element.",
  parameters: {
    workgroupSize: range(0, 7).map((i) => 2 ** i),
    memsrcSize: range(10, 26).map((i) => 2 ** i),
  },
  trials: 10,
  kernel: (workgroupSize, opsPerKernel) => {
    /* wsgl */ var k = `
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${workgroupSize}) fn maddKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i = id.y * nwg.x * ${workgroupSize} + id.x;
        if (i < arrayLength(&memSrc)) {
        var f = memSrc[i];
        /* 2^-22 = 2.38418579e-7 */
        var b = f * 2.38418579e-7 + 1.0;
        /* b is a float btwn 1 and 2 */`;
    while (opsPerKernel > 0) {
      k = k + "    f = f * b + b;\n";
      opsPerKernel -= 2;
    }
    k = k + "    memDest[i] = f;\n}\n}";
    return k;
  },
  validate: (input, output) => {
    var f = input;
    const b = f * 2.38418579e-7 + 1.0;
    /* b is a float btwn 1 and 2 */
    f = f * b + b;
    f = f * b + b;
    f = f * b + b;
    f = f * b + b;
    f = f * b + b;
    f = f * b + b;
    f = f * b + b;
    // allow for a bit of FP error
    return Math.abs(f - output) / f < 0.00001;
  },
  bytesTransferred: (memInput, memOutput) => {
    return memInput.byteLength + memOutput.byteLength;
  },
  threadCount: (memInput) => {
    return memInput.byteLength / 4;
  },
  flopsPerThread: () => {
    return 256;
  },
  gflops: (threads, flopsPerThread, time) => {
    return (threads * flopsPerThread) / time;
  },
  plots: [
    {
      x: { x: "threadCount", label: "Active threads" },
      y: { y: "gflops", label: "GFLOPS" },
      stroke: { stroke: "workgroupSize" },
    },
  ],
};

const reducePerWGTest = {
  name: "reduce per wg",
  workgroupSizes: range(2, 7).map((i) => 2 ** i),
  memsrcSizes: range(16, 17).map((i) => 2 ** i),
  trials: 10,
  kernel: (workgroupSize) => /* wsgl */ `
    enable subgroups;
    // var<workgroup> sum: f32; // zero initialized?
    /* output */
    @group(0) @binding(0) var<storage, read_write> memDest: array<f32>;
    /* input */
    @group(0) @binding(1) var<storage, read> memSrc: array<f32>;

    @compute @workgroup_size(${workgroupSize}) fn reducePerWGKernel(
      @builtin(global_invocation_id) id: vec3u,
      @builtin(num_workgroups) nwg: vec3u,
      @builtin(workgroup_id) wgid: vec3u) {
        let i = id.y * nwg.x * ${workgroupSize} + id.x;
        let sa = subgroupAdd(memSrc[i]);
        memDest[i] = sa;
    }
  `,
  validate: (f) => {
    return f;
  },
};

// TODO
// strided reads
// random reads

const tests = [membwTest, maddTest];

for (const test of tests) {
  const data = [];
  for (const param of combinations(test.parameters)) {
    const memsrcSize = param.memsrcSize;
    const workgroupSize = param.workgroupSize;
    const timingHelper = new TimingHelper(device);

    const workgroupCount = memsrcSize / workgroupSize;
    const dispatchGeometry = [workgroupCount, 1];
    while (
      dispatchGeometry[0] > adapter.limits.maxComputeWorkgroupsPerDimension
    ) {
      dispatchGeometry[0] /= 2;
      dispatchGeometry[1] *= 2;
    }
    console.log(`workgroup count: ${workgroupCount}
      workgroup size: ${workgroupSize}
      maxComputeWGPerDim: ${adapter.limits.maxComputeWorkgroupsPerDimension}
      dispatchGeometry: ${dispatchGeometry}`);

    const memsrc = new Float32Array(memsrcSize);
    for (let i = 0; i < memsrc.length; i++) {
      memsrc[i] = i & (2 ** 22 - 1); // roughly, range of 32b significand
    }

    const memcpyModule = device.createShaderModule({
      label: `module: ${test.name}`,
      code: test.kernel(workgroupSize, 256),
    });

    const kernelPipeline = device.createComputePipeline({
      label: "memcpy compute pipeline",
      layout: "auto",
      compute: {
        module: memcpyModule,
      },
    });

    // create buffers on the GPU to hold data
    // read-only inputs:
    const memsrcBuffer = device.createBuffer({
      label: "memory source buffer",
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

    const memdestBuffer = device.createBuffer({
      label: "memory destination buffer",
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const mappableMemdstBuffer = device.createBuffer({
      label: "mappable memory destination buffer",
      size: memsrc.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
    const kernelBindGroup = device.createBindGroup({
      label: "bindGroup for memcpy kernel",
      layout: kernelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: memdestBuffer } },
        { binding: 1, resource: { buffer: memsrcBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({
      label: "kernel encoder",
    });
    /* run a bunch of kernels before we start timing, don't time overhead */
    const kernelPrepass = encoder.beginComputePass(encoder, {
      label: "untimed kernel compute pass",
    });
    kernelPrepass.setPipeline(kernelPipeline);
    kernelPrepass.setBindGroup(0, kernelBindGroup);
    for (var i = 0; i < test.trials; i++) {
      kernelPrepass.dispatchWorkgroups(...dispatchGeometry);
    }
    kernelPrepass.end();

    const kernelPass = timingHelper.beginComputePass(encoder, {
      label: "timed kernel compute pass",
    });
    kernelPass.setPipeline(kernelPipeline);
    kernelPass.setBindGroup(0, kernelBindGroup);
    // TODO handle not evenly divisible by wgSize
    for (var i = 0; i < test.trials; i++) {
      kernelPass.dispatchWorkgroups(...dispatchGeometry);
    }
    kernelPass.end();

    // Encode a command to copy the results to a mappable buffer.
    // this is (from, to)
    encoder.copyBufferToBuffer(
      memdestBuffer,
      0,
      mappableMemdstBuffer,
      0,
      mappableMemdstBuffer.size
    );

    // Finish encoding and submit the commands
    const command_buffer = encoder.finish();
    device.queue.submit([command_buffer]);

    // Read the results
    await mappableMemdstBuffer.mapAsync(GPUMapMode.READ);
    const memdest = new Float32Array(
      mappableMemdstBuffer.getMappedRange().slice()
    );
    mappableMemdstBuffer.unmap();
    let errors = 0;
    for (let i = 0; i < memdest.length; i++) {
      if (!test.validate(memsrc[i], memdest[i])) {
        if (errors < 5) {
          console.log(
            `Error ${errors}: i=${i}, input=${memsrc[i]}, output=${
              memdest[i]
            }, expected=${test.validate(memsrc[i])}`
          );
        }
        errors++;
      }
    }
    if (errors > 0) {
      console.log(`Memdest size: ${memdest.length} | Errors: ${errors}`);
    } else {
      console.log(`Memdest size: ${memdest.length} | No errors!`);
    }

    timingHelper.getResult().then((ns) => {
      const result = {
        name: test.name,
        time: ns / test.trials,
        workgroupSize: workgroupSize,
        memsrcSize: memsrcSize,
      };
      if (test.bytesTransferred) {
        result.bytesTransferred = test.bytesTransferred(memsrc, memdest);
        result.bandwidth = result.bytesTransferred / result.time;
      }
      if (test.threadCount) {
        result.threadCount = test.threadCount(memsrc);
      }
      if (test.flopsPerThread) {
        result.flopsPerThread = test.flopsPerThread();
      }
      if (test.gflops && result.threadCount && result.flopsPerThread) {
        result.gflops = test.gflops(
          result.threadCount,
          result.flopsPerThread,
          result.time
        );
      }
      data.push(result);
      console.log(result);
    });
  }
  console.log(data);

  for (const testPlot of test.plots) {
    const plot = Plot.plot({
      marks: [
        Plot.lineY(data, {
          x: testPlot.x.x,
          y: testPlot.y.y,
          stroke: testPlot.stroke.stroke,
          tip: true,
        }),
        Plot.text(
          data,
          Plot.selectLast({
            x: testPlot.x.x,
            y: testPlot.y.y,
            z: testPlot.stroke.stroke,
            text: testPlot.stroke.stroke,
            textAnchor: "start",
            dx: 3,
          })
        ),
      ],
      x: { type: "log", label: testPlot?.x?.label ?? "XLABEL" },
      y: { type: "log", label: testPlot?.y?.label ?? "YLABEL" },
      color: { type: "ordinal", legend: true },
    });
    const div = document.querySelector("#plot");
    div.append(plot);
  }
}
