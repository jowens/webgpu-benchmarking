import { TimingHelper } from "./webgpufundamentals-timing.mjs";

if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
} else {
  // running in Chrome
  const urlParams = new URL(window.location.href).searchParams;
}

// import primitive only, no test suite
import { AtomicGlobalU32Reduce } from "./reduce.mjs";

export async function main(navigator) {
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
  const memsrcSize = 2 ** 20;
  const memsrcu32 = new Uint32Array(memsrcSize);
  for (let i = 0; i < memsrcSize; i++) {
    memsrcu32[i] = i == 0 ? 0 : memsrcu32[i - 1] + 1; // trying to get u32s
  }
  const memdestBytes = 4;

  const primitive = new AtomicGlobalU32Reduce({
    workgroupSize: 128,
    workgroupCount: memsrcSize / 128,
  });

  const computeModule = device.createShaderModule({
    label: `module: ${primitive.constructor.name}`,
    code: primitive.kernel(),
  });

  const kernelPipeline = device.createComputePipeline({
    label: `${primitive.constructor.name} compute pipeline`,
    layout: "auto",
    compute: {
      module: computeModule,
    },
  });

  // allocate/create buffers on the GPU to hold in/out data
  const memsrcuBuffer = device.createBuffer({
    label: "memory source buffer (uint)",
    size: memsrcu32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(memsrcuBuffer, 0, memsrcu32);

  const memdestBuffer = device.createBuffer({
    label: "memory destination buffer",
    size: memdestBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const mappableMemdestBuffer = device.createBuffer({
    label: "mappable memory destination buffer",
    size: memdestBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
  const kernelBindGroup = device.createBindGroup({
    label: `bindGroup for ${primitive.constructor.name} kernel`,
    layout: kernelPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: memdestBuffer } },
      {
        binding: 1,
        resource: {
          buffer: memsrcuBuffer,
        },
      },
    ],
  });

  const timingHelper = new TimingHelper(device);
  const encoder = device.createCommandEncoder({
    label: "timed kernel run encoder",
  });
  const kernelPass = timingHelper.beginComputePass(encoder, {
    label: "timed kernel compute pass",
  });
  kernelPass.setPipeline(kernelPipeline);
  kernelPass.setBindGroup(0, kernelBindGroup);
  kernelPass.dispatchWorkgroups(primitive.workgroupCount);
  kernelPass.end();
  encoder.copyBufferToBuffer(
    memdestBuffer,
    0,
    mappableMemdestBuffer,
    0,
    mappableMemdestBuffer.size
  );

  // Finish encoding and submit the commands
  const commandBuffer = encoder.finish();
  await device.queue.onSubmittedWorkDone();
  const passStartTime = performance.now();
  device.queue.submit([commandBuffer]);
  await device.queue.onSubmittedWorkDone();
  const passEndTime = performance.now();

  const resolveEncoder = device.createCommandEncoder({
    label: "timestamp resolve encoder",
  });
  timingHelper.resolveTiming(resolveEncoder);
  const resolveCommandBuffer = resolveEncoder.finish();
  await device.queue.onSubmittedWorkDone();
  device.queue.submit([resolveCommandBuffer]);

  // Read the results
  await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
  const memdest = new Uint32Array(
    mappableMemdestBuffer.getMappedRange().slice()
  );
  mappableMemdestBuffer.unmap();

  console.info(`${computeModule.label}
workgroupCount: ${primitive.workgroupCount}
workgroup size: ${primitive.workgroupSize}`);
  if (primitive.validate) {
    const errorstr = primitive.validate(memsrcu32, memdest);
    if (errorstr == "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed: ${errorstr}`);
    }
  }
  console.debug(`memdest: ${memdest}`);

  timingHelper.getResult().then((ns) => {
    console.log(ns, "ns");
  });
}
