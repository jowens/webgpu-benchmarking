import {
  BinOpAddU32,
  BinOpMinU32,
  BinOpMaxU32,
  BinOpAddF32,
  BinOpMinF32,
  BinOpMaxF32,
} from "./binop.mjs";
import { datatypeToTypedArray } from "./util.mjs";

if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
} else {
  // running in Chrome
  // eslint-disable-next-line no-unused-vars
  const urlParams = new URL(window.location.href).searchParams;
}

// import primitive only, no test suite
import { NoAtomicPKReduce } from "./reduce.mjs";
import { WGScan, HierarchicalScan } from "./scan.mjs";

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
    console.error("Fatal error: Device does not support WebGPU.");
  }
  const memsrcLength = 2 ** 20; // items, not bytes
  const datatype = "f32";
  const memsrcX32 = new (datatypeToTypedArray(datatype))(memsrcLength);
  for (let i = 0; i < memsrcLength; i++) {
    switch (datatype) {
      case "u32":
        memsrcX32[i] = i == 0 ? 11 : memsrcX32[i - 1] + 1; // trying to get u32s
        break;
      case "f32":
        memsrcX32[i] = i + 42;
        break;
    }
  }

  // eslint-disable-next-line no-unused-vars
  const reducePrimitive = new NoAtomicPKReduce({
    device,
    binop: BinOpAddF32,
    datatype: datatype,
    gputimestamps: true, //// TODO should work without this
    // inputBuffer and outputBuffer are Reduce-specific names
    // inputBuffer: { buffer: memsrcBuffer, offset: 0 },
    // inputBuffer: memsrcBuffer,
    // outputBuffer: memdestBuffer,
  });

  // eslint-disable-next-line no-unused-vars
  const iscanPrimitive = new HierarchicalScan({
    device,
    binop: BinOpAddF32,
    type: "inclusive",
    datatype: datatype,
    gputimestamps: true, //// TODO should work without this
  });

  // eslint-disable-next-line no-unused-vars
  const escanPrimitive = new HierarchicalScan({
    device,
    binop: BinOpAddF32,
    type: "exclusive",
    datatype: datatype,
    gputimestamps: true, //// TODO should work without this
  });

  // const primitive = iscanPrimitive;
  // const primitive = escanPrimitive;
  const primitive = reducePrimitive;

  let memdestBytes;
  switch (primitive.constructor.name) {
    case "NoAtomicPKReduce":
      memdestBytes = 4;
      break;
    default:
      memdestBytes = memsrcX32.byteLength;
      break;
  }

  // allocate/create buffers on the GPU to hold in/out data
  const memsrcBuffer = device.createBuffer({
    label: `memory source buffer (${datatype})`,
    size: memsrcX32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(memsrcBuffer, 0, memsrcX32);

  const memdestBuffer = device.createBuffer({
    label: "memory destination buffer",
    size: memdestBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST /* COPY_DST necessary for initialization */,
  });

  const mappableMemdestBuffer = device.createBuffer({
    label: "mappable memory destination buffer",
    size: memdestBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  await primitive.execute({
    inputBuffer: memsrcBuffer,
    outputBuffer: memdestBuffer,
  });

  // copy output back to host
  const encoder = device.createCommandEncoder({
    label: "copy result CPU->GPU encoder",
  });
  encoder.copyBufferToBuffer(
    memdestBuffer,
    0,
    mappableMemdestBuffer,
    0,
    mappableMemdestBuffer.size
  );
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // Read the results
  await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
  const memdest = new (datatypeToTypedArray(datatype))(
    mappableMemdestBuffer.getMappedRange().slice()
  );
  mappableMemdestBuffer.unmap();

  if (primitive.validate) {
    const errorstr = primitive.validate({
      inputBuffer: memsrcX32,
      outputBuffer: memdest,
    });
    if (errorstr == "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed: ${errorstr}`);
    }
  }
  // currently no timing computation, that's fine
}
