import { TimingHelper } from "./webgpufundamentals-timing.mjs";
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
  const urlParams = new URL(window.location.href).searchParams;
}

// import primitive only, no test suite
import { NoAtomicPKReduce } from "./reduce.mjs";

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
  const memsrcSize = 2 ** 20; // items, not bytes
  const datatype = "f32";
  const memsrcX32 = new (datatypeToTypedArray(datatype))(memsrcSize);
  for (let i = 0; i < memsrcSize; i++) {
    switch (datatype) {
      case "u32":
        memsrcX32[i] = i == 0 ? 11 : memsrcX32[i - 1] + 1; // trying to get u32s
        break;
      case "f32":
        memsrcX32[i] = i + 42;
        break;
    }
  }
  const memdestBytes = 4;

  // allocate/create buffers on the GPU to hold in/out data
  const memsrcBuffer = device.createBuffer({
    label: "memory source buffer (uint)",
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

  const primitive = new NoAtomicPKReduce({
    device,
    binop: BinOpMaxF32,
    datatype: datatype,
    gputimestamps: true, //// TODO should work without this
    // inputBuffer and outputBuffer are Reduce-specific names
    inputBuffer: { buffer: memsrcBuffer, offset: 0 },
    outputBuffer: memdestBuffer,
  });

  await primitive.execute();

  // copy output back to host
  const encoder = device.createCommandEncoder({
    label: "timed kernel run encoder",
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
    const errorstr = primitive.validate({ in: memsrcX32, out: memdest });
    if (errorstr == "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed: ${errorstr}`);
    }
  }
  // currently no timing computation, that's fine
}
