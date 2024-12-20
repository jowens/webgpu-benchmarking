import { TimingHelper } from "./webgpufundamentals-timing.mjs";
import { BinOpAddU32, BinOpMinU32, BinOpMaxU32 } from "./binop.mjs";

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
  const memsrcSize = 2 ** 20;
  const memsrcu32 = new Uint32Array(memsrcSize);
  for (let i = 0; i < memsrcSize; i++) {
    memsrcu32[i] = i == 0 ? 11 : memsrcu32[i - 1] + 1; // trying to get u32s
  }
  const memdestBytes = 4;

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
    params: {
      /* tunable parameters - if none, use defaults */
    },
    datatype: "u32",
    gputimestamps: true, //// TODO should work without this
    binop: BinOpMinU32,
    buffers: [memdestBuffer, memsrcuBuffer],
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
  const memdest = new Uint32Array(
    mappableMemdestBuffer.getMappedRange().slice()
  );
  mappableMemdestBuffer.unmap();

  if (primitive.validate) {
    const errorstr = primitive.validate({ in: memsrcu32, out: memdest });
    if (errorstr == "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed: ${errorstr}`);
    }
  }
  // currently no timing computation, that's fine
}
