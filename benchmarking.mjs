import { combinations, range, fail } from "./util.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";

let Plot, JSDOM;
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
  Plot = await import("@observablehq/plot");
  JSDOM = await import("jsdom");
} else {
  Plot = await import(
    "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"
  );
}

// tests
import { membwTest, membwGSLTest } from "./membwtest.mjs";
import { maddTest } from "./maddtest.mjs";
import { reducePerWGTest } from "./reduce.mjs";

// TODO
// strided reads
// random reads

async function main(navigator) {
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

  // const tests = [membwTest, maddTest];
  // const tests = [membwTest, membwGSLTest];
  const tests = [membwTest];

  for (const test of tests) {
    const expts = new Array();
    for (const param of combinations(test.parameters)) {
      const memsrcSize = param.memsrcSize;
      const workgroupSize = param.workgroupSize;
      const timingHelper = new TimingHelper(device);

      const workgroupCount = memsrcSize / workgroupSize;
      /* given number of workgroups, compute dispatch geometry that respects limits */
      let dispatchGeometry;
      if (Object.hasOwn(test, "dispatchGeometry")) {
        dispatchGeometry = test.dispatchGeometry(param);
      } else {
        dispatchGeometry = [workgroupCount, 1];
        while (
          dispatchGeometry[0] > device.limits.maxComputeWorkgroupsPerDimension
        ) {
          dispatchGeometry[0] /= 2;
          dispatchGeometry[1] *= 2;
        }
      }
      console.log(`workgroup count: ${workgroupCount}
workgroup size: ${workgroupSize}
dispatchGeometry: ${dispatchGeometry}
maxComputeWGPerDim: ${device.limits.maxComputeWorkgroupsPerDimension}`);

      const memsrc = new Float32Array(memsrcSize);
      for (let i = 0; i < memsrc.length; i++) {
        memsrc[i] = i & (2 ** 22 - 1); // roughly, range of 32b significand
      }

      const computeModule = device.createShaderModule({
        label: `module: ${test.name}`,
        code: test.kernel(param),
      });

      const kernelPipeline = device.createComputePipeline({
        label: `${test.name} compute pipeline`,
        layout: "auto",
        compute: {
          module: computeModule,
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

      const mappableMemdestBuffer = device.createBuffer({
        label: "mappable memory destination buffer",
        size: memsrc.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const maxBindingSize = device.limits.maxStorageBufferBindingSize;
      if (
        memsrcBuffer.size <= maxBindingSize &&
        memdestBuffer.size <= maxBindingSize &&
        mappableMemdestBuffer.size <= maxBindingSize
      ) {
        /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
        const kernelBindGroup = device.createBindGroup({
          label: "bindGroup for memcpy kernel",
          layout: kernelPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: memdestBuffer } },
            { binding: 1, resource: { buffer: memsrcBuffer } },
          ],
        });

        const prepassEncoder = device.createCommandEncoder({
          label: "prepass kernel encoder",
        });
        /* run the kernel before we start timing, don't time overhead */
        const kernelPrepass = prepassEncoder.beginComputePass({
          label: "untimed kernel compute prepass",
        });
        kernelPrepass.setPipeline(kernelPipeline);
        kernelPrepass.setBindGroup(0, kernelBindGroup);
        for (let i = 0; i < 1; i++) {
          /* just prime with one iteration */
          kernelPrepass.dispatchWorkgroups(...dispatchGeometry);
        }
        kernelPrepass.end();
        // Encode a command to copy the results to a mappable buffer.
        // this is (from, to)
        prepassEncoder.copyBufferToBuffer(
          memdestBuffer,
          0,
          mappableMemdestBuffer,
          0,
          mappableMemdestBuffer.size
        );
        const prepassCommandBuffer = prepassEncoder.finish();
        device.queue.submit([prepassCommandBuffer]);

        const encoder = device.createCommandEncoder({
          label: "timed kernel run encoder",
        });
        const kernelPass = timingHelper.beginComputePass(encoder, {
          label: "timed kernel compute pass",
        });
        kernelPass.setPipeline(kernelPipeline);
        kernelPass.setBindGroup(0, kernelBindGroup);
        // TODO handle not evenly divisible by wgSize
        for (let i = 0; i < test.trials; i++) {
          kernelPass.dispatchWorkgroups(...dispatchGeometry);
        }
        kernelPass.end();

        // Finish encoding and submit the commands
        const command_buffer = encoder.finish();
        await device.queue.onSubmittedWorkDone();
        const passStartTime = performance.now();
        device.queue.submit([command_buffer]);
        await device.queue.onSubmittedWorkDone();
        const passEndTime = performance.now();

        // Read the results
        await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
        const memdest = new Float32Array(
          mappableMemdestBuffer.getMappedRange().slice()
        );
        mappableMemdestBuffer.unmap();
        let errors = 0;
        for (let i = 0; i < memdest.length; i++) {
          if (!test.validate(memsrc[i], memdest[i], param)) {
            if (errors < 5) {
              console.log(
                `Error ${errors}: i=${i}, input=${memsrc[i]}, output=${memdest[i]}`
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
            param: param,
          };
          result.cpuns =
            ((passEndTime - passStartTime) * 1000000.0) / test.trials;
          if (result.time == 0) {
            result.time = result.cpuns;
          }
          result.cpugpuDelta = result.cpuns - result.time;
          if (test.bytesTransferred) {
            result.bytesTransferred = test.bytesTransferred(memsrc, memdest);
            result.bandwidth = result.bytesTransferred / result.time;
            result.bandwidthCPU = result.bytesTransferred / result.cpuns;
          }
          if (test.threadCount) {
            result.threadCount = test.threadCount(memsrc);
          }
          if (test.flopsPerThread) {
            result.flopsPerThread = test.flopsPerThread(param);
          }
          if (test.gflops && result.threadCount && result.flopsPerThread) {
            result.gflops = test.gflops(
              result.threadCount,
              result.flopsPerThread,
              result.time
            );
          }
          expts.push(result);
          console.log(result);
        });
      }
      /* tear down */
      memsrcBuffer.destroy();
      memdestBuffer.destroy();
      mappableMemdestBuffer.destroy();
    }
    console.log(expts);

    for (const testPlot of test.plots) {
      /* default: if filter not specified, only take expts from this test */
      const filteredExpts = testPlot?.filter
        ? testPlot?.filter(expts)
        : expts.filter((row) => row.name == test.name);
      const schema = {
        marks: [
          Plot.lineY(filteredExpts, {
            x: testPlot.x.field,
            y: testPlot.y.field,
            ...(Object.hasOwn(testPlot, "fy") && { fy: testPlot.fy.field }),
            stroke: testPlot.stroke.field,
            tip: true,
          }),
          Plot.text(
            filteredExpts,
            Plot.selectLast({
              x: testPlot.x.field,
              y: testPlot.y.field,
              z: testPlot.stroke.field,
              ...(Object.hasOwn(testPlot, "fy") && { fy: testPlot.fy.field }),
              text: testPlot.stroke.field,
              textAnchor: "start",
              dx: 3,
            })
          ),
          Plot.text([testPlot?.text_tl ?? ""], {
            lineWidth: 30,
            dx: 5,
            frameAnchor: "top-left",
          }),
          Plot.text([testPlot?.text_br ?? ""], {
            lineWidth: 30,
            dy: -10,
            frameAnchor: "bottom-right",
          }),
        ],
        x: { type: "log", label: testPlot?.x?.label ?? "XLABEL" },
        y: { type: "log", label: testPlot?.y?.label ?? "YLABEL" },
        ...(Object.hasOwn(testPlot, "fy") && {
          fy: { label: testPlot.fy.label },
        }),
        ...(Object.hasOwn(testPlot, "fy") && { grid: true }),
        color: { type: "ordinal", legend: true },
        title: testPlot?.title,
        subtitle: testPlot?.subtitle,
        caption: testPlot?.caption,
      };
      console.log(schema);
      const plot = Plot.plot(schema);
      const div = document.querySelector("#plot");
      div.append(plot);
    }
  }
}
export { main };
