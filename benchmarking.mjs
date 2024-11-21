import { combinations, range, fail, delay } from "./util.mjs";
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
import {
  MembwSimpleTest,
  MembwGSLTest,
  MembwAdditionalPlots,
} from "./membwtest.mjs";
import { stridedReadTest } from "./stridedreadtest.mjs";
import { randomReadTest } from "./randomreadtest.mjs";
import { maddTest } from "./maddtest.mjs";
import { reducePerWGTest } from "./reduce.mjs";
import { subgroupSumWGTest } from "./subgroups.mjs";

// TODO
// strided reads
// random reads

async function main(navigator) {
  const adapter = await navigator.gpu?.requestAdapter();
  const hasSubgroups = adapter.features.has("subgroups");
  const canTimestamp = adapter.features.has("timestamp-query");
  const device = await adapter?.requestDevice({
    requiredLimits: {
      maxBufferSize: 4294967296,
      maxStorageBufferBindingSize: 4294967292,
    },
    requiredFeatures: [
      ...(canTimestamp ? ["timestamp-query"] : []),
      ...(hasSubgroups ? ["subgroups"] : []),
    ],
  });

  if (!device) {
    fail("Fatal error: Device does not support WebGPU.");
  }

  // const tests = [membwTest, maddTest];
  const tests = [MembwSimpleTest, MembwGSLTest, MembwAdditionalPlots];
  // const tests = [membwTest];
  // const tests = [stridedReadTest];
  // const tests = [randomReadTest];
  // const tests = [stridedReadTest, randomReadTest];
  // const tests = [subgroupSumWGTest];

  const expts = new Array(); // push new rows (experiments) onto this
  for (const testClass of tests) {
    const test = new testClass();
    if ("kernel" in test) {
      /* skip computation if no kernel */
      for (const param of combinations(test.parameters)) {
        /** general hierarchy of setting these key parameters:
         * - First, use the value from test.parameters
         * - Second, use the function in the test
         * - Third, use a reasonable default
         */
        const memsrcSize = param.memsrcSize ?? test.memsrcSize(param);
        const memdestSize =
          param.memdestSize ?? test.memdestSize?.(param) ?? memsrcSize;
        const workgroupSize = param.workgroupSize ?? test.workgroupSize(param);
        const workgroupCount =
          param.workgroupCount ??
          test.workgroupCount?.(param) ??
          Math.ceil(memsrcSize / workgroupSize);

        /* given number of workgroups, compute dispatch geometry that respects limits */
        /* TODO: handle non-powers-of-two workgroup sizes here */
        let dispatchGeometry;
        if ("dispatchGeometry" in test) {
          dispatchGeometry = test.dispatchGeometry(param);
        } else {
          dispatchGeometry = [workgroupCount, 1];
          while (
            dispatchGeometry[0] > device.limits.maxComputeWorkgroupsPerDimension
          ) {
            dispatchGeometry[0] = Math.ceil(dispatchGeometry[0] / 2);
            dispatchGeometry[1] *= 2;
          }
        }

        const memsrcf32 = new Float32Array(memsrcSize);
        const memsrcu32 = new Uint32Array(memsrcSize);
        for (let i = 0; i < memsrcSize; i++) {
          memsrcf32[i] = i & (2 ** 22 - 1); // roughly, range of 32b significand
          memsrcu32[i] = i == 0 ? 0 : memsrcu32[i - 1] + 1; // trying to get u32s
        }
        if (
          memsrcf32.byteLength != memsrcSize * 4 ||
          memsrcu32.byteLength != memsrcSize * 4
        ) {
          fail(
            `Test ${test.category} / ${test.testname}: memsrc{f,i}.byteLength (${memsrcf32.byteLength}, ${memsrcu32.byteLength}) incompatible with memsrcSize (${memsrcSize}))`
          );
        }
        const memdestBytes = memdestSize * 4;

        const computeModule = device.createShaderModule({
          label: `module: ${test.category} ${test.testname}`,
          code: test.kernel(
            param,
            memsrcSize /* this is "number of threads" */
          ),
        });

        const kernelPipeline = device.createComputePipeline({
          label: `${test.category} ${test.testname} compute pipeline`,
          layout: "auto",
          compute: {
            module: computeModule,
          },
        });

        // allocate/create buffers on the GPU to hold in/out data
        const memsrcuBuffer = device.createBuffer({
          label: "memory source buffer (int)",
          size: memsrcu32.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(memsrcuBuffer, 0, memsrcu32);
        const memsrcfBuffer = device.createBuffer({
          label: "memory source buffer (float)",
          size: memsrcf32.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(memsrcfBuffer, 0, memsrcf32);

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

        const maxBindingSize = device.limits.maxStorageBufferBindingSize;
        if (
          (memsrcuBuffer.size <= maxBindingSize ||
            memsrcfBuffer.size <= maxBindingSize) &&
          memdestBuffer.size <= maxBindingSize &&
          mappableMemdestBuffer.size <= maxBindingSize
        ) {
          /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
          const kernelBindGroup = device.createBindGroup({
            label: "bindGroup for memcpy kernel",
            layout: kernelPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: memdestBuffer } },
              {
                binding: 1,
                resource: {
                  buffer:
                    test.datatype == "u32" ? memsrcuBuffer : memsrcfBuffer,
                },
              },
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

          const timingHelper = new TimingHelper(device);
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
          const memdest =
            test.datatype == "u32"
              ? new Uint32Array(mappableMemdestBuffer.getMappedRange().slice())
              : new Float32Array(
                  mappableMemdestBuffer.getMappedRange().slice()
                );
          mappableMemdestBuffer.unmap();
          let errors = 0;
          let last = 0;
          if (test.validate) {
            for (let i = 0; i < memdest.length; i++) {
              if (!test.validate(memsrcu32[i], memdest[i], param)) {
                if (errors < 5) {
                  console.log(
                    `Error ${errors}: i=${i}, input=0x${memsrcu32[i].toString(
                      16
                    )}, output=0x${memdest[i].toString(16)}`
                  );
                }
                errors++;
                last = i;
              }
            }
          }
          console.log(
            `Last error: i=${last}, input=0x${memsrcu32[last].toString(
              16
            )}, output=0x${memdest[last].toString(16)}`
          );

          console.log(`workgroupCount: ${workgroupCount}
workgroup size: ${workgroupSize}
dispatchGeometry: ${dispatchGeometry}`);
          if (test.dumpF) {
            console.log(`memdest: ${memdest}`);
          }
          // dump | memdest: ${[].map.call(memdest, (x) => "0x" + x.toString(16))}`);

          if (errors > 0) {
            console.log(`Memdest size: ${memdest.length} | Errors: ${errors}`);
          } else {
            console.log(`Memdest size: ${memdest.length} | No errors!`);
          }

          timingHelper.getResult().then((ns) => {
            const result = {
              category: test.category,
              testname: test.testname,
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
              result.bytesTransferred = test.bytesTransferred(
                memsrcu32,
                memdest
              );
              result.bandwidth = result.bytesTransferred / result.time;
              result.bandwidthCPU = result.bytesTransferred / result.cpuns;
            }
            if (test.threadCount) {
              result.threadCount = test.threadCount(memsrcu32);
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
        memsrcuBuffer.destroy();
        memsrcfBuffer.destroy();
        memdestBuffer.destroy();
        mappableMemdestBuffer.destroy();
      }
    }
    // delay is just to make sure previous jobs finish before plotting
    // almost certainly the timer->then clause above should be written in a way
    //   that lets me wait on it instead
    await delay(2000);
    console.log(expts);

    for (const testPlot of test.plots) {
      /* default: if filter not specified, only take expts from this test */
      const filteredExpts = expts.filter(
        testPlot.filter ??
          ((row) =>
            row.testname == test.testname && row.category == test.category)
      );
      console.log(
        "Filtered experiments for",
        testPlot.caption,
        testPlot.filter,
        filteredExpts,
        expts.length,
        filteredExpts.length
      );
      const schema = {
        marks: [
          Plot.lineY(filteredExpts, {
            x: testPlot.x.field,
            y: testPlot.y.field,
            ...("fy" in testPlot && { fy: testPlot.fy.field }),
            ...("stroke" in testPlot && {
              stroke: testPlot.stroke.field,
            }),
            tip: true,
          }),
          Plot.text(
            filteredExpts,
            Plot.selectLast({
              x: testPlot.x.field,
              y: testPlot.y.field,
              ...("stroke" in testPlot && {
                z: testPlot.stroke.field,
              }),
              ...("fy" in testPlot && { fy: testPlot.fy.field }),
              ...("stroke" in testPlot && {
                text: testPlot.stroke.field,
              }),
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
        ...("fy" in testPlot && {
          fy: { label: testPlot.fy.label },
        }),
        ...("fy" in testPlot && { grid: true }),
        color: { type: "ordinal", legend: true },
        title: testPlot?.title,
        subtitle: testPlot?.subtitle,
        caption: testPlot?.caption,
      };
      console.log(schema);
      const plot = Plot.plot(schema);
      const div = document.querySelector("#plot");
      div.append(plot);
      div.append(document.createElement("hr"));
    }
  }
}
export { main };
