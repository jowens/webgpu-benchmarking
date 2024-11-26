import { combinations, range, fail, delay, download } from "./util.mjs";
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
  MembwSimpleTestSuite,
  MembwGSLTestSuite,
  MembwAdditionalPlotsSuite,
} from "./membwtest.mjs";
import { StridedReadTestSuite } from "./stridedreadtest.mjs";
import { RandomReadTestSuite } from "./randomreadtest.mjs";
import { MaddTestSuite } from "./maddtest.mjs";
import {
  SubgroupIDTestSuite,
  SubgroupSumSGTestSuite,
  SubgroupSumWGTestSuite,
} from "./subgroups.mjs";
import {
  AtomicGlobalU32ReduceTestSuite,
  AtomicGlobalU32SGReduceTestSuite,
  AtomicGlobalU32WGReduceTestSuite,
  AtomicGlobalF32WGReduceTestSuite,
  AtomicGlobalNonAtomicWGF32ReduceTest,
  AtomicGlobalPrimedNonAtomicWGF32ReduceTest,
} from "./reduce.mjs";

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

  // const testSuites = [MaddTestSuite];
  // const testSuites = [
  //   MembwSimpleTestSuite,
  //   MembwGSLTestSuite,
  //   MembwAdditionalPlotsSuite,
  // ];
  // const testSuites = [StridedReadTestSuite, RandomReadTestSuite];
  //  const testSuites = [
  //  SubgroupIDTestSuite,
  //  SubgroupSumSGTestSuite,
  //  SubgroupSumWGTestSuite,
  //];
  const testSuites = [
    // AtomicGlobalU32ReduceTestSuite,
    // AtomicGlobalU32SGReduceTestSuite,
    // AtomicGlobalU32WGReduceTestSuite,
    AtomicGlobalF32WGReduceTestSuite,
    AtomicGlobalNonAtomicWGF32ReduceTest,
    AtomicGlobalPrimedNonAtomicWGF32ReduceTest,
  ];

  let lastTestSeen = { testname: "", category: "" };

  const expts = new Array(); // push new rows (experiments) onto this
  for (const testSuite of testSuites) {
    for (const params of combinations(testSuite.params)) {
      const test = new testSuite.class(params);
      lastTestSeen = { testname: test.testname, category: test.category };
      /* skip computation if no kernel */
      if ("kernel" in test) {
        /* given number of workgroups, compute dispatch geometry that respects limits */
        /* TODO: handle non-powers-of-two workgroup sizes here */
        let dispatchGeometry;
        if ("dispatchGeometry" in test) {
          dispatchGeometry = test.dispatchGeometry;
        } else {
          dispatchGeometry = [test.workgroupCount, 1];
          while (
            dispatchGeometry[0] > device.limits.maxComputeWorkgroupsPerDimension
          ) {
            dispatchGeometry[0] = Math.ceil(dispatchGeometry[0] / 2);
            dispatchGeometry[1] *= 2;
          }
        }

        const memsrcf32 = new Float32Array(test.memsrcSize);
        const memsrcu32 = new Uint32Array(test.memsrcSize);
        for (let i = 0; i < test.memsrcSize; i++) {
          memsrcf32[i] = test?.randomizeInput
            ? Math.random() * 2.0 - 1.0
            : i & (2 ** 22 - 1);
          // Rand: [-1,1]; non-rand: roughly, range of 32b significand
          memsrcu32[i] = i == 0 ? 0 : memsrcu32[i - 1] + 1; // trying to get u32s
        }
        if (
          memsrcf32.byteLength !=
            test.memsrcSize * memsrcf32.BYTES_PER_ELEMENT ||
          memsrcu32.byteLength != test.memsrcSize * memsrcf32.BYTES_PER_ELEMENT
        ) {
          fail(
            `Test ${test.category} / ${test.testname}: memsrc{f,i}.byteLength (${memsrcf32.byteLength}, ${memsrcu32.byteLength}) incompatible with memsrcSize (${memsrcSize}))`
          );
        }
        const memdestBytes = test.memdestSize * 4;

        const computeModule = device.createShaderModule({
          label: `module: ${test.category} ${test.testname}`,
          code: test.kernel(),
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

          console.info(`workgroupCount: ${test.workgroupCount}
workgroup size: ${test.workgroupSize}
dispatchGeometry: ${dispatchGeometry}`);
          if (test.validate) {
            const errorstr = test.validate(
              // default: u32
              test.datatype == "f32" ? memsrcf32 : memsrcu32,
              memdest
            );
            if (errorstr == "") {
              console.info("Validation passed");
            } else {
              console.error(`Validation failed: ${errorstr}`);
            }
          }
          if (test.dumpF) {
            console.debug(`memdest: ${memdest}`);
          }

          timingHelper.getResult().then((ns) => {
            const result = {};
            /* copy test fields into result */
            for (const key in test) {
              if (typeof test[key] !== "function" && key !== "description") {
                result[key] = test[key];
              }
            }
            result.date = new Date();
            result.gpuinfo = adapter.info;
            result.time = ns / test.trials;
            result.cpuns =
              ((passEndTime - passStartTime) * 1000000.0) / test.trials;
            if (result.time == 0) {
              result.time = result.cpuns;
            }
            result.cpugpuDelta = result.cpuns - result.time;
            result.bandwidth = result.bytesTransferred / result.time;
            result.bandwidthCPU = result.bytesTransferred / result.cpuns;
            if (test.gflops) {
              result.gflops = test.gflops(result.time);
            }
            expts.push(result);
          });
        } // buffer size fits within device limits
        /* tear down */
        memsrcuBuffer.destroy();
        memsrcfBuffer.destroy();
        memdestBuffer.destroy();
        mappableMemdestBuffer.destroy();
      } // if there's a kernel to run
    } // end of running all combinations for this testSuite

    // delay is just to make sure previous jobs finish before plotting
    // almost certainly the timer->then clause above should be written in a way
    //   that lets me wait on it instead
    await delay(2000);
    console.info(expts);

    for (let plot of testSuite.class.plots) {
      /* default: if filter not specified, only take expts from the last test we ran */
      let filteredExpts = expts.filter(
        plot.filter ??
          ((row) =>
            row.testname == lastTestSeen.testname &&
            row.category == lastTestSeen.category)
      );
      console.info(
        "Filtered experiments for",
        plot.caption,
        "filter:",
        plot.filter,
        "filtered experiments:",
        filteredExpts,
        "unfiltered length:",
        expts.length,
        "filtered length:",
        filteredExpts.length
      );
      const schema = {
        marks: [
          Plot.lineY(filteredExpts, {
            x: plot.x.field,
            y: plot.y.field,
            ...("fy" in plot && { fy: plot.fy.field }),
            ...("stroke" in plot && {
              stroke: plot.stroke.field,
            }),
            tip: true,
          }),
          Plot.text(
            filteredExpts,
            Plot.selectLast({
              x: plot.x.field,
              y: plot.y.field,
              ...("stroke" in plot && {
                z: plot.stroke.field,
                text: plot.stroke.field,
              }),
              ...("fy" in plot && { fy: plot.fy.field }),
              textAnchor: "start",
              clip: false,
              dx: 3,
            })
          ),
          Plot.text([plot?.text_tl ?? ""], {
            lineWidth: 30,
            dx: 5,
            frameAnchor: "top-left",
          }),
          Plot.text([plot?.text_br ?? ""], {
            lineWidth: 30,
            dy: -10,
            frameAnchor: "bottom-right",
          }),
        ],
        x: { type: "log", label: plot?.x?.label ?? "XLABEL" },
        y: { type: "log", label: plot?.y?.label ?? "YLABEL" },
        ...("fy" in plot && {
          fy: { label: plot.fy.label },
        }),
        ...("fy" in plot && { grid: true }),
        color: { type: "ordinal", legend: true },
        title: plot?.title,
        subtitle: plot?.subtitle,
        caption: plot?.caption,
      };
      const plotted = Plot.plot(schema);
      const div = document.querySelector("#plot");
      div.append(plotted);
      div.append(document.createElement("hr"));
    }
  }
  // download(expts, "application/json", "foo.json");
}
export { main };
