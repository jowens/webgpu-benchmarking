import { combinations, range, fail, delay, download } from "./util.mjs";
import { TimingHelper } from "./webgpufundamentals-timing.mjs";
import { TestInputBuffer, TestOutputBuffer } from "./testbuffer.mjs";

let Plot, JSDOM;
let saveJSON = false;
let saveSVG = false;
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
  Plot = await import("@observablehq/plot");
  JSDOM = await import("jsdom");
} else {
  // running in Chrome
  Plot = await import(
    "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"
  );
  /* begin https://github.com/sharonchoong/svg-exportJS */
  /* svg-exportJS prerequisite: canvg */
  await import("https://cdnjs.cloudflare.com/ajax/libs/canvg/3.0.9/umd.js");
  /* svg-exportJS plugin */
  await import("https://sharonchoong.github.io/svg-exportJS/svg-export.min.js");
  /* end https://github.com/sharonchoong/svg-exportJS */
  const urlParams = new URL(window.location.href).searchParams;
  saveJSON = urlParams.get("saveJSON"); // string or undefined
  if (saveJSON == "false") {
    saveJSON = false;
  }
  saveSVG = urlParams.get("saveSVG"); // string or undefined
  if (saveSVG == "false") {
    saveSVG = false;
  }
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
  AtomicGlobalU32ReduceBinOpsTestSuite,
  NoAtomicPKReduceTestSuite,
  AtomicGlobalU32SGReduceTestSuite,
  AtomicGlobalU32WGReduceTestSuite,
  AtomicGlobalF32WGReduceTestSuite,
  AtomicGlobalNonAtomicWGF32ReduceTestSuite,
  AtomicGlobalPrimedNonAtomicWGF32ReduceTestSuite,
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
  adapter.info.toJSON = function () {
    return {
      architecture: this.architecture,
      backend: this.backend,
      description: this.description,
      driver: this.driver,
      vendor: this.vendor,
    };
  };

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
    // AtomicGlobalU32ReduceBinOpsTestSuite,
    NoAtomicPKReduceTestSuite,
    //AtomicGlobalU32SGReduceTestSuite,
    //AtomicGlobalU32WGReduceTestSuite,
    //AtomicGlobalF32WGReduceTestSuite,
    //AtomicGlobalNonAtomicWGF32ReduceTestSuite,
    //AtomicGlobalPrimedNonAtomicWGF32ReduceTestSuite,
  ];
  //const testSuites = [AtomicGlobalU32ReduceTestSuite];

  let lastTestSeen = { testSuite: "", category: "" };

  const expts = new Array(); // push new rows (experiments) onto this
  for (const testSuite of testSuites) {
    console.log(testSuite);
    lastTestSeen = {
      testSuite: testSuite.testSuite,
      category: testSuite.category,
    };
    /* do we perform a computation? */
    if (testSuite?.primitive?.prototype.compute) {
      const uniqueRuns = new Set(); // if uniqueRuns is defined, don't run dups
      for (const params of combinations(testSuite.params)) {
        const primitive = testSuite.getPrimitive({ device, params });

        if (testSuite.uniqueRuns) {
          /* check if we've done this specific run before */
          /* fingerprint is a string, since strings can be keys in a Set() */
          const key = testSuite.uniqueRuns.map((x) => primitive[x]).join();
          if (uniqueRuns.has(key)) {
            /* already seen it, don't run it */
            continue;
          } else {
            uniqueRuns.add(key);
          }
        }

        /** for test purposes, let's initialize some buffers.
         *    Initializing is the responsibility of the test suite.
         *    Who determines their size/datatype?
         * Philosophy of a TestSuite:
         *    The primitive makes those decisions BECAUSE the test suite
         *    has passed in parameters in the parameter sweep that
         *    should let the primitive compute the relevant sizes.
         */
        const buffers = { in: [], out: [], uniforms: [] };
        for (let i = 0; i < primitive.numInputBuffers; i++) {
          buffers["in"].push(
            new TestInputBuffer(device, {
              datatype: primitive.datatype,
              size: primitive.memsrcSize,
            })
          );
        }
        primitive.inputs = buffers["in"];

        for (let i = 0; i < primitive.numOutputBuffers; i++) {
          buffers["out"].push(
            new TestOutputBuffer(device, {
              datatype: primitive.datatype,
              size: primitive.memdestSize,
            })
          );
        }
        primitive.outputs = buffers["out"];
        // TODO: uniforms

        // TEST FOR CORRECTNESS
        if (testSuite.validate) {
          // submit ONE run just for correctness
          await primitive.execute();

          // copy output back to host
          const copyEncoder = device.createCommandEncoder({
            label: "encoder: GPU output buffer data -> mappable buffers",
          });
          for (let i = 0; i < primitive.numOutputBuffers; i++) {
            copyEncoder.copyBufferToBuffer(
              buffers["out"][i].gpuBuffer,
              0,
              buffers["out"][i].mappableGPUBuffer,
              0,
              buffers["out"][i].mappableGPUBuffer.size
            );
          }
          const copyCommandBuffer = copyEncoder.finish();
          device.queue.submit([copyCommandBuffer]);

          // Copy results to CPU and validate them
          for (let i = 0; i < primitive.numOutputBuffers; i++) {
            await buffers["out"][i].mappableGPUBuffer.mapAsync(GPUMapMode.READ);
            buffers["out"][i].cpuBuffer = new (buffers["out"][
              i
            ].datatypeToTypedArray())(
              buffers["out"][i].mappableGPUBuffer.getMappedRange().slice()
            );
            buffers["out"][i].mappableGPUBuffer.unmap();
          }

          if (primitive.validate) {
            /* TODO: this is currently hardcoded to validating (in[0], out[0]) */
            const errorstr = primitive.validate(buffers);
            if (errorstr == "") {
              console.info("Validation passed");
            } else {
              console.error(`Validation failed: ${errorstr}`);
            }
          } else {
            console.error(
              `Primitive ${primitive.label} has no validation routine`
            );
          }
        } // end of TEST FOR CORRECTNESS

        // TEST FOR PERFORMANCE
        if (testSuite?.trials > 0) {
          primitive.execute({
            trials: testSuite.trials,
            enableCPUTiming: true,
          });

          primitive.getResult().then(({ gpuTotalTimeNS, cpuTotalTimeNS }) => {
            console.log(gpuTotalTimeNS, cpuTotalTimeNS);
            const result = {
              testSuite: testSuite.testSuite,
              category: testSuite.category,
            };
            if (gpuTotalTimeNS instanceof Array) {
              // gpuTotalTimeNS might be a list, in which case just add together for now
              result.gpuTotalTimeNSArray = gpuTotalTimeNS;
              gpuTotalTimeNS = gpuTotalTimeNS.reduce((x, a) => x + a, 0);
            }
            /* copy primitive's fields into result */
            for (const [field, value] of Object.entries(primitive)) {
              if (typeof value !== "function") {
                if (typeof value !== "object") {
                  result[field] = value;
                } else {
                  /* object - if it's got a constructor, use the name */
                  /* useful for "binop" or other parameters */
                  if (value?.constructor?.name) {
                    result[field] = value.constructor.name;
                  }
                }
              }
            }
            result.date = new Date();
            result.gpuinfo = adapter.info;
            result.gputime = gpuTotalTimeNS / testSuite.trials;
            result.cputime = cpuTotalTimeNS / testSuite.trials;
            result.cpugpuDelta = result.cputime - result.gputime;
            result.bandwidth = result.bytesTransferred / result.gputime;
            result.bandwidthCPU = result.bytesTransferred / result.cputime;
            if (primitive.gflops) {
              result.gflops = primitive.gflops(result.gputime);
            }
            expts.push(result);
          });
        } // end of TEST FOR PERFORMANCE
      } // end of running all combinations for this testSuite

      // delay is just to make sure previous jobs finish before plotting
      // almost certainly the timer->then clause above should be written in a way
      //   that lets me wait on it instead
      await delay(2000);
    }
    console.info(expts);

    for (let plot of testSuite.getPlots()) {
      /* default: if filter not specified, only take expts from the last test we ran */
      let filteredExpts = expts.filter(
        plot.filter ??
          ((row) =>
            row.testSuite == lastTestSeen.testSuite &&
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
          Plot.text([plot.text_tl ?? ""], {
            lineWidth: 30,
            dx: 5,
            frameAnchor: "top-left",
          }),
          Plot.text(plot.text_br ?? "", {
            lineWidth: 30,
            dx: 5,
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
      console.log(schema);
      const plotted = Plot.plot(schema);
      const div = document.querySelector("#plot");
      div.append(plotted);
      if (saveSVG) {
        svgExport.downloadSvg(
          div.lastChild,
          `${testSuite.testsuite}-${testSuite.category}`, // chart title: file name of exported image
          {}
        );
      }
      div.append(document.createElement("hr"));
    }
  }
  if (saveJSON) {
    download(expts, "application/json", "foo.json");
  }
}
export { main };
