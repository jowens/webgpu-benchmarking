import {
  combinations,
  fail,
  delay,
  download,
  datatypeToBytes,
} from "./util.mjs";
import { Buffer } from "./buffer.mjs";

let Plot, JSDOM;
let saveJSON = false;
let saveSVG = false;
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
  Plot = await import("@observablehq/plot");
  // eslint-disable-next-line no-unused-vars
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
  if (saveJSON === "false") {
    saveJSON = false;
  }
  saveSVG = urlParams.get("saveSVG"); // string or undefined
  if (saveSVG === "false") {
    saveSVG = false;
  }
}

if (Window.crossOriginIsolated) {
  console.info("IS cross-origin isolated");
} else {
  console.info("is NOT cross-origin isolated");
}

// tests
// import { NoAtomicPKReduceTestSuite } from "./reduce.mjs";
// import { HierarchicalScanTestSuite } from "./scan.mjs";
import {
  // DLDFScanTestSuite,
  // DLDFReduceTestSuite,
  DLDFScanAccuracyRegressionSuite,
  DLDFCachePerfTestSuite,
  DLDFDottedCachePerfTestSuite,
  DLDFDottedCachePerf2TestSuite,
  DLDFScanMiniSuite,
  DLDFFailureSuite,
  DLDFSingletonWithTimingSuite,
} from "./scandldf.mjs";
import { StoneberryScanMiniSuite } from "./stoneberry-scan.mjs";
import { subgroupAccuracyRegressionSuites } from "./subgroupRegression.mjs";
import { SortOneSweepRegressionSuite } from "./onesweep.mjs";
import { BasePrimitive } from "./primitive.mjs";

async function main(navigator) {
  const adapter = await navigator.gpu?.requestAdapter();
  console.info(adapter, adapter.features);
  const hasSubgroups = adapter.features.has("subgroups");
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  const device = await adapter?.requestDevice({
    requiredLimits: {
      maxBufferSize: 2147483648,
      maxStorageBufferBindingSize: 2147483644,
      maxComputeWorkgroupStorageSize: 32768,
    },
    requiredFeatures: [
      ...(hasTimestampQuery ? ["timestamp-query"] : []),
      ...(hasSubgroups ? ["subgroups"] : []),
    ],
  });
  console.info(device, JSON.stringify(device));
  adapter.info.toJSON = function () {
    return {
      architecture: this.architecture,
      description: this.description,
      vendor: this.vendor,
      subgroupMinSize: this.subgroupMinSize,
      subgroupMaxSize: this.subgroupMaxSize,
      isFallbackAdapter: this.isFallbackAdapter,
    };
  };
  console.info(JSON.stringify(adapter.info));

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
  // const testSuites = [
  // AtomicGlobalU32ReduceTestSuite,
  // AtomicGlobalU32ReduceBinOpsTestSuite,
  // NoAtomicPKReduceTestSuite,
  // HierarchicalScanTestSuite,
  // DLDFScanTestSuite,
  // DLDFReduceTestSuite,
  //AtomicGlobalU32SGReduceTestSuite,
  //AtomicGlobalU32WGReduceTestSuite,
  //AtomicGlobalF32WGReduceTestSuite,
  //AtomicGlobalNonAtomicWGF32ReduceTestSuite,
  //AtomicGlobalPrimedNonAtomicWGF32ReduceTestSuite,
  // ];
  //const testSuites = [AtomicGlobalU32ReduceTestSuite];

  //   let testSuites = subgroupAccuracyRegressionSuites;
  //testSuites.push(
  //  SortOneSweepRegressionSuite,
  //  DLDFScanAccuracyRegressionSuite,
  //  DLDFScanMiniSuite
  //);
  // let testSuites = [DLDFScanMiniSuite];
  // let testSuites = [DLDFScanAccuracyRegressionSuite];
  // let testSuites = [DLDFDottedCachePerfTestSuite];
  let testSuites = [DLDFDottedCachePerf2TestSuite];
  // let testSuites = [DLDFSingletonWithTimingSuite];
  // let testSuites = [DLDFFailureSuite];
  // let testSuites = [StoneberryScanMiniSuite];

  const expts = new Array(); // push new rows (experiments) onto this
  let primitiveCacheStats;
  let validations = { done: 0, errors: 0 };
  for (const testSuite of testSuites) {
    console.log(testSuite);
    const lastTestSeen = {
      testSuite: testSuite.testSuite,
      category: testSuite.category,
    };
    /* do we perform a computation? */
    if (testSuite?.primitive?.prototype.compute) {
      const uniqueRuns = new Set(); // if uniqueRuns is defined, don't run dups
      let testInputBuffer,
        testValuesBuffer, // key-value sort
        testOutputBuffer;
      const inputBufferIsOutputBuffer =
        testSuite.category === "sort" && testSuite.testSuite === "onesweep";
      for (const params of combinations(testSuite.params)) {
        if (params.binopbase && params.datatype && !params.binop) {
          /** we're iterating over both binopbase and datatype, which we can use
           * to construct binop */
          params.binop = new params.binopbase({ datatype: params.datatype });
        }
        const primitive = testSuite.getPrimitive({ device, ...params });

        /** for test purposes, let's initialize some buffers.
         *    Initializing is the responsibility of the test suite.
         *    Who determines their size/datatype?
         * Philosophy of a TestSuite:
         *    The primitive makes those decisions BECAUSE the test suite
         *    has passed in parameters in the parameter sweep that
         *    should let the primitive compute the relevant sizes.
         */

        /* these next two buffers have both CPU and GPU buffers within them */
        if (
          testInputBuffer?.datatype === primitive.datatype &&
          testInputBuffer?.length === primitive.inputLength &&
          !inputBufferIsOutputBuffer /* because we change the input buffer */
        ) {
          /* do nothing, keep existing buffer */
          /* this is (1) to reduce work and (2) to not reset the input data */
        } else {
          testInputBuffer = new Buffer({
            device,
            datatype: primitive.datatype,
            length: primitive.inputLength,
            label: "inputBuffer",
            createCPUBuffer: true,
            // initializeCPUBuffer: true /* fill with default data */,
            initializeCPUBuffer:
              testSuite.initializeCPUBuffer ??
              "randomizeAbsUnder1024" /* fill with default data */,
            storeCPUBackup: inputBufferIsOutputBuffer,
            createGPUBuffer: true,
            initializeGPUBuffer: true /* with CPU data */,
            createMappableGPUBuffer: inputBufferIsOutputBuffer,
          });
        }
        if (
          testSuite.category === "sort" &&
          testSuite.testSuite === "onesweep"
        ) {
          testInputBuffer.label = "keysInOut"; /* sort wants a different name */
        }
        if (
          testSuite.category === "scan" &&
          testSuite.testSuite === "stoneberry-scan"
        ) {
          testInputBuffer.label = "src";
        }
        primitive.registerBuffer(testInputBuffer);

        if (
          testSuite.category === "sort" &&
          testSuite.testSuite === "onesweep"
        ) {
          /* register the temp buffer */
          primitive.registerBuffer(
            new Buffer({
              device,
              datatype: primitive.datatype,
              length: primitive.inputLength,
              label: "keysTemp",
              createCPUBuffer: true /* debugging */,
              createGPUBuffer: true,
              createMappableGPUBuffer: true /* debugging */,
            })
          );
          testOutputBuffer = testInputBuffer; /* two names for same buffer */
        } else {
          testOutputBuffer = new Buffer({
            device,
            datatype:
              testSuite.category === "subgroups" &&
              testSuite.testSuite === "subgroupBallot"
                ? "vec4u"
                : primitive.datatype,
            length:
              "type" in primitive && primitive.type === "reduce"
                ? 1
                : primitive.inputLength,
            label: "outputBuffer",
            createGPUBuffer: true,
            createMappableGPUBuffer: true,
          });
          if (
            testSuite.category === "scan" &&
            testSuite.testSuite === "stoneberry-scan"
          ) {
            /* stoneberry scan expects a different label */
            testOutputBuffer.label = "prefixScan";
          }
          primitive.registerBuffer(testOutputBuffer);
        }
        const usesPayloadBuffers =
          testSuite.category === "sort" &&
          testSuite.testSuite === "onesweep" &&
          primitive.type === "keyvalue";

        if (usesPayloadBuffers) {
          /* these will eventually need to be named variables */
          testValuesBuffer = new Buffer({
            device,
            datatype: primitive.datatype,
            length: testInputBuffer.length,
            label: "payloadInOut",
            createCPUBuffer: true,
            initializeCPUBuffer: testSuite.initializeCPUBuffer,
            storeCPUBackup: inputBufferIsOutputBuffer,
            createGPUBuffer: true,
            initializeGPUBuffer: true /* with CPU data */,
            createMappableGPUBuffer: true,
          });
          primitive.registerBuffer(testValuesBuffer);
          primitive.registerBuffer(
            new Buffer({
              device,
              datatype: primitive.datatype,
              length: testInputBuffer.length,
              label: "payloadTemp",
              createCPUBuffer: true,
              createGPUBuffer: true,
              createMappableGPUBuffer: true /* debugging */,
            })
          );
          /** if we wanted "hist" and "passHist" visible to the CPU,
           * we would register them here */
        }

        let testDebugBuffer;
        if (primitive.knownBuffers.includes("debugBuffer")) {
          testDebugBuffer = new Buffer({
            device,
            datatype:
              "u32" /* probably need primitive.knownBufferDatatype[] here */,
            length: primitive.inputLength,
            label: "debugBuffer",
            createGPUBuffer: true,
            createMappableGPUBuffer: true,
          });
          primitive.registerBuffer(testDebugBuffer);
        }

        // TEST FOR CORRECTNESS
        if (testSuite.validate && primitive.validate) {
          // submit ONE run just for correctness
          await primitive.execute();
          await testOutputBuffer.copyGPUToCPU();
          if (
            testSuite.category === "sort" &&
            testSuite.testSuite === "onesweep" &&
            primitive.type === "keyvalue"
          ) {
            await testValuesBuffer.copyGPUToCPU();
          }
          if (testDebugBuffer) {
            await testDebugBuffer.copyGPUToCPU();
          }
          let validateArgs = {};
          if (
            testSuite.category === "sort" &&
            testSuite.testSuite === "onesweep"
          ) {
            validateArgs = {
              inputKeys: testInputBuffer.cpuBufferBackup,
              outputKeys: primitive.getBuffer("keysInOut"),
            };
            if (usesPayloadBuffers) {
              validateArgs.inputPayload = testValuesBuffer.cpuBufferBackup;
              validateArgs.outputPayload = primitive.getBuffer("payloadInOut");
            }
          }
          const errorstr = primitive.validate(validateArgs);
          if (errorstr === "") {
            // console.info("Validation passed", params);
          } else {
            validations.errors++;
            console.error("Validation failed for", params, ":", errorstr);
          }
          validations.done++;
        } // end of TEST FOR CORRECTNESS

        if (testSuite.uniqueRuns) {
          /* check if we've done this specific run before, and don't rerun if so */
          /* while it would be nice to put this test earlier, some params are
           * not set until the primitive is run once, so we'll just do it after
           * the validation run */
          /* fingerprint is a string, since strings can be keys in a Set() */
          const key = testSuite.uniqueRuns.map((x) => primitive[x]).join();
          if (uniqueRuns.has(key)) {
            /* already seen it, don't run it */
            continue;
          } else {
            uniqueRuns.add(key);
          }
        }

        // TEST FOR PERFORMANCE
        if (testSuite?.trials > 0) {
          if (inputBufferIsOutputBuffer) {
            /* Previous run's output erased current input */
            testInputBuffer.copyCPUBackupToCPU();
            testInputBuffer.copyCPUToGPU();
            if (usesPayloadBuffers) {
              testValuesBuffer.copyCPUBackupToCPU();
              testValuesBuffer.copyCPUToGPU();
            }
          }
          await primitive.execute({
            trials: testSuite.trials,
            enableGPUTiming: hasTimestampQuery,
            enableCPUTiming: true,
          });

          const { gpuTotalTimeNS, cpuTotalTimeNS } =
            await primitive.getTimingResult();
          processAndRecordResults(
            expts,
            gpuTotalTimeNS,
            cpuTotalTimeNS,
            primitive,
            testSuite,
            adapter.info
          );
        } // end of TEST FOR PERFORMANCE
      } // end of running all combinations for this testSuite

      /* record statistics every time through, overwriting any previous statistics */
      primitiveCacheStats = BasePrimitive.cacheStats(device);

      if (primitiveCacheStats) {
        console.info("Primitive cache stats", primitiveCacheStats);
      }
      if (validations.done > 0) {
        console.info(
          `${validations.done} validations complete${
            validations?.tested ? ` (${validations.tested})` : ""
          }, ${validations.errors} error${validations.errors != 1 ? "s" : ""}.`
        );
      }
    }

    console.info(expts);
    const plots = testSuite.getPlots();
    for (let plot of plots) {
      /* default: if filter not specified, only take expts from the last test we ran */
      let filteredExpts = expts.filter(
        plot.filter ??
          ((row) =>
            row.testSuite === lastTestSeen.testSuite &&
            row.category === lastTestSeen.category)
      );
      const mark = plot.mark ?? "lineY";
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
          Plot[mark](filteredExpts, {
            x: plot.x.field,
            y: plot.y.field,
            ...("fx" in plot && { fx: plot.fx.field }),
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
              ...("fx" in plot && { fx: plot.fx.field }),
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
        ...("fx" in plot && {
          fx: { label: plot.fx.label },
        }),
        ...("fy" in plot && {
          fy: { label: plot.fy.label },
        }),
        ...(("fx" in plot || "fy" in plot) && { grid: true }),
        color: { type: "ordinal", legend: true },
        width: 1280,
        title: plot?.title,
        subtitle: plot?.subtitle,
        caption: plot?.caption,
      };
      console.log(schema);
      const plotted = Plot.plot(schema);
      const div = document.querySelector("#plot");
      div.append(plotted);
      if (saveSVG) {
        // eslint-disable-next-line no-undef
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
  console.log("Finished.");
}

function processAndRecordResults(
  expts,
  gpuTotalTimeNS,
  cpuTotalTimeNS,
  primitive,
  testSuite,
  adapter
) {
  // ... (Calculates bandwidth, gflops, etc., and pushes to `expts`) ...
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
    if (typeof value !== "function" && !Array.isArray(value)) {
      if (typeof value !== "object") {
        result[field] = value;
      } else {
        /* object - if it's got a constructor, use a (useful) name */
        /* useful for "binop" or other parameters */
        if (value?.constructor?.name !== "Object") {
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
  result.inputBytes =
    primitive.inputLength * datatypeToBytes(primitive.datatype);
  result.bandwidthGPU = primitive.bytesTransferred / result.gputime;
  result.bandwidthCPU = primitive.bytesTransferred / result.cputime;
  result.inputItemsPerSecondE9GPU = primitive.inputLength / result.gputime;
  result.inputItemsPerSecondE9CPU = primitive.inputLength / result.cputime;
  if (primitive.gflops) {
    result.gflops = primitive.gflops(result.gputime);
  }
  expts.push({
    ...result,
    timing: "GPU",
    bandwidth: result.bandwidthGPU,
    inputItemsPerSecondE9: result.inputItemsPerSecondE9GPU,
  });
  expts.push({
    ...result,
    timing: "CPU",
    bandwidth: result.bandwidthCPU,
    inputItemsPerSecondE9: result.inputItemsPerSecondE9CPU,
  });
}

export { main };
