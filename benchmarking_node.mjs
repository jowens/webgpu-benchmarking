import { globals, create } from "webgpu";
import { hostname } from "os";
Object.assign(globalThis, globals);
const gpuCreateFeatures = [
  "enable-dawn-features=use_user_defined_labels_in_backend",
];
if (process.platform === "win32" && hostname() === "jdowens2") {
  /* cannot figure out how to make Windows pick my discrete GPU */
  gpuCreateFeatures.push("adapter=NVIDIA RTX 4000 Ada Generation Laptop GPU");
}
const navigator = {
  gpu: create(gpuCreateFeatures),
};

import { main } from "./benchmarking.mjs";

main(navigator);
