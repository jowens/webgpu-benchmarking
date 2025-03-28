import { globals, create } from "webgpu";
Object.assign(globalThis, globals);
const navigator = {
  gpu: create(["enable-dawn-features=use_user_defined_labels_in_backend"]),
};

import { main } from "./benchmarking.mjs";

main(navigator);
