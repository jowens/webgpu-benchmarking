"use strict";
import pkg from "../../src/dawn-build/dawn.node";
const { create, globals } = pkg;
Object.assign(globalThis, globals); // Provides constants like GPUBufferUsage.MAP_READ
import { fail } from "./util.mjs";
import { main } from "./benchmarking.mjs";
let navigator = {
  gpu: create(["enable-dawn-features=use_user_defined_labels_in_backend"]),
};
import * as Plot from "@observablehq/plot";
// import { JSDOM } from "jsdom";

if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
} else {
  fail("Use this only from Node.");
}
main();
