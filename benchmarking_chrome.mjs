const isLocalhost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:";

const modulePath =
  (isLocalhost ? "http://localhost:8000" : "https://jowens.github.io") +
  "/webgpu-benchmarking/benchmarking.mjs";

import(modulePath)
  .then(({ main }) => {
    main(navigator);
  })
  .catch((error) => {
    console.error("Error loading module", error);
  });

//import { main } from "http://localhost:8000/webgpu-benchmarking/benchmarking.mjs";
// main(navigator);
