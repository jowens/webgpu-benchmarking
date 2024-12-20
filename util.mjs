// not thrilled this uses idx, would rather do destructuring and not rely on idx/length
// would also like to see this done with a generator
export function combinations(obj) {
  const keys = Object.keys(obj);
  const values = Object.values(obj);

  function gen(values, idx) {
    if (idx === values.length) {
      return [{}]; // Base case: return an array with an empty object
    }

    const current = values[idx];
    const remaining = gen(values, idx + 1); // recurse

    const combos = [];
    for (const item of current) {
      for (const combination of remaining) {
        combos.push({ ...combination, [keys[idx]]: item }); // Combine the current item with each of the combinations from the rest
      }
    }
    return combos;
  }

  return gen(values, 0);
}

export const range = (min, max /* [min, max] */) =>
  [...Array(max - min + 1).keys()].map((i) => i + min);

export function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}

export function datatypeToTypedArray(datatype) {
  switch (datatype) {
    case "f32":
      return Float32Array;
    case "i32":
      return Int32Array;
    case "u32":
      return Uint32Array;
  }
  return undefined;
}

// https://stackoverflow.com/questions/8896327/jquery-wait-delay-1-second-without-executing-code
export const delay = (millis) =>
  new Promise((resolve, reject) => {
    setTimeout((_) => resolve(), millis);
  });

// https://stackoverflow.com/questions/3665115/how-to-create-a-file-in-memory-for-user-to-download-but-not-through-server
export function download(content, mimeType, filename) {
  const a = document.createElement("a"); // Create "a" element
  if (mimeType == "application/json") {
    content = JSON.stringify(content);
  }
  const blob = new Blob([content], { type: mimeType }); // Create a blob (file-like object)
  const url = URL.createObjectURL(blob); // Create an object URL from blob
  a.setAttribute("href", url); // Set "a" element link
  a.setAttribute("download", filename); // Set download filename
  a.click(); // Start downloading
  URL.revokeObjectURL(url);
}
