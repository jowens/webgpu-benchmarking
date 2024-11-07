// not thrilled this uses idx, would rather do destructuring and not rely on idx/length
// would also like to see this done with a generatora
function combinations(obj) {
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

const range = (min, max /* [min, max] */) =>
  [...Array(max - min + 1).keys()].map((i) => i + min);

function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}

export { combinations, range, fail };
