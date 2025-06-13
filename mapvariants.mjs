export class CountingMap {
  #hits = 0; // Private class field for hits
  #misses = 0; // Private class field for misses
  #map; // Private class field for the internal Map
  #enabled;

  constructor({ iterable, enabled }) {
    this.#map = new Map(iterable);
    if (enabled) {
      this.enable();
    } else {
      this.disable();
    }
  }

  // Override the get() method to count hits and misses
  get(key) {
    if (this.disabled) {
      return undefined;
    }
    if (this.#map.has(key)) {
      this.#hits++; // Accessing private field
      return this.#map.get(key);
    } else {
      this.#misses++; // Accessing private field
      return undefined;
    }
  }

  // Delegate other Map methods to the internal #map
  set(key, value) {
    if (this.enabled) {
      return this.#map.set(key, value);
    } else {
      return undefined;
    }
  }

  has(key) {
    return this.enabled && this.#map.has(key);
  }

  delete(key) {
    return this.#map.delete(key);
  }

  clear() {
    this.#map.clear();
    this.#hits = 0; // Reset private counts on clear
    this.#misses = 0; // Reset private counts on clear
  }

  get size() {
    return this.#map.size;
  }

  // Public getters to expose the private counts
  get hits() {
    return this.#hits;
  }

  get misses() {
    return this.#misses;
  }

  enable() {
    this.#enabled = true;
  }

  disable() {
    this.#enabled = false;
  }

  get enabled() {
    return this.#enabled;
  }

  get disabled() {
    return !this.#enabled;
  }

  // Iteration methods (delegate to the internal #map's iterators)
  forEach(callbackFn, thisArg) {
    this.#map.forEach(callbackFn, thisArg);
  }

  keys() {
    return this.#map.keys();
  }

  values() {
    return this.#map.values();
  }

  entries() {
    return this.#map.entries();
  }

  // Make it iterable
  [Symbol.iterator]() {
    return this.#map[Symbol.iterator]();
  }
}
