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

export class NonCachingMap {
  // eslint-disable-next-line no-unused-vars
  constructor(_iterable) {
    // The constructor intentionally does nothing with the iterable
    // as no data is ever stored.
  }

  // has() always returns false as nothing is ever in the map.
  // eslint-disable-next-line no-unused-vars
  has(_key) {
    return false;
  }

  // get() always returns undefined as nothing is ever retrieved.
  // eslint-disable-next-line no-unused-vars
  get(_key) {
    return undefined;
  }

  // set() does nothing as no data is ever stored.
  // eslint-disable-next-line no-unused-vars
  set(_key, _value) {
    // Return 'this' for chainability, consistent with Map's set() behavior.
    return this;
  }

  // delete() always returns false as nothing can be deleted.
  // eslint-disable-next-line no-unused-vars
  delete(_key) {
    return false;
  }

  // clear() does nothing as there's nothing to clear.
  clear() {
    // No operation
  }

  // size always returns 0 as there are no entries.
  get size() {
    return 0;
  }

  get hits() {
    return 0;
  }

  get misses() {
    return 0;
  }

  // forEach() does nothing as there are no entries to iterate.
  // eslint-disable-next-line no-unused-vars
  forEach(_callbackFn, _thisArg) {
    // No operation
  }

  // keys() returns an empty iterator.
  keys() {
    return [].values(); // An iterator for an empty array
  }

  // values() returns an empty iterator.
  values() {
    return [].values(); // An iterator for an empty array
  }

  // entries() returns an empty iterator.
  entries() {
    return [].values(); // An iterator for an empty array
  }

  // Make it iterable, returning an empty iterator.
  [Symbol.iterator]() {
    return [].values(); // An iterator for an empty array
  }
}
