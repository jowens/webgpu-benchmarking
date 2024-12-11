export class BaseTestSuite {
  constructor(config) {
    Object.assign(this, config);
    this.config = config; // just to keep this encapsulated
    // depending on what's in config, instantiate:
    // primitive, processResults, plots, summarize
    //   each are a class
    // params
    //   is an object
    if ("plots" in config) {
      /** this complexity is to make sure that a plot
       * that uses a template literal has it evaluated
       * in the context of this test suite. By default,
       * it's lexically scoped, which is useless. call()
       * makes sure it's evaluated in this class's context.
       */
      function f(plot, env) {
        if (typeof plot === "function") {
          return plot.call(env);
        } else {
          // presumably a plain object
          return plot;
        }
      }
      // TODO: i do not understand why 'map' does not work
      this.processedPlots = [];
      for (let plot of this.plots) {
        this.processedPlots.push(f(plot, this));
      }
    }
  }
  getPrimitive(deviceAndParams) {
    // factory
    // arg to primitive is a single object, so what's below combines device and params
    // we also need to add amy primitive-specific configuration info, specified in this.primitiveConfig
    Object.assign(deviceAndParams, this.primitiveConfig);
    const primitive = new this.primitive(deviceAndParams);
    /* copy all string fields from TestSuite -> Primitive */
    for (const key of Object.keys(this)) {
      if (typeof this[key] == "string") {
        primitive[key] = this[key];
      }
    }
    return primitive;
  }
  getPlots() {
    return this.processedPlots;
  }
}
