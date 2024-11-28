export class BaseTest {
  constructor(params) {
    Object.assign(this, params);
    if (this.constructor === BaseTest) {
      throw new Error("Cannot instantiate abstract class BaseTest directly.");
    }
  }
}

export class BasePrimitive {
  constructor(params) {
    Object.assign(this, params);
  }
}

export class BasePlot {}

export class BaseTestSuite {
  constructor(config) {
    Object.assign(this, config);
    this.config = config; // just to keep this encapsulated
    // depending on what's in config, instantiate:
    // Primitive, ProcessResults, Plot, Summarize
    //   each are a class
    // params
    //   is an object
    if ("plots" in config) {
      function f(plot, env) {
        if (typeof plot === "function") {
          return plot.call(env);
        } else {
          // presumably a plain object
          return plot;
        }
      }
      this.processedPlots = [];
      for (let plot of this.plots) {
        this.processedPlots.push(f(plot, this));
      }
    }
  }
  getTest(params) {
    // factory
    return new this.primitive(params);
  }
  getPlots() {
    return this.processedPlots;
  }
  getPlotsX() {
    function f(plot, env) {
      if (typeof plot === "function") {
        return plot.call(env);
      } else {
        // presumably a plain object
        return plot;
      }
    }
    console.log(this.plots[0].call(this));
    // return [this.plots[0].call(this)];
    // return [f(this.plots[0])];
    return this.plots.map(f);
    //return this.plots.map((Plot) => {
    //  Plot.call(this);
    //});
  }
}
