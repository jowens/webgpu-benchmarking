export class BaseTest {
  constructor(params, info) {
    Object.assign(this, params);
    Object.assign(this, info);
  }
}

export class BaseTestSuite {
  constructor(info, testClass, params, plots) {
    // put info objects into this class AND set them aside for later
    Object.assign(this, info);
    this.info = info;
    this.class = testClass;
    this.params = params;
    this.plots = plots;
  }
  getPlots() {
    if (this.plots instanceof Function) {
      return this.plots(this);
    } else {
      return this.plots;
    }
  }
  getTest(params) {
    return new this.class(params, this.info);
  }
}
