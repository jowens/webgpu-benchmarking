// copied from
// https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html by gman@

function assert(cond, msg = "") {
  if (!cond) {
    throw new Error(msg);
  }
}

class TimingHelper {
  #canTimestamp;
  #device;
  #querySet;
  #resolveBuffer;
  #resultBuffer;
  #resultBuffers = [];
  #passNumber;
  #maxPasses;
  // state can be 'free', 'in progress', 'need resolve', 'wait for result'
  #state = "free";

  constructor(device, maxPasses = 1) {
    this.#device = device;
    this.#passNumber = 0;
    this.#maxPasses = maxPasses;
    this.#canTimestamp = device.features.has("timestamp-query");
    if (this.#canTimestamp) {
      this.#querySet = device.createQuerySet({
        type: "timestamp",
        count: maxPasses * 2,
      });
      this.#resolveBuffer = device.createBuffer({
        size: this.#querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  #beginTimestampPass(encoder, fnName, descriptor) {
    if (this.#canTimestamp) {
      assert(
        /* haven't started or finished all passes yet */
        this.#state === "free" || this.#state == "in progress",
        `state not free (state = ${this.#state})`
      );

      const pass = encoder[fnName]({
        ...descriptor,
        ...{
          timestampWrites: {
            querySet: this.#querySet,
            beginningOfPassWriteIndex: this.#passNumber * 2,
            endOfPassWriteIndex: this.#passNumber * 2 + 1,
          },
        },
      });

      this.#passNumber++;
      if (this.#passNumber == this.#maxPasses) {
        /* finished all passes */
        this.#state = "need resolve";
      } else {
        /* still have passes to do */
        this.#state = "in progress";
      }

      const resolve = () => this.#resolveTiming(encoder);
      pass.end = (function (origFn) {
        return function () {
          origFn.call(this);
          resolve();
        };
      })(pass.end);

      return pass;
    } else {
      return encoder[fnName](descriptor);
    }
  }

  beginRenderPass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, "beginRenderPass", descriptor);
  }

  beginComputePass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, "beginComputePass", descriptor);
  }

  #resolveTiming(encoder) {
    if (!this.#canTimestamp) {
      return;
    }
    if (this.#passNumber != this.#maxPasses) {
      return;
    }
    assert(this.#state === "need resolve", "must call addTimestampToPass");
    this.#state = "wait for result";

    this.#resultBuffer =
      this.#resultBuffers.pop() ||
      this.#device.createBuffer({
        size: this.#resolveBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

    encoder.resolveQuerySet(
      this.#querySet,
      0,
      this.#querySet.count,
      this.#resolveBuffer,
      0
    );
    encoder.copyBufferToBuffer(
      this.#resolveBuffer,
      0,
      this.#resultBuffer,
      0,
      this.#resultBuffer.size
    );
  }

  async getResult() {
    if (!this.#canTimestamp) {
      return 0;
    }
    assert(this.#state === "wait for result", "must call resolveTiming");
    this.#state = "free";

    const resultBuffer = this.#resultBuffer;
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(resultBuffer.getMappedRange());
    /* I need to read about functional programming in JS to make below pretty */
    const durations = [];
    for (var idx = 0; idx < times.length; idx += 2) {
      durations.push(Number(times[idx + 1] - times[idx]));
    }
    resultBuffer.unmap();
    this.#resultBuffers.push(resultBuffer);
    this.#querySet.destroy();
    this.#resolveBuffer.destroy();
    return durations;
  }
}
