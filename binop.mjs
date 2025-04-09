/** Define binary operations for both GPU and CPU
 * Each class can optionally define:
 *
 * identity: the identity value for the operation
 * op: a JS function that takes two args and returns a value
 * wgslfn: a string that defines a WGSL function "binop" that takes two args
 *   and returns a value
 * wgslatomic: a string that names an atomic function (same function
 *   as wgslfn)
 * subgroupReduceOp: a string that names a function that reduces
 *   across a subgroup
 * subgroup{Inclusive, Exclusive}ScanOp: a string that names a function that
 *   {inclusive, exclusive}-scans across a subgroup
 */

class BinOp {
  constructor(args) {
    // no defaults! if something is undefined, go with it
    Object.assign(this, args);
  }
  toString() {
    return this.constructor.name;
  }
}

export class BinOpNop extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 42;
    // eslint-disable-next-line no-unused-vars
    this.op = (a, _b) => a;
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a;}`;
  }
}
export const BinOpNopU32 = new BinOpNop({ datatype: "u32" });

export class BinOpAdd extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 0;
    if (args.datatype == "f32") {
      const f32array = new Float32Array(3);
      this.op = (a, b) => {
        f32array[1] = a;
        f32array[2] = b;
        f32array[0] = f32array[1] + f32array[2];
        return f32array[0];
      };
    } else {
      this.op = (a, b) => a + b;
    }
    switch (this.datatype) {
      case "f32":
        break;
      case "i32":
        break;
      case "u32": // fall-through OK
      default:
        this.wgslatomic = "atomicAdd"; // u32 only
        break;
    }
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;}`;
    this.subgroupReduceOp = "subgroupAdd";
    this.subgroupInclusiveScanOp = "subgroupInclusiveAdd";
    this.subgroupExclusiveScanOp = "subgroupExclusiveAdd";
  }
}

export const BinOpAddU32 = new BinOpAdd({ datatype: "u32" });
export const BinOpAddF32 = new BinOpAdd({ datatype: "f32" });
export const BinOpAddI32 = new BinOpAdd({ datatype: "i32" });

export class BinOpMin extends BinOp {
  constructor(args) {
    super(args);
    /* identity depends on datatype */
    switch (this.datatype) {
      case "f32":
        this.identity = 3.402823466385288e38; // FLT_MAX
        break;
      case "i32":
        this.identity = 0x7fffffff;
        break;
      case "u32": // fall-through OK
      default:
        this.identity = 0xffffffff;
        break;
    }
    this.op = (a, b) => Math.min(a, b);
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return min(a,b);}`;
    this.wgslatomic = "atomicMin";
    this.subgroupReduceOp = "subgroupMin";
  }
}

export const BinOpMinU32 = new BinOpMin({ datatype: "u32" });
export const BinOpMinF32 = new BinOpMin({ datatype: "f32" });
export const BinOpMinI32 = new BinOpMin({ datatype: "i32" });

export class BinOpMax extends BinOp {
  constructor(args) {
    super(args);
    /* identity depends on datatype */
    switch (this.datatype) {
      case "f32":
        this.identity = -3.402823466385288e38; // -FLT_MAX
        break;
      case "i32":
        this.identity = 0x80000000;
        break;
      case "u32": // fall-through OK
      default:
        this.identity = 0;
        break;
    }
    this.op = (a, b) => Math.max(a, b);
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return max(a,b);}`;
    this.wgslatomic = "atomicMax";
    this.subgroupReduceOp = "subgroupMax";
  }
}

export const BinOpMaxU32 = new BinOpMax({ datatype: "u32" });
export const BinOpMaxF32 = new BinOpMax({ datatype: "f32" });
export const BinOpMaxI32 = new BinOpMax({ datatype: "i32" });

export class BinOpMultiply extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 1;
    this.op = (a, b) => a * b;
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a*b;}`;
    this.wgslatomic = "atomicMul";
    this.subgroupReduceOp = "subgroupMul";
  }
}

export const BinOpMultiplyU32 = new BinOpMultiply({ datatype: "u32" });
export const BinOpMultiplyF32 = new BinOpMultiply({ datatype: "f32" });
export const BinOpMultiplyI32 = new BinOpMultiply({ datatype: "i32" });
