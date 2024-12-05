/* Define binary operations for both GPU and CPU */

class BinOp {
  constructor(args) {
    // no defaults! if something is undefined, go with it
    Object.assign(this, args);
  }
}

class BinOpAdd extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 0;
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
    this.op = (a, b) => a + b;
  }
}

export const BinOpAddU32 = new BinOpAdd({ datatype: "u32" });
export const BinOpAddF32 = new BinOpAdd({ datatype: "f32" });
export const BinOpAddI32 = new BinOpAdd({ datatype: "i32" });

class BinOpMin extends BinOp {
  constructor(args) {
    super(args);
    /* identity depends on datatype */
    switch (this.datatype) {
      case "f32":
        this.identity = 3.40282346638529e38; // FLT_MAX
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
  }
}

export const BinOpMinU32 = new BinOpMin({ datatype: "u32" });
export const BinOpMinF32 = new BinOpMin({ datatype: "f32" });
export const BinOpMinI32 = new BinOpMin({ datatype: "i32" });

class BinOpMax extends BinOp {
  constructor(args) {
    super(args);
    /* identity depends on datatype */
    switch (this.datatype) {
      case "f32":
        this.identity = -3.40282346638529e38; // -FLT_MAX
        break;
      case "i32":
        this.identity = 0xf0000000;
        break;
      case "u32": // fall-through OK
      default:
        this.identity = 0;
        break;
    }
    this.op = (a, b) => Math.max(a, b);
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return max(a,b);}`;
    this.wgslatomic = "atomicMax";
  }
}

export const BinOpMaxU32 = new BinOpMax({ datatype: "u32" });
export const BinOpMaxF32 = new BinOpMax({ datatype: "f32" });
export const BinOpMaxI32 = new BinOpMax({ datatype: "i32" });

class BinOpMultiply extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 1;
    this.op = (a, b) => a * b;
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a*b;}`;
  }
}

export const BinOpMultiplyU32 = new BinOpMultiply({ datatype: "u32" });
export const BinOpMultiplyF32 = new BinOpMultiply({ datatype: "f32" });
export const BinOpMultiplyI32 = new BinOpMultiply({ datatype: "i32" });
