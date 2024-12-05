/* Define binary operations for both GPU and CPU */

class BinOp {
  constructor(args) {
    // defaults are BinOpAdd
    this.datatype = args.datatype ?? "u32";
    this.identity = args.identity ?? 0;
    this.op = args.op ?? ((a, b) => a + b);
    this.wgslfn =
      args.wgslfn ??
      `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;}`;
    this.wgslatomic = args.wgslatomic ?? "atomicAdd";
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
    this.wgslfn =
      args.wgslfn ??
      `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;}`;
    this.op = args.op ?? ((a, b) => a + b);
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
