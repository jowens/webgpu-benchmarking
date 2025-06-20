/* key*U32 derived from:
 * - http://stereopsis.com/radix.html, via
 * - https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingD3D12/Shaders/SortCommon.hlsl
 */

export class Datatype {
  constructor(datatypeString) {
    this.datatype = datatypeString;
    switch (this.datatype) {
      case "u32":
        this.keyToU32 = /* wgsl */ `fn keyToU32(u: ${this.datatype}) -> u32 { return u; }
fn keyToU32Native(u: ${this.datatype}) -> u32 { return keyToU32(u); }`;
        this.keyFromU32 = /* wgsl */ `fn keyFromU32(u: u32) -> ${this.datatype} { return u; }`;
        this.max = "0xffffffff";
        break;
      case "u64":
        this.keyToU32 = /* wgsl */ `fn keyToU32(u: ${this.wgslDatatype}) -> u32 { return u[0]; }
fn keyHighToU32(u: ${this.wgslDatatype}) -> u32 { return u[1]; }
fn keyToU32Native(u: ${this.wgslDatatype}) -> u32 { return u; }`;
        this.keyFromU32 = /* wgsl */ `fn keyFromU32(u: u32) -> ${this.wgslDatatype} {
          ${this.wgslDatatype} v;
          v[0] = u; v[1] = u;
          return v;
        }`;
        this.max = "0xffffffff";
        break;
      case "i32":
        this.keyToU32 = /* wgsl */ `fn keyToU32(i: ${this.datatype}) -> u32 {
  return bitcast<u32>(i) ^ 0x80000000;
}
fn keyToU32Native(i: ${this.datatype}) -> u32 { return keyToU32(i); }`;
        this.keyFromU32 = /* wgsl */ `fn keyFromU32(u: u32) -> ${this.datatype} {
  return bitcast<${this.datatype}>(u ^ 0x80000000);
}`;
        this.max = "0x7fffffff";
        break;
      case "f32":
        /** This one is tricky to get the typechecking correct.
         * When the sign bit is set, xor with 0xFFFFFFFF (flip every bit)
         * When the sign bit is unset, xor with 0x80000000 (flip the sign bit)
         */
        this.keyToU32 = /* wgsl */ `fn keyToU32(f: ${this.datatype}) -> u32 {
  var mask: u32 = bitcast<u32>(-(bitcast<i32>(bitcast<u32>(f) >> 31))) | 0x80000000;
  return bitcast<u32>(f) ^ mask;
}
fn keyToU32Native(f: ${this.datatype}) -> u32 { return keyToU32(f); }`;
        this.keyFromU32 = /* wgsl */ `fn keyFromU32(u: u32) -> ${this.datatype} {
  var mask: u32 = ((u >> 31) - 1) | 0x80000000;
  return bitcast<f32>(u ^ mask);
}`;
        this.max = "3.402823466385288e38"; // FLT_MAX
        break;
      default:
        console.info(
          `Datatype class construction: datatype ${datatypeString} unknown`
        );
        break;
    }
  }
  get wgslDatatype() {
    /* this is the NATIVE datatype, whatever key we're sorting */
    /* the non-u64 64b values are not supported but are correct */
    switch (this.datatype) {
      case "u64":
        return "vec2u";
      case "i64":
        return "vec2i";
      case "f64":
        return "vec2f";
      default:
        return this.datatype;
    }
  }

  get wgslU32Datatype() {
    /* this is the NATIVE data width but a u32 datatype */
    /* the non-u64 64b values are not supported but are correct */
    switch (this.datatype) {
      case "u64":
      case "i64":
      case "f64":
        return "vec2u";
      default: /* 32b */
        return "u32";
    }
  }

  get is64Bit() {
    switch (this.datatype) {
      case "u64":
        return true;
      default:
        return false;
    }
  }

  get bytesPerElement() {
    return this.is64Bit ? 8 : 4;
  }

  get bitsPerElement() {
    return this.bytesPerElement * 8;
  }
}
