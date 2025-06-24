/* key*U32 derived from:
 * - http://stereopsis.com/radix.html, via
 * - https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingD3D12/Shaders/SortCommon.hlsl
 */

/**
 * OneSweep sorts uint keys internally. Any other datatype needs to be
 * converted to a uint, or a vector of uints, for OneSweep's use.
 *
 * For 32b datatypes, this is straightforward:
 *
 * keyToU32(datatype) -> u32: Converts a native datatype to a u32
 * keyFromU32(u32) -> datatype: Converts a u32 to a native datatype
 *
 * For 64b datatypes, the conversion goes to and from vec2u instead of u32.
 */

export class Datatype {
  constructor(datatypeString) {
    this.datatype = datatypeString;
    switch (this.datatype) {
      case "u32":
        this.keyUintConversions = /* wgsl */ `fn keyToU32(u: ${this.datatype}) -> u32 { return u; }
fn keyFromU32(u: u32) -> ${this.datatype} { return u; }`;
        this.max = "0xffffffff";
        break;
      case "u64":
        this.keyUintConversions = /* wgsl */ `fn keyToU32(u2: ${this.wgslDatatype}) -> vec2u { return u2; }
fn keyFromU32(u2: vec2u) -> ${this.wgslDatatype} { return u2; }`;
        this.max = "0xffffffff";
        break;
      case "i32":
        this.keyUintConversions = /* wgsl */ `fn keyToU32(i: ${this.datatype}) -> u32 {
  return bitcast<u32>(i) ^ 0x80000000;
}
fn keyFromU32(u: u32) -> ${this.datatype} {
  return bitcast<${this.datatype}>(u ^ 0x80000000);
}`;
        this.max = "0x7fffffff";
        break;
      case "f32":
        /** This one is tricky to get the typechecking correct.
         * When the sign bit is set, xor with 0xFFFFFFFF (flip every bit)
         * When the sign bit is unset, xor with 0x80000000 (flip the sign bit)
         */
        this.keyUintConversions = /* wgsl */ `fn keyToU32(f: ${this.datatype}) -> u32 {
  var mask: u32 = bitcast<u32>(-(bitcast<i32>(bitcast<u32>(f) >> 31))) | 0x80000000;
  return bitcast<u32>(f) ^ mask;
}
fn keyFromU32(u: u32) -> ${this.datatype} {
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
      case "i64":
      case "f64":
        return true;
      default:
        return false;
    }
  }

  get bytesPerElement() {
    return this.is64Bit ? 8 : 4;
  }

  get wordsPerElement() {
    return this.is64Bit ? 2 : 1;
  }

  get bitsPerElement() {
    return this.bytesPerElement * 8;
  }
}
