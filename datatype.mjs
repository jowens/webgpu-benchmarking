/* key*U32 derived from:
 * - http://stereopsis.com/radix.html, via
 * - https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingD3D12/Shaders/SortCommon.hlsl
 */

export class Datatype {
  constructor(datatypeString) {
    this.datatype = datatypeString;
    switch (this.datatype) {
      case "u32":
        this.keyToU32 = /* wgsl */ `fn keyToU32(u: ${this.datatype}) -> u32 { return u; }`;
        this.keyFromU32 = /* wgsl */ `fn keyFromU32(u: u32) -> ${this.datatype} { return u; }`;
        this.max = "0xffffffff";
        break;
      case "i32":
        this.keyToU32 = /* wgsl */ `fn keyToU32(i: ${this.datatype}) -> u32 {
  return bitcast<u32>(i) ^ 0x80000000;
}`;
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
}`;
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
}
