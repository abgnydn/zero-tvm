// IEEE-754 half-precision (f16) conversion helpers.
//
// Node 22 in CI does not expose Float16Array, and the kernels store
// activations/weights as f16, so we need encode/decode that exactly match
// what the GPU reads back. encode (round-to-nearest-even) and decode are
// self-consistent: decode(encode(x)) is the f16-rounded value of x, and the
// GPU decodes the same bits identically.

const _f = new Float32Array(1)
const _u = new Uint32Array(_f.buffer)

/** f32 number -> u16 holding its IEEE-754 half-precision bit pattern. */
export function f32ToF16Bits(val) {
  _f[0] = val
  const x = _u[0]
  const sign = (x >>> 16) & 0x8000
  let exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0) // inf / nan
  exp = exp - 127 + 15
  if (exp >= 0x1f) return sign | 0x7c00 // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return sign // underflow -> signed zero
    mant |= 0x800000
    const sh = 14 - exp
    let h = mant >> sh
    const rem = mant & ((1 << sh) - 1)
    const half = 1 << (sh - 1)
    if (rem > half || (rem === half && (h & 1))) h++ // round to nearest even
    return sign | h
  }
  let h = (exp << 10) | (mant >> 13)
  const rem = mant & 0x1fff
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++ // round to nearest even
  return sign | h
}

/** u16 IEEE-754 half bit pattern -> f32 number. */
export function f16BitsToF32(h) {
  const sign = (h & 0x8000) << 16
  const exp = (h >>> 10) & 0x1f
  const mant = h & 0x3ff
  let bits
  if (exp === 0) {
    if (mant === 0) {
      bits = sign
    } else {
      let e = -1
      let m = mant
      do {
        e++
        m <<= 1
      } while (!(m & 0x400))
      bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13)
    }
  } else if (exp === 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13)
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13)
  }
  _u[0] = bits >>> 0
  return _f[0]
}

/** Round a JS number to the nearest representable f16 value. */
export const toF16 = (x) => f16BitsToF32(f32ToF16Bits(x))

/** number[] -> Uint16Array of f16 bit patterns (ready for writeBuffer). */
export const f16Array = (arr) => Uint16Array.from(arr, f32ToF16Bits)

/** Uint16Array of f16 bits -> number[]. */
export const fromF16Array = (u16) => Array.from(u16, f16BitsToF32)

// f16-rounded arithmetic, for kernels that accumulate in f16 (e.g. fused_ffn).
export const f16add = (a, b) => toF16(a + b)
export const f16mul = (a, b) => toF16(a * b)
/** Single-rounding fused multiply-add, matching WGSL fma() in f16. */
export const f16fma = (a, b, c) => toF16(a * b + c)
