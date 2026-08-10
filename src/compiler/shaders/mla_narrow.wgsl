// MLA_NARROW — f32 → f16, elementwise. One seam, one dispatch.
//
// mla_combine accumulates the latent context in f32 (a sum over every cached
// position — precisely the reduction f16 cannot hold) and mla_proj reads f16,
// so something has to narrow between them. Doing it inside mla_combine would
// save this dispatch and is the wrong trade: the real-weights check normalises
// by max|ref|, where f16's own representation error is ~4.9e-4 against a 1e-3
// bound the kernel already spends 3-4e-4 of. Rounding once at the END of the
// accumulation is the same arithmetic as rounding the result here; rounding
// INSIDE it is not, and it would be paid on a kernel already verified against
// real weights. So the narrowing lives in its own ten lines and the three
// verified MLA kernels stay byte-identical.
//
// Grid: ceil(n / 256). One thread per element.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_v : array<f16>;   // [n]
@group(0) @binding(1) var<storage, read> in_v : array<f32>;          // [n]

struct PODArgs {
  // Elements to cast (heads * kv_lora_rank). READ below, deliberately: with
  // layout:'auto' a binding the entry point never touches is dropped from the
  // pipeline's layout, and the engine's 3-entry bind group would then be
  // rejected against a 2-entry layout — a validation error at boot, not a
  // wrong number, but a confusing one to chase.
  n: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn mla_narrow(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= podArgs.n) { return; }
  out_v[i] = f16(in_v[i]);
}
