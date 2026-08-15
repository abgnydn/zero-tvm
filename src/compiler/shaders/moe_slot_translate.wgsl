// MOE_SLOT_TRANSLATE — expert ids → pool slot ids through a device-resident
// map, so a pooled decode token's hit path needs no CPU round trip.
//
// The optimistic recorder (docs/MOE_CHUNK_PLAN.md, 2026-08-15) records a whole
// token into ONE submit: the router writes expert ids, this kernel rewrites
// them into slot indices using the map the CPU last uploaded, and the raw
// expert ids are copied to a staging buffer the CPU reads AFTER the token.
// The expert matmuls never see an expert id, only a slot — same contract the
// serial pooled path established, minus the mid-token readback.
//
// A MISS (map[e] == 0xffffffffu) translates to slot 0 — in-bounds garbage.
// That is deliberate: there is no correct answer resident, the late-arriving
// readback will name the miss, and the token replays from this layer with the
// expert uploaded. Wrongness here must be BOUNDED, not avoided; clamping to a
// real slot keeps every downstream read in range so the only casualty is the
// value, never memory safety.
//
// ids_in and ids_out are separate bindings on purpose: the staging copy must
// carry EXPERT ids (the CPU resolves routing from them), so the translation
// cannot be in place.

@group(0) @binding(0) var<storage, read_write> ids_out : array<u32>;
@group(0) @binding(1) var<storage, read> ids_in : array<u32>;
@group(0) @binding(2) var<storage, read> slot_map : array<u32>;

struct PODArgs {
  N : u32   // entries to translate: topK plus the shared expert when present
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn moe_slot_translate(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= podArgs.N) { return; }
  let s : u32 = slot_map[ids_in[gid.x]];
  ids_out[gid.x] = select(s, 0u, s == 0xffffffffu);
}
