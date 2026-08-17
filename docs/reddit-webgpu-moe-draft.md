TITLE: A 35B sparse MoE runs in a browser tab on hand-written WGSL — no compiler stack

BODY:

I've been writing an LLM inference engine for the browser by hand: TypeScript and
WGSL, no TVM, no compiler in the loop. The thing I wanted to know was whether the
compiler is load-bearing at all, or whether you can just write the shaders.

You can. The engine now runs Qwen3.6-35B-A3B — a sparse mixture-of-experts with
3-bit expert weights — entirely in a browser tab on a 32 GB M2 Max, at 65.56 tok/s
total wall clock (prefill + decode, 128-token target, median of 5 runs). Decode
alone is 74.87. WebLLM's pipeline emits 85 TVM-autotuned WGSL shaders for
Phi-3-mini; this engine has 51 hand-written shader files for the whole roster.

What made the MoE work in a tab, as opposed to just fitting:

- 3-bit expert stacks, so the weights land in a range where the browser will
  actually hand you the buffers. Allocation is not the hard part — usability is.
  The naive layout allocates fine and then thrashes the compressor.
- Experts are paged rather than resident. Top-k routing means most experts are
  cold on any given token, and the working set is what has to stay hot.
- One dispatch per fused region rather than per operation. The per-dispatch tax
  is the thing that decides whether browser compute is worth doing at all.

Also live in the same engine: Qwen3-30B-A3B (74.96 tok/s), Qwen3.5-9B, Phi-3-mini,
Llama-3.2-1B. All numbers are same-machine, same-session, and the protocol is in
BENCH.md — an M2 Max with 32 GB. Smaller machines run the smaller models; the 35B
needs the RAM.

Live (Chrome/Edge with WebGPU, nothing to install): https://zerotvm.com
Code: https://github.com/abgnydn/zero-tvm

Happy to answer anything about the shader side. The part I found most surprising
is how much of the compiler's value turned out to be scheduling I could do by
hand once, rather than search.
