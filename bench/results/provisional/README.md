# Provisional — NOT verified, do not cite

These TTFT-vs-prompt-length measurements were taken while the machine was
running unrelated CPU-heavy work (three `vnc_rhythm.py` processes at ~195%
CPU combined). A later repeat of the 252-token point measured 106.7 s where
three earlier sweeps agreed on 18.6–22.9 s — a 5x discrepancy attributable
to contention, not to either engine.

Additionally, the two engines were measured in SEPARATE sessions rather than
interleaved in one, so these numbers do not satisfy the same-session pairing
rule the rest of `bench/results/` follows. That rule exists precisely because
absolute throughput drifts with machine state while same-session ratios do not.

Re-measure interleaved, on an idle machine, before using any of this.

Also here: `qwen35-4b-zt-decode-depth.json`, which exists to record that the
8-token sampling window used by the sweep is too short to measure decode rate
— the same configuration re-measured with 64 generated tokens moved from
4.7 tok/s to 23.25 tok/s. Decode figures from the sweep are unusable.
