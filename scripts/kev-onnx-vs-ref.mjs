// KEV-ONNX-VS-REF — Nico Martin's onnx-community kev ONNX export, through his
// open-jev library, against kev's own f32 torch forward on the same records —
// the same comparison scripts/kev-parity.mjs makes for the engine's files.
// Questions are passed with kev's already-rendered option strings so both
// paths pack identical text.
//
// Runs where `open-jev` and `@huggingface/transformers` are installed (this
// repository does not depend on them; the run recorded in
// docs/kev-parity/kev-onnx-q4.txt used ~/dev/open-jev-test with open-jev 0.1.2,
// Transformers.js 4.3.0, onnxruntime-node 1.30.0, dtype q4 on the CPU):
//
//   cp scripts/kev-onnx-vs-ref.mjs <dir with open-jev>/ && cd <that dir> &&
//   node kev-onnx-vs-ref.mjs ~/dev/zero-tvm/tests/fixtures/kev-records.json /tmp/ref-kev [kev-0.6b|kev-4b] [q4|q4f16]
import { readFileSync } from "node:fs";
import { OpenJev } from "open-jev";

const [recordsPath, refDir, modelArg, dtypeArg] = process.argv.slice(2);
const records = JSON.parse(readFileSync(recordsPath, "utf8"));
const ref = JSON.parse(readFileSync(`${refDir}/meta.json`, "utf8")).records;
const jev = await OpenJev.load({ model: modelArg ?? "kev-0.6b", dtype: dtypeArg ?? "q4" });
console.log(`onnx: ${jev.runtime.model} ${jev.runtime.dtype} on ${jev.runtime.device}`);

let agree = 0, n = 0, maxDp = 0, sumDp = 0;
for (let i = 0; i < records.length; i++) {
  const qs = records[i].questions.map((q) =>
    q.options.length === 2 && q.options[0] === "no" && q.options[1] === "yes"
      ? { type: "noul", instructions: q.instr }
      : { type: "choice", instructions: q.instr, options: q.options });
  const ans = await jev.decide(records[i].state, qs);
  ans.forEach((a, k) => {
    const p = a.type === "noul" ? [1 - a.probability, a.probability] : records[i].questions[k].options.map((o) => a.probabilities[o]);
    const want = ref[i].probs[k];
    const am = p.indexOf(Math.max(...p)), ar = want.indexOf(Math.max(...want));
    const dp = Math.max(...p.map((v, j) => Math.abs(v - want[j])));
    maxDp = Math.max(maxDp, dp); sumDp += dp; n++; if (am === ar) agree++;
    console.log(`rec${i + 1} q${k + 1} onnx ${p.map((v) => v.toFixed(3)).join(" ")} | f32 ${want.map((v) => v.toFixed(3)).join(" ")} | Δmax ${dp.toFixed(3)}${am !== ar ? "  ARGMAX DIFFERS" : ""}`);
  });
}
console.log(`\nONNX q4 vs torch f32: argmax ${agree}/${n}, mean|Δp| ${(sumDp / n).toFixed(3)}, max|Δp| ${maxDp.toFixed(3)}`);
await jev.dispose();
