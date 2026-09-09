"""
Extracts the Qwen3.5-4B export's OWN gated-delta-rule `Scan` into a standalone
~2.5 KB model, so its cost can be timed in isolation on WebGPU.

The Scan body is pure elementwise + reduce arithmetic with no weights, so this
needs only the 1.4 MB graph file — not the 2.5 GB of external weights.
"""
import onnx
from onnx import helper as h, TensorProto as TP
from huggingface_hub import hf_hub_download

src = hf_hub_download("onnx-community/Qwen3.5-4B-ONNX", "onnx/decoder_model_merged_q4f16.onnx")
m = onnx.load(src, load_external_data=False)

def find_scan(g):
    for n in g.node:
        if n.op_type == "Scan":
            return n
        for a in n.attribute:
            if a.g and a.g.node:
                r = find_scan(a.g)
                if r is not None:
                    return r

scan = find_scan(m.graph)
body = [a.g for a in scan.attribute if a.g][0]

# Shapes are implied by the body's own op chain: state [1,H,DK,DV]; the scanned
# inputs carry the sequence on axis 2, so the body sees one position at a time.
H, DK, DV, S, F = 32, 128, 128, "seq", TP.FLOAT
ins = [h.make_tensor_value_info("state0", F, [1, H, DK, DV]),
       h.make_tensor_value_info("q",    F, [1, H, S, DK]),
       h.make_tensor_value_info("k",    F, [1, H, S, DK]),
       h.make_tensor_value_info("v",    F, [1, H, S, DV]),
       h.make_tensor_value_info("beta", F, [1, H, S]),
       h.make_tensor_value_info("g",    F, [1, H, S])]
outs = [h.make_tensor_value_info("state_fin", F, [1, H, DK, DV]),
        h.make_tensor_value_info("y",         F, [1, H, S, DV])]

node = h.make_node("Scan", ["state0", "q", "k", "v", "beta", "g"], ["state_fin", "y"],
                   body=body, num_scan_inputs=5,
                   scan_input_axes=[2, 2, 2, 2, 2], scan_output_axes=[2])
out = h.make_model(h.make_graph([node], "real_gdn_scan", ins, outs),
                   opset_imports=list(m.opset_import))
out.ir_version = 9
onnx.checker.check_model(out)
onnx.save(out, "real_scan.onnx")
print(f"wrote real_scan.onnx — the export's own Scan body ({len(body.node)} nodes), one layer")
