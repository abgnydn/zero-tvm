import onnx, numpy as np
from onnx import helper as h, TensorProto as TP

D = 8
# --- Scan body: a miniature gated delta rule (Mul/ReduceSum/Exp/Sub/Add) ---
body = h.make_graph(
    [h.make_node("Exp",       ["g_in"],              ["g_e"]),
     h.make_node("Mul",       ["state_in", "g_e"],   ["decayed"]),
     h.make_node("Mul",       ["k_in", "v_in"],      ["kv"]),
     h.make_node("ReduceSum", ["kv"],                ["kv_s"], keepdims=1),
     h.make_node("Sub",       ["decayed", "kv_s"],   ["upd"]),
     h.make_node("Add",       ["upd", "kv"],         ["state_out"]),
     h.make_node("Mul",       ["state_out", "q_in"], ["y_out"])],
    "delta_rule_body",
    [h.make_tensor_value_info("state_in", TP.FLOAT, [1, D]),
     h.make_tensor_value_info("q_in",     TP.FLOAT, [1, D]),
     h.make_tensor_value_info("k_in",     TP.FLOAT, [1, D]),
     h.make_tensor_value_info("v_in",     TP.FLOAT, [1, D]),
     h.make_tensor_value_info("g_in",     TP.FLOAT, [1, D])],
    [h.make_tensor_value_info("state_out", TP.FLOAT, [1, D]),
     h.make_tensor_value_info("y_out",     TP.FLOAT, [1, D])],
)

scan = h.make_node(
    "Scan", ["state0", "q", "k", "v", "g"], ["state_fin", "y"],
    body=body, num_scan_inputs=4,
    scan_input_axes=[1, 1, 1, 1], scan_output_axes=[1],   # walk the sequence axis
)
# a MatMul outside the Scan, as a control: this one the WebGPU EP definitely supports
mm = h.make_node("MatMul", ["y", "W"], ["out"])

S = "seq"
g = h.make_graph(
    [scan, mm], "toy_gdn",
    [h.make_tensor_value_info("state0", TP.FLOAT, [1, D]),
     h.make_tensor_value_info("q", TP.FLOAT, [1, S, D]),
     h.make_tensor_value_info("k", TP.FLOAT, [1, S, D]),
     h.make_tensor_value_info("v", TP.FLOAT, [1, S, D]),
     h.make_tensor_value_info("g", TP.FLOAT, [1, S, D])],
    [h.make_tensor_value_info("out", TP.FLOAT, [1, S, D])],
    [h.make_tensor("W", TP.FLOAT, [D, D], np.eye(D, dtype=np.float32).ravel().tolist())],
)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, "toy_scan.onnx")
print("wrote toy_scan.onnx — 1 Scan (body: Exp/Mul/ReduceSum/Sub/Add) + 1 MatMul control")
