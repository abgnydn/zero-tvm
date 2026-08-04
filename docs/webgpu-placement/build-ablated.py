"""
Build an ablated copy of the decoder with every `Scan` replaced by shape-preserving
Identity nodes. Numerically meaningless, but structurally valid and ~free to run —
so (full - ablated) is the recurrence's real in-situ cost, measured in place
instead of extrapolated from an isolated model.

  Scan(state_f, qf, kf, vf, bf, g_t) -> out_rec_f32, scan_y_out
    out_rec_f32  [1,32,128,128]  <- Identity(state_f)   same shape
    scan_y_out   [1,32,S,128]    <- Identity(vf)        same shape (scan axis 2)
"""
import onnx, pathlib, collections
D = pathlib.Path('.weights-local/onnx/Qwen3.5-4B-ONNX/onnx')
m = onnx.load(str(D/'decoder_model_merged_q4f16.onnx'), load_external_data=False)

replaced = 0
def ablate(g):
    global replaced
    new_nodes = []
    for n in g.node:
        if n.op_type == 'Scan':
            state_in, v_in = n.input[0], n.input[3]      # state_f, vf
            st_out, y_out  = n.output[0], n.output[1]
            new_nodes.append(onnx.helper.make_node('Identity', [state_in], [st_out],
                                                   name=(n.name or 'scan') + '_abl_state'))
            new_nodes.append(onnx.helper.make_node('Identity', [v_in], [y_out],
                                                   name=(n.name or 'scan') + '_abl_y'))
            replaced += 1
            continue
        for a in n.attribute:
            if a.g and a.g.node:
                ablate(a.g)
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)

ablate(m.graph)
ops = collections.Counter()
def walk(g):
    for n in g.node:
        ops[n.op_type] += 1
        for a in n.attribute:
            if a.g and a.g.node: walk(a.g)
walk(m.graph)

out = D/'decoder_ablated_q4f16.onnx'
onnx.save(m, str(out))
print(f"replaced {replaced} Scan nodes with Identity pairs")
print(f"ablated graph: Scan={ops['Scan']}  GroupQueryAttention={ops['GroupQueryAttention']}  Identity={ops['Identity']}")
print(f"wrote {out.name} ({out.stat().st_size/1e6:.2f} MB) next to the shared .onnx_data")
