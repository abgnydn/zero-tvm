import torch, torch.nn as nn, warnings, onnx, collections
from onnx import helper as h, TensorProto as T
warnings.filterwarnings('ignore')
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
H_,DK,DV,C = 32,128,128,64                      # REAL Qwen3.5 GDN dims

class ChunkStep(nn.Module):
    def forward(self, state,q,k,v,g,beta):
        o,st = torch_chunk_gated_delta_rule(q,k,v,g,beta,chunk_size=C,
                                            initial_state=state,output_final_state=True)
        return st,o
args=(torch.randn(1,H_,DK,DV),torch.randn(1,C,H_,DK),torch.randn(1,C,H_,DK),
      torch.randn(1,C,H_,DV),-torch.rand(1,C,H_),torch.rand(1,C,H_))
torch.onnx.export(ChunkStep(),args,'/tmp/rbody.onnx',opset_version=17,dynamo=False,
    input_names=['state_in','q_in','k_in','v_in','g_in','beta_in'],
    output_names=['state_out','o_out'])
body=onnx.load('/tmp/rbody.onnx').graph; body.name='chunk_body'
ops=collections.Counter(n.op_type for n in body.node)
print(f"real chunk body: {len(body.node)} nodes, MatMul={ops['MatMul']}")
scan=h.make_node('Scan',['state0','qc','kc','vc','gc','bc'],['state_fin','oc'],
    body=body,num_scan_inputs=5,scan_input_axes=[1]*5,scan_output_axes=[1])
NC='nchunks'
ins=[h.make_tensor_value_info('state0',T.FLOAT,[1,H_,DK,DV]),
     h.make_tensor_value_info('qc',T.FLOAT,[1,NC,C,H_,DK]),
     h.make_tensor_value_info('kc',T.FLOAT,[1,NC,C,H_,DK]),
     h.make_tensor_value_info('vc',T.FLOAT,[1,NC,C,H_,DV]),
     h.make_tensor_value_info('gc',T.FLOAT,[1,NC,C,H_]),
     h.make_tensor_value_info('bc',T.FLOAT,[1,NC,C,H_])]
outs=[h.make_tensor_value_info('state_fin',T.FLOAT,[1,H_,DK,DV]),
      h.make_tensor_value_info('oc',T.FLOAT,[1,NC,C,H_,DV])]
m=h.make_model(h.make_graph([scan],'real_chunked_scan',ins,outs),opset_imports=[h.make_opsetid('',17)])
m.ir_version=9; onnx.checker.check_model(m)
onnx.save(m,'/Users/ahmetbarisgunaydin/.claude/jobs/371bd978/tmp/webgpu-test/real_chunked_scan.onnx')
import os; print("saved:", os.path.getsize('/Users/ahmetbarisgunaydin/.claude/jobs/371bd978/tmp/webgpu-test/real_chunked_scan.onnx')/1e6, "MB")
