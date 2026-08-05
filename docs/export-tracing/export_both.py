import torch, torch.nn as nn, warnings, collections, io, os
warnings.filterwarnings('ignore')
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)

B,S,H,DK,DV = 1,128,4,32,32     # small; structure is what matters

class Chunked(nn.Module):
    def forward(self,q,k,v,g,beta,state):
        o,st = torch_chunk_gated_delta_rule(q,k,v,g,beta,initial_state=state,output_final_state=True)
        return o,st
class Recurrent(nn.Module):
    def forward(self,q,k,v,g,beta,state):
        o,st = torch_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=state,output_final_state=True)
        return o,st

# layout per source: q/k/v [B,S,H,D]; beta/g [B,S,H]; state [B,H,DK,DV]
args = (torch.randn(B,S,H,DK), torch.randn(B,S,H,DK), torch.randn(B,S,H,DV),
        -torch.rand(B,S,H), torch.rand(B,S,H), torch.randn(B,H,DK,DV))

def census(path):
    import onnx
    m = onnx.load(path)
    ops = collections.Counter()
    def walk(g):
        for n in g.node:
            ops[n.op_type]+=1
            for a in n.attribute:
                if a.g and a.g.node: walk(a.g)
    walk(m.graph)
    return ops, sum(ops.values())

for name, mod in [('chunked', Chunked()), ('recurrent', Recurrent())]:
    for mode in ['dynamo','legacy']:
        p=f'/tmp/{name}_{mode}.onnx'
        try:
            if mode=='dynamo':
                ep = torch.onnx.export(mod, args, dynamic_shapes={
                        'q':{1:'s'},'k':{1:'s'},'v':{1:'s'},'g':{1:'s'},'beta':{1:'s'},'state':None},
                        dynamo=True, verbose=False)
                ep.save(p)
            else:
                torch.onnx.export(mod, args, p, opset_version=17,
                    input_names=['q','k','v','g','beta','state'], output_names=['o','st'],
                    dynamic_axes={n:{1:'s'} for n in ['q','k','v','g','beta']}, dynamo=False)
            ops,tot = census(p)
            loopish = {k:v for k,v in ops.items() if k in ('Scan','Loop','If')}
            print(f"{name:<10} {mode:<7} OK   total_nodes={tot:<6} control_flow={loopish or 'NONE (unrolled)'}  MatMul={ops['MatMul']}")
        except Exception as e:
            print(f"{name:<10} {mode:<7} FAIL {type(e).__name__}: {str(e)[:110]}")
