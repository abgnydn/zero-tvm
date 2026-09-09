import torch, numpy as np, onnxruntime as ort, warnings
warnings.filterwarnings('ignore')
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
B,H,DK,DV=1,4,32,32
torch.manual_seed(0)

def check(path, fn, S):
    q=torch.randn(B,S,H,DK); k=torch.randn(B,S,H,DK); v=torch.randn(B,S,H,DV)
    g=-torch.rand(B,S,H); beta=torch.rand(B,S,H); st=torch.randn(B,H,DK,DV)
    with torch.no_grad():
        ref,_=fn(q,k,v,g,beta,initial_state=st,output_final_state=True)
    s=ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    feed=dict(zip([i.name for i in s.get_inputs()],
                  [x.numpy() for x in (q,k,v,g,beta,st)]))
    got=s.run(None,feed)[0]
    r=ref.float().numpy()
    err=np.abs(r-got).max()/ (np.abs(r).max()+1e-9)
    return err

for path,fn,label in [('/tmp/chunked_dynamo.onnx',torch_chunk_gated_delta_rule,'chunked (dynamo)'),
                      ('/tmp/chunked_legacy.onnx',torch_chunk_gated_delta_rule,'chunked (legacy)'),
                      ('/tmp/recurrent_legacy.onnx',torch_recurrent_gated_delta_rule,'recurrent (legacy)')]:
    out=[]
    for S in (128,192,252):
        try: out.append(f"S={S}: rel_err {check(path,fn,S):.2e}")
        except Exception as e: out.append(f"S={S}: FAIL {str(e)[:45]}")
    print(f"{label:<20} " + " | ".join(out))
