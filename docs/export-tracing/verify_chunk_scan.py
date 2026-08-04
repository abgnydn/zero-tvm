import torch, numpy as np, onnxruntime as ort, warnings
warnings.filterwarnings('ignore')
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
H_,DK,DV,C = 4,32,32,64
s = ort.InferenceSession('/tmp/chunked_scan.onnx', providers=['CPUExecutionProvider'])
torch.manual_seed(0)
print("chunk-level Scan vs PyTorch (max relative error):")
for S in (64,128,192,256,512):
    q=torch.randn(1,S,H_,DK); k=torch.randn(1,S,H_,DK); v=torch.randn(1,S,H_,DV)
    g=-torch.rand(1,S,H_); beta=torch.rand(1,S,H_); st=torch.randn(1,H_,DK,DV)
    with torch.no_grad():
        ref,_ = torch_chunk_gated_delta_rule(q,k,v,g,beta,chunk_size=C,
                                             initial_state=st,output_final_state=True)
    nc = S//C
    r5 = lambda x,d: x.reshape(1,nc,C,H_,d).numpy()
    out = s.run(None, {'state0':st.numpy(),
                       'qc':r5(q,DK),'kc':r5(k,DK),'vc':r5(v,DV),
                       'gc':g.reshape(1,nc,C,H_).numpy(),'bc':beta.reshape(1,nc,C,H_).numpy()})
    got = out[1].reshape(1,S,H_,DV)
    ref = ref.float().numpy()
    err = np.abs(ref-got).max()/(np.abs(ref).max()+1e-9)
    print(f"   S={S:<5} chunks={nc:<3} rel_err={err:.2e}  {'OK' if err<1e-4 else 'WRONG'}")
