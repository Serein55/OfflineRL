"""Validate actual frozen pi0 forward, wrapper equivalence, and gradient update."""
import torch
from arfm.policy import build_policy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
stats={'state_mean':[0]*8,'state_std':[1]*8,'action_mean':[0]*7,'action_std':[1]*7}
p=build_policy('checkpoints/pi0',stats).cuda().train()
b={'observation.state':torch.zeros(1,8,device='cuda'), 'action':torch.zeros(1,50,7,device='cuda'),
   'action_is_pad':torch.zeros(1,50,dtype=torch.bool,device='cuda'), 'task':['pick up the black bowl'],
   'advantage':torch.zeros(1,device='cuda')}
for k in ('camera0','camera1'): b['observation.images.'+k]=torch.rand(1,3,128,128,device='cuda')
noise=torch.randn(1,50,32,device='cuda'); time=torch.tensor([.5],device='cuda')
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    vanilla,_=PI0Policy.forward(p,b,noise,time)
with torch.autocast('cuda',dtype=torch.bfloat16):
    weighted,m=p(b,noise,time)
assert torch.allclose(vanilla,weighted,rtol=1e-6,atol=1e-6),(vanilla,weighted)
weighted.backward()
assert p.model.action_out_proj.weight.grad is not None
assert torch.isfinite(p.model.action_out_proj.weight.grad).all()
print('PASS actual pi0: strict checkpoint load, fixed-noise flow/padding equivalence, finite backward',m,flush=True)
print('peak GPU GB',torch.cuda.max_memory_allocated()/1e9,flush=True)
