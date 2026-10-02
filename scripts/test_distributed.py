"""Check four-rank gradients equal one global ARFM objective."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from arfm.objective import weighted_loss,solve_alpha
local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); dist.init_process_group('nccl')
model=torch.nn.Linear(1,1,bias=False).cuda(); model.weight.data.fill_(.3)
wrapped=DDP(model,device_ids=[local])
x=torch.arange(1,17,device='cuda',dtype=torch.float32)/16
adv=x-.5
per=wrapped(x[local*4:(local+1)*4,None]).flatten().square()
loss,_=weighted_loss(per,adv[local*4:(local+1)*4]); loss.backward()
a=solve_alpha(adv,(x*.3).square()); expected=(torch.softmax(a*adv,0)*2*.3*x.square()).sum()
assert torch.allclose(model.weight.grad.flatten()[0],expected,atol=1e-6)
if local==0: print('PASS four-GPU global loss/gradient equivalence',flush=True)
dist.destroy_process_group()
