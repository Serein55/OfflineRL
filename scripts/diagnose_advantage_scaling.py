"""Real π0 forward passes only: identical batches/losses for raw and z-scored weights.

No optimizer, backward, parameter update, or model checkpoint write. Keep per-sample
loss/advantage pairs so weight sweeps can be reproduced offline without more GPU work.
"""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from arfm.data import LiberoChunks,TaskBalancedSampler
from arfm.policy import build_policy
from arfm.objective import gather_detached,solve_alpha
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),min=float(a.min()),p05=float(np.quantile(a,.05)),
                median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def weight_sweep(losses,raw,scaled):
    definitions=[('raw_adaptive',raw,None,5e-4),('zscore_adaptive',scaled,None,5e-4)]
    definitions += [(f'zscore_rwr_{alpha:g}',scaled,alpha,5e-4) for alpha in (.1,.5,1.)]
    result={}
    for name,advantages,fixed,lam in definitions:
        metrics={k:[] for k in ('alpha','ess','weight_max','weight_min','entropy')}
        for loss,adv in zip(losses,advantages):
            l=torch.as_tensor(loss,dtype=torch.float64); a=torch.as_tensor(adv,dtype=torch.float64)
            alpha=solve_alpha(a,l,lam=lam) if fixed is None else fixed
            weights=torch.softmax(alpha*a,0)
            for key,value in dict(alpha=alpha,ess=float(1/weights.square().sum()),
                                  weight_max=float(weights.max()),weight_min=float(weights.min()),
                                  entropy=float(-(weights*weights.clamp_min(1e-30).log()).sum())).items():
                metrics[key].append(value)
        result[name]={k:describe(v) for k,v in metrics.items()}
        ess=np.asarray(metrics['ess'])
        result[name]['fraction_ess_8_to_14']=float(np.mean((ess>=8)&(ess<=14)))
        result[name]['fraction_ess_below_4']=float(np.mean(ess<4))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',type=int,default=256)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--output',default='artifacts/scaling_diagnostic')
    args=p.parse_args(); rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    if 16%world: raise ValueError('world size must divide 16')
    torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    dataset=LiberoChunks('data/processed/manifest.json',augment=True,advantage_normalization='task_zscore')
    if rank==0:
        (output/'task_statistics.json').write_text(json.dumps(dataset.advantage_stats,indent=2))
    policy=build_policy('checkpoints/pi0',dataset.manifest,training=False).cuda(local)
    policy.train()  # Match training behavior, but no gradients or updates.
    all_results={}
    for stage in ['base_pi0','vanilla_40k']:
        if stage=='vanilla_40k':
            state=torch.load('artifacts/vanilla_uniform_seed42/latest.pt',map_location='cpu',weights_only=False)
            policy.load_state_dict(state['policy'],strict=True);del state
        random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
        loader=DataLoader(dataset,batch_size=16//world,
            sampler=TaskBalancedSampler(dataset,args.batches,rank=rank,world=world,seed=args.seed),
            num_workers=4,pin_memory=True)
        captured={k:[] for k in ('losses','raw','zscore','task_id')}
        for step,batch in enumerate(loader):
            batch={k:v.cuda(local,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,detail=PI0Policy.forward(policy,batch,time=torch.rand(len(batch['action']),device='cuda'))
                per_sample=detail['losses_after_rm_padding'].float().mean((1,2))
            values={'losses':per_sample,'raw':batch['raw_advantage'],'zscore':batch['advantage'],'task_id':batch['task_id']}
            for key,value in values.items():
                gathered=gather_detached(value)
                if rank==0: captured[key].append(gathered.cpu().numpy())
            if rank==0 and (step+1)%32==0: print(stage,'forward batches',step+1,'/',args.batches,flush=True)
        assert all(p.grad is None for p in policy.parameters())
        if rank==0:
            arrays={k:np.stack(v) for k,v in captured.items()}
            assert np.isfinite(arrays['losses']).all()
            np.savez(output/f'{stage}_batches.npz',**arrays)
            all_results[stage]=weight_sweep(arrays['losses'],arrays['raw'],arrays['zscore'])
            (output/'summary.json').write_text(json.dumps(all_results,indent=2))
            print(stage,json.dumps(all_results[stage]),flush=True)
        if world>1: dist.barrier()
    if rank==0:
        (output/'protocol.json').write_text(json.dumps(dict(seed=args.seed,batches_per_stage=args.batches,
            global_batch=16,augmentation=True,time_sampler='uniform',updates=0,
            advantage_grouping='all chunk starts within each concrete task',eps=1e-8,
            reference_checkpoints=['checkpoints/pi0','artifacts/vanilla_uniform_seed42/checkpoints/step_040000/model.pt']),indent=2))
        print('DONE: offline-only real-loss weight diagnostic',flush=True)
    if world>1: dist.destroy_process_group()
if __name__=='__main__':main()
