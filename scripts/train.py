import argparse,json,os,time,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader
from arfm.data import LiberoChunks,TaskBalancedSampler
from arfm.policy import build_policy
from arfm.objective import learning_rate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['vanilla','rwr','arfm'],default='arfm')
    p.add_argument('--time-sampler',choices=['uniform','beta'],default='uniform')
    p.add_argument('--steps',type=int,default=40000); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint',default='checkpoints/pi0'); p.add_argument('--manifest',default='data/processed/manifest.json')
    p.add_argument('--output',required=True); p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--fixed-alpha',type=float,default=.1)
    p.add_argument('--advantage-normalization',choices=['none','task_zscore'],default='none')
    p.add_argument('--advantage-eps',type=float,default=1e-8)
    p.add_argument('--arfm-lambda',type=float,default=5e-4)
    p.add_argument('--smoke-inference',action='store_true'); p.add_argument('--resume'); p.add_argument('--no-augment',action='store_true')
    args=p.parse_args()
    rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    if world not in (1,4): raise ValueError('Supported runs: 1 GPU diagnostic or GPUs 0–3')
    torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl')
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Output already contains training metrics; use a fresh directory or --resume')
    dataset=LiberoChunks(args.manifest,augment=not args.no_augment,
                         advantage_normalization=args.advantage_normalization,advantage_eps=args.advantage_eps)
    policy=build_policy(args.checkpoint,dataset.manifest,args.method,args.time_sampler).cuda(local)
    policy.fixed_alpha=args.fixed_alpha
    policy.arfm_lambda=args.arfm_lambda
    print(f'rank {rank}: loaded {sum(p.numel() for p in policy.parameters()):,} parameters',flush=True)
    start=0
    optim_cls=ZeroRedundancyOptimizer if world>1 else torch.optim.AdamW
    kw={'optimizer_class':torch.optim.AdamW} if world>1 else {}
    optimizer=optim_cls(policy.parameters(),lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,**kw)
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ckpt['stats']!=dataset.manifest:
            raise ValueError('Resume requires the same data manifest, action transform and normalization statistics')
        defaults={'advantage_normalization':'none','advantage_eps':1e-8,'arfm_lambda':5e-4}
        for key in ('method','time_sampler','seed','fixed_alpha','advantage_normalization','advantage_eps','arfm_lambda'):
            if ckpt['args'].get(key,defaults.get(key))!=getattr(args,key):
                raise ValueError(f'Resume changes {key}; start a separate experiment instead')
        policy.load_state_dict(ckpt['policy']); start=ckpt['step']
        if ckpt.get('world_size',world)!=world: raise ValueError('Resume requires original world size')
        if ckpt.get('optimizer_sharded'):
            shard=torch.load(Path(args.resume).resolve().parent/f'optimizer_rank{rank}.pt',map_location='cpu',weights_only=False)
            (optimizer.optim if world>1 else optimizer).load_state_dict(shard)
            del shard
        else:
            optimizer.load_state_dict(ckpt['optimizer'])
        rng=ckpt['rng'][rank]; torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'],local)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); del ckpt
    model=DDP(policy,device_ids=[local],find_unused_parameters=True,gradient_as_bucket_view=True) if world>1 else policy
    model.train()
    sampler=TaskBalancedSampler(dataset,args.steps,rank=rank,world=world,seed=args.seed,start=start)
    loader=DataLoader(dataset,batch_size=16//world,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    if rank==0:
        (out/'config.json').write_text(json.dumps(vars(args),indent=2))
        (out/'advantage_statistics.json').write_text(json.dumps(dataset.advantage_stats,indent=2))
        log=(out/'metrics.jsonl').open('a')
    for step,batch in enumerate(loader,start):
        begin=time.monotonic()
        batch={k:v.cuda(local,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        lr=learning_rate(step)
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,metrics=model(batch)
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        metrics.update(step=step+1,lr=lr,grad_norm=float(norm),seconds=time.monotonic()-begin,
                       peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        if rank==0:
            log.write(json.dumps(metrics)+'\n'); log.flush()
            if step<10 or (step+1)%50==0: print(json.dumps(metrics),flush=True)
        if (step+1)%args.save_every==0 or step+1==args.steps:
            checkpoint_dir=out/'checkpoints'/f'step_{step+1:06d}'
            checkpoint_dir.mkdir(parents=True,exist_ok=True)
            shard_temp=checkpoint_dir/f'optimizer_rank{rank}.pt.tmp'
            torch.save((optimizer.optim if world>1 else optimizer).state_dict(),shard_temp)
            shard_temp.replace(checkpoint_dir/f'optimizer_rank{rank}.pt')
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(local),'python':random.getstate(),'numpy':np.random.get_state()}
            states=[None]*world
            if world>1: dist.all_gather_object(states,rng)
            else: states=[rng]
            if world>1: dist.barrier()
            if rank==0:
                temporary=checkpoint_dir/'model.pt.tmp'
                torch.save({'policy':policy.state_dict(),'optimizer_sharded':True,'world_size':world,'step':step+1,
                            'rng':states,'args':vars(args),'stats':dataset.manifest},temporary)
                temporary.replace(checkpoint_dir/'model.pt')
                link=out/'latest.pt.tmp'
                if link.is_symlink(): link.unlink()
                link.symlink_to(Path('checkpoints')/checkpoint_dir.name/'model.pt')
                link.replace(out/'latest.pt')
                print(f'saved step {step+1}',flush=True)
                import shutil
                for old in sorted((out/'checkpoints').glob('step_*'))[:-2]:
                    shutil.rmtree(old)
            if world>1: dist.barrier()
    if args.smoke_inference and rank==0:
        policy.eval(); policy.reset()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            action=policy.select_action(batch)
        assert action.shape==(16//world,7) and torch.isfinite(action).all()
        print('PASS trained-policy inference',tuple(action.shape),flush=True)
    if world>1:
        dist.barrier()
        dist.destroy_process_group()
if __name__=='__main__': main()
