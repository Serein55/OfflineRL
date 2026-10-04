"""Fixed LIBERO init states and 50-action predictions with configurable replanning."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch

def replan_steps(value):
    value=int(value)
    if not 1 <= value <= 50:
        raise argparse.ArgumentTypeError('replan steps must be between 1 and 50')
    return value

def select_replanned_action(policy,batch,step,interval):
    # Keep the model's 50-step horizon. Discard the unused queued actions and
    # infer a fresh chunk from the current observation every `interval` steps.
    if step % interval == 0:
        policy.reset()
    return policy.select_action(batch)

def load_existing_results(path,interval,seed):
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for row in rows:
        # Earlier evaluator versions always executed 50 steps per prediction.
        if row.get('replan_steps',50)!=interval or row['seed']!=seed:
            raise ValueError('Existing results use a different replan interval or seed; choose a new --output path')
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--rollouts',type=int,default=50)
    p.add_argument('--replan-steps',type=replan_steps,default=50,
                   help='Execute this many actions before predicting again (1–50; default: 50). The predicted chunk stays 50.')
    p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--suites',nargs='+',default=['libero_goal','libero_spatial','libero_object','libero_10'])
    args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    results=load_existing_results(out,args.replan_steps,args.seed)
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from arfm.policy import build_policy
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    extra_delta=ckpt['stats'].get('extra_delta_transform',False)
    if any(r.get('extra_delta_transform',False)!=extra_delta for r in results):
        raise ValueError('Existing evaluation uses a different action transform; choose a new output')
    policy=build_policy('checkpoints/pi0',ckpt['stats'],training=False)
    policy.load_state_dict(ckpt['policy'],strict=True); del ckpt
    policy.cuda().eval()
    if policy.config.chunk_size!=50 or policy.config.n_action_steps!=50:
        raise ValueError('Expected unchanged 50-step pi0 prediction horizon')
    completed={(r['suite'],r['task_id'],r['episode']) for r in results}
    limits={'libero_goal':280,'libero_spatial':220,'libero_object':280,'libero_10':520}
    for suite_name in args.suites:
        suite=benchmark.get_benchmark_dict()[suite_name]()
        for task_id in args.task_ids:
            task=suite.get_task(task_id); states=suite.get_task_init_states(task_id)
            if args.rollouts>len(states): raise ValueError('rollouts exceeds available fixed init states')
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
            for episode in range(args.rollouts):
                if (suite_name,task_id,episode) in completed: continue
                torch.manual_seed(args.seed+1000*task_id+episode)
                env.seed(args.seed+episode); env.reset(); obs=env.set_init_state(states[episode]); policy.reset()
                for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
                success=False
                for t in range(limits[suite_name]):
                    state=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype('float32')
                    batch={'observation.state':torch.from_numpy(state)[None].cuda(),'task':[task.language]}
                    for name,key in [('agentview_image','camera0'),('robot0_eye_in_hand_image','camera1')]:
                        rgb=obs[name].copy()
                        batch['observation.images.'+key]=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        action=select_replanned_action(policy,batch,t,args.replan_steps)[0].float().cpu().numpy()
                    obs,_,done,_=env.step(action.tolist())
                    if done: success=True; break
                row={'suite':suite_name,'task_id':task_id,'episode':episode,'seed':args.seed,'success':success,'steps':t+1,
                     'replan_steps':args.replan_steps,'extra_delta_transform':extra_delta}
                results.append(row)
                with out.open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            env.close()
    summary={s:float(np.mean([r['success'] for r in results if r['suite']==s])) for s in args.suites}
    summary['average']=float(np.mean(list(summary.values())))
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__': main()
