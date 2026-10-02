"""Fixed LIBERO init states, 50 rollouts/task, 10 denoising steps, 50-action chunks."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from arfm.policy import build_policy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--rollouts',type=int,default=50)
    p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--suites',nargs='+',default=['libero_goal','libero_spatial','libero_object','libero_10'])
    args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    policy=build_policy('checkpoints/pi0',ckpt['stats'],training=False)
    policy.load_state_dict(ckpt['policy'],strict=True); del ckpt
    policy.cuda().eval(); results=[]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): results=[json.loads(line) for line in out.read_text().splitlines()]
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
                        action=policy.select_action(batch)[0].float().cpu().numpy()
                    obs,_,done,_=env.step(action.tolist())
                    if done: success=True; break
                row={'suite':suite_name,'task_id':task_id,'episode':episode,'seed':args.seed,'success':success,'steps':t+1}
                results.append(row)
                with out.open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            env.close()
    summary={s:float(np.mean([r['success'] for r in results if r['suite']==s])) for s in args.suites}
    summary['average']=float(np.mean(list(summary.values())))
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__': main()
