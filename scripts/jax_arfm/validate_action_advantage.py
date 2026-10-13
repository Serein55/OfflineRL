"""Paired same-state action-prefix interventions, followed by a fixed policy.

A diagnostic, not a benchmark success-rate evaluation or training data sidecar.
Branches copy complete MuJoCo data and controller state. A duplicate-prefix
probe verifies restoration dynamics, rather than restoring only qpos/qvel.
"""
import argparse
import collections
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from arfm_jax.action_quality import event_rewards
spec=importlib.util.spec_from_file_location('official_libero', ROOT/'vendor/openpi_official/examples/libero/main.py')
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)
from libero.libero import benchmark
from openpi_client import image_tools,websocket_client_policy


def request(obs,prompt,seed):
    def img(key):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key][::-1,::-1]),224,224))
    return {'observation/image':img('agentview_image'),'observation/wrist_image':img('robot0_eye_in_hand_image'),
            'observation/state':np.concatenate((obs['robot0_eef_pos'],official._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos'])),
            'prompt':prompt,'_diagnostic_episode_seed':int(seed)}


def measure(env):
    goals=env.env.parsed_problem['goal_state']
    pred=next(g for g in goals if g[0]=='on')
    target=env.env.objects_dict[pred[1]];goal=env.env.objects_dict[pred[2]]
    obj=env.sim.data.body_xpos[env.sim.model.body_name2id(target.root_body)].copy()
    dst=env.sim.data.body_xpos[env.sim.model.body_name2id(goal.root_body)].copy()
    eef=env.sim.data.site_xpos[env.robots[0].eef_site_id].copy()
    contact=bool(env.env._check_grasp(env.robots[0].gripper,target))
    return dict(contact=contact,goal=bool(env.check_success()),eef=eef.tolist(),object=obj.tolist(),destination=dst.tolist(),
                reach_distance=float(np.linalg.norm(eef-obj)),goal_xy_distance=float(np.linalg.norm(obj[:2]-dst[:2])),height=float(obj[2]-dst[2]))



def snapshot(env,obs):
    robot=env.robots[0]
    return dict(data=copy.copy(env.sim.data._data),obs=copy.deepcopy(obs),
                controller=copy.deepcopy({k:v for k,v in robot.controller.__dict__.items() if k!='sim'}),
                robot=copy.deepcopy({k:v for k,v in robot.__dict__.items() if k.startswith('recent_') or k=='torques'}),
                gripper=copy.deepcopy(robot.gripper.current_action),
                env=copy.deepcopy({k:getattr(env.env,k) for k in ['timestep','cur_time','done','_obs_cache','_observables']}),
                rng=np.random.get_state())


def restore(env,saved):
    env.sim.data._data=copy.copy(saved['data'])
    robot=env.robots[0]
    for k,v in saved['controller'].items():setattr(robot.controller,k,copy.deepcopy(v))
    for k,v in saved['robot'].items():setattr(robot,k,copy.deepcopy(v))
    robot.gripper.current_action=copy.deepcopy(saved['gripper'])
    for k,v in saved['env'].items():setattr(env.env,k,copy.deepcopy(v))
    np.random.set_state(saved['rng'])
    return copy.deepcopy(saved['obs'])


def check_restore(env,saved,chunk):
    results=[]
    for _ in range(2):
        restore(env,saved)
        for action in chunk:env.step(action.tolist())
        results.append(env.get_sim_state().copy())
    error=float(np.max(np.abs(results[0]-results[1])))
    if error>1e-10:raise RuntimeError(f'Duplicate action restoration error {error}')
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8125);p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--variants',nargs='+',choices=['policy_a','policy_b','hold_translation','open_gripper'],default=['policy_a','policy_b','hold_translation','open_gripper'])
    p.add_argument('--replicas',type=int,default=1)
    p.add_argument('--episode-start',type=int,default=0);p.add_argument('--continuation-replica',type=int,default=0)
    p.add_argument('--tasks',type=int,nargs='+',default=list(range(10)));p.add_argument('--anchors',type=int,nargs='+',default=[40,90])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/advantage_counterfactual/exact_snapshot');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(checkpoint='jax_2k/vanilla_seed42/30000',tasks=args.tasks,episodes=args.episodes,episode_start=args.episode_start,continuation_replica=args.continuation_replica,replicas=args.replicas,anchors=args.anchors,
                  candidates=args.variants,prefix=5,episode_horizon=220,replan=5,
                  seed=7,resolution=128,paired_noise=True,restore='Copy complete MjData, controller, gripper, observation buffers, environment clock and RNG; duplicate prefix dynamics error <= 1e-10',
                  development_tasks=[0,4,5],heldout_tasks=[1,2,3,6,7,8,9],notes=['Natural candidates and artificial negative controls reported separately.',
                  'One continuation seed per state: exploratory, not a calibrated success probability.',
                  'H50 includes closed-loop continuation and cannot validate the 5-step prefix by itself.',
                  'Branch results must not be treated as independent episodes in uncertainty estimates.'])
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text());old.setdefault('replicas',1)
        assert old==protocol
        if args.replicas>1 and any(args.output.glob('replica_*/task*.json')):
            raise ValueError('Multi-seed snapshots cannot resume across processes: use a fresh directory')
    else:protocol_path.write_text(json.dumps(protocol,indent=2))
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    assert client.get_server_metadata().get('paired_episode_rng')
    suite=benchmark.get_benchmark_dict()['libero_spatial']();started=time.time()
    for tid in args.tasks:
        task=suite.get_task(tid);initial=suite.get_task_init_states(tid)
        env,prompt=official._get_libero_env(task,128,7)
        def reset(ep):
            env.seed(7);np.random.seed(7);env.reset();obs=env.set_init_state(initial[ep])
            for _ in range(10):obs,_,_,_=env.step(official.LIBERO_DUMMY_ACTION)
            return obs
        def infer(obs,seed):
            action=np.asarray(client.infer(request(obs,prompt,seed))['actions']);assert action.shape==(50,7) and np.isfinite(action).all()
            return action[:5].copy()
        try:
            for ep in range(args.episode_start,args.episode_start+args.episodes):
                key=7+tid*10000+ep*1000;obs=reset(ep);history=[];features=[measure(env)];anchors={};plan=collections.deque()
                for step in range(max(args.anchors)+1):
                    if step in args.anchors:
                        anchors[step]=(np.array(history),list(features),env.get_sim_state().copy(),infer(obs,key+step),snapshot(env,obs))
                    if step==max(args.anchors) or features[-1]['goal']:break
                    if not plan:plan.extend(infer(obs,key+step))
                    action=plan.popleft();obs,_,done,_=env.step(action.tolist());history.append(action.copy());features.append(measure(env))
                    if done:break
                for anchor,(prefix,past,state,base_chunk,saved) in anchors.items():
                    if args.replicas==1:
                        existing=[(args.output/f'task{tid:02d}_ep{ep:02d}_t{anchor:03d}_{v}.json').exists() for v in protocol['candidates']]
                        if all(existing):continue
                        if any(existing):raise ValueError('Partial state group cannot resume across processes; quarantine and rerun whole group')
                    restore_error=check_restore(env,saved,base_chunk)
                    for replica in range(args.continuation_replica,args.continuation_replica+args.replicas):
                        replica_output=args.output if args.replicas==1 else args.output/f'replica_{replica}'
                        replica_output.mkdir(parents=True,exist_ok=True)
                        for variant in protocol['candidates']:
                            path=replica_output/f'task{tid:02d}_ep{ep:02d}_t{anchor:03d}_{variant}.json'
                            if path.exists():continue
                            obs=restore(env,saved)
                            error=float(np.max(np.abs(env.get_sim_state()-state)))
                            if error>=1e-7:raise RuntimeError(f'Nonidentical branch state {tid}/{ep}/{anchor}: {error}')
                            chunk=base_chunk.copy()
                            if variant=='policy_b':chunk=infer(obs,key+anchor+500000)
                            elif variant=='hold_translation':chunk[:,:3]=0
                            elif variant=='open_gripper':chunk[:,-1]=-1
                            actual=[];branch=[];success=False;plan=collections.deque(chunk)
                            for step in range(anchor,220):
                                if not plan:plan.extend(infer(obs,key+step+1000000+replica*10000000))
                                action=plan.popleft();obs,_,done,_=env.step(action.tolist());actual.append(action.copy());branch.append(measure(env))
                                if done:success=True;break
                            events=event_rewards(past+branch)[len(past):]
                            row=dict(task=tid,episode=ep,anchor=anchor,variant=variant,state_error=error,duplicate_prefix_error=restore_error,continuation_replica=replica,success=success,steps=len(branch),
                                     event_h5=float(events[:5].sum()),event_h50=float(events[:50].sum()),before=past[-1],after5=branch[min(4,len(branch)-1)],
                                     features=branch,actions=np.asarray(actual).tolist(),candidate_chunk=chunk.tolist(),prefix_actions=prefix.tolist())
                            path.write_text(json.dumps(row));print(json.dumps({k:row[k] for k in ('task','episode','anchor','variant','success','steps','event_h5','event_h50','state_error')}),flush=True)
                    (args.output/'status.json').write_text(json.dumps(dict(task=tid,episode=ep,anchor=anchor,elapsed_seconds=time.time()-started)))
        finally:env.close()
    (args.output/'complete.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started)))

if __name__=='__main__':main()
