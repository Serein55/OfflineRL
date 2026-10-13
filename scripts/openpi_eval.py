"""Official LIBERO client protocol, with per-episode JSON and fail-fast errors."""
import argparse,collections,importlib.util,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_libero_eval',ROOT/'vendor/openpi_official/examples/libero/main.py');official=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=official;spec.loader.exec_module(official)
from libero.libero import benchmark
from openpi_client import image_tools,websocket_client_policy
p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--rollouts',type=int,default=50);p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)))
p.add_argument('--failure-video-dir');p.add_argument('--paired-rng',action='store_true');a=p.parse_args()
if a.failure_video_dir:
    import imageio.v2 as imageio
    video_dir=Path(a.failure_video_dir);video_dir.mkdir(parents=True,exist_ok=True)
def video_frame(obs):
    return np.concatenate([image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key][::-1,::-1]),224,224)) for key in ('agentview_image','robot0_eye_in_hand_image')],axis=1)
np.random.seed(7);suite=benchmark.get_benchmark_dict()[a.suite]()
client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',a.port)
if a.paired_rng and not client.get_server_metadata().get('paired_episode_rng'):
    raise ValueError('Paired RNG requires a server launched with --paired-rng')
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
# A restarted inference server restarts its RNG: do not silently resume partial official evaluations.
if output.exists():raise ValueError('Choose a fresh output, or explicitly archive the incomplete evaluation before rerun')
limit={'libero_goal':300,'libero_spatial':220,'libero_object':280,'libero_10':520}[a.suite]
for task_id in a.task_ids:
    task=suite.get_task(task_id);initial=suite.get_task_init_states(task_id)
    resolution=int(os.environ.get('OPENPI_EVAL_RESOLUTION','256'))
    env,description=official._get_libero_env(task,resolution,7)
    try:
        for ep in range(a.rollouts):
            env.reset();obs=env.set_init_state(initial[ep]);plan=collections.deque();success=False;frames=[]
            policy_seed=7+task_id*1000+ep
            for _ in range(10):obs,_,_,_=env.step(official.LIBERO_DUMMY_ACTION)
            for step in range(limit):
                if a.failure_video_dir:frames.append(video_frame(obs))
                if not plan:
                    img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224))
                    wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
                    request={'observation/image':img,'observation/wrist_image':wrist,'observation/state':np.concatenate((obs['robot0_eef_pos'],official._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),'prompt':description}
                    if a.paired_rng and step==0:request['_diagnostic_episode_seed']=policy_seed
                    chunk=client.infer(request)['actions'];assert chunk.shape==(50,7) and np.isfinite(chunk).all();plan.extend(chunk[:5])
                obs,_,done,_=env.step(plan.popleft().tolist())
                if done:success=True;break
            row=dict(suite=a.suite,task_id=task_id,episode=ep,seed=7,success=success,steps=step+1,replan_steps=5,render_backend=os.environ.get('MUJOCO_GL'),render_resolution=resolution,protocol='openpi_expert2k_128' if resolution==128 else 'official_openpi_c015073')
            if a.paired_rng:row.update(protocol='spatial_cross_resolution_v1',policy_episode_seed=policy_seed)
            if a.failure_video_dir and not success:
                frames.append(video_frame(obs));video=video_dir/f'task_{task_id:02d}_episode_{ep:03d}_failure.mp4'
                imageio.mimwrite(str(video),frames,fps=20,codec='libx264',quality=8)
                row['failure_video']=str(video.resolve())
            with output.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    finally:env.close()
