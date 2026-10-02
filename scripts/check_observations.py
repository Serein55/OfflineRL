"""Replay an official demonstration state to verify image orientation and state layout."""
from pathlib import Path
import json,h5py,numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
suite=benchmark.get_benchmark_dict()['libero_spatial'](); task=suite.get_task(0)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
path=Path('data/raw/libero_spatial')/(task.name+'_demo.hdf5'); report={}
with h5py.File(path) as f:
    d=f['data']['demo_0']; env.reset(); obs=env.set_init_state(d['states'][0])
    for stored,rendered in [('agentview_rgb','agentview_image'),('eye_in_hand_rgb','robot0_eye_in_hand_image')]:
        ref=d['obs'][stored][0].astype(float); img=obs[rendered]
        scores={name:float(np.mean((ref-x)**2)) for name,x in [('identity',img),('vertical',img[::-1]),('horizontal',img[:,::-1]),('rotate180',img[::-1,::-1])]}
        assert min(scores,key=scores.get)=='identity',scores
        report[stored]=scores
    for stored,current in [('ee_pos',obs['robot0_eef_pos']),('ee_ori',quat2axisangle(obs['robot0_eef_quat'])),('gripper_states',obs['robot0_gripper_qpos'])]:
        error=float(np.max(np.abs(d['obs'][stored][0]-current)))
        report[stored+'_max_error']=error
        assert error<.05,(stored,error)
env.close()
Path('artifacts/observation_validation.json').write_text(json.dumps(report,indent=2))
print('PASS observation replay',json.dumps(report),flush=True)
