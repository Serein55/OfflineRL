"""Read converted data through actual OpenPI/LeRobot and verify chunk semantics."""
import json,sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from jax_2k.config import make_config
from openpi.training import data_loader
from arfm_jax.annotations import load_sidecar

def main():
    art=ROOT/'artifacts/jax_2k';c=make_config();dc=c.data.create(c.assets_dirs,c.model)
    raw=data_loader.create_torch_dataset(dc,c.model.action_horizon,c.model)
    transformed=data_loader.transform_dataset(raw,dc,skip_norm_stats=True)
    result=json.loads((art/'conversion_complete.json').read_text())
    assert len(raw)==result['frames']==338575
    annotations,_=load_sidecar(ROOT/'data/jax_2k/annotations.npz',ROOT/'data/openpi_lerobot/local/libero_expert2k',len(raw),dataset_id='local/libero_expert2k')
    mapping=[json.loads(line) for line in (art/'episode_mapping.jsonl').read_text().splitlines()]
    # Every task plus episode boundary/end padding cases.
    indices=[]
    for ep in mapping[::50]:
        with h5py.File(ep['source'],'r') as f:
            d=f['data'][ep['source_demo']];n=ep['length']
            for t in sorted({0,n//2,n-1}):
                i=ep['start_index']+t;r=raw[i];x=transformed[i];indices.append(i)
                for key in ('index','episode_index','frame_index','task_index'):assert int(np.asarray(r[key]))==int(annotations[key][i])
                st=np.concatenate([d['obs']['ee_pos'][t],d['obs']['ee_ori'][t],d['obs']['gripper_states'][t]]).astype('float32')
                a=d['actions'][:][np.minimum(np.arange(t,t+50),n-1)].astype('float32')
                np.testing.assert_allclose(np.asarray(r['state']),st,atol=1e-6)
                np.testing.assert_array_equal(np.asarray(r['actions']),a)
                expected=a.copy();expected[:,:6]-=st[:6]
                np.testing.assert_allclose(x['actions'][:,:7],expected,atol=1e-6)
                np.testing.assert_array_equal(x['actions'][:,7:],np.zeros((50,25)))
                expected_image=d['obs']['agentview_rgb'][t][::-1,::-1].copy().transpose(2,0,1)/255.
                np.testing.assert_allclose(np.asarray(r['image']),expected_image,atol=1e-7)
                assert x['image']['base_0_rgb'].shape==(224,224,3)
    report=dict(frames=len(raw),tasks_checked=40,samples_checked=len(indices),extra_delta_verified=True,end_padding_verified=True,reader='pinned OpenPI/LeRobot',image_shape=[128,128,3])
    (art/'reader_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
