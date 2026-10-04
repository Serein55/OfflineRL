"""Recompute state and transformed action moments over all 50-step training chunks."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from arfm.action_transform import extra_delta

def action_chunks(actions,states,horizon=50):
    indices=np.minimum(np.arange(len(actions))[:,None]+np.arange(horizon),len(actions)-1)
    return extra_delta(actions[indices],states)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/processed/manifest.json')
    p.add_argument('--output',default='data/processed/manifest_extra_delta.json')
    args=p.parse_args()
    if Path(args.output).resolve()==Path(args.manifest).resolve():
        raise ValueError('Use a separate manifest; preserve raw baseline statistics')
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest.get('extra_delta_transform',False): raise ValueError('Expected raw manifest')
    s1=np.zeros(8);s2=np.zeros(8);a1=np.zeros(7);a2=np.zeros(7);ns=na=episodes=0
    for filename in manifest['files']:
        with np.load(filename) as meta:
            source=str(meta['source']);keys=meta['episodes'].copy()
        with h5py.File(source,'r') as f:
            for key in keys:
                d=f['data'][str(key)];obs=d['obs']
                # Match the float32 dataset transform, accumulate moments in float64.
                state=np.concatenate([obs['ee_pos'][:],obs['ee_ori'][:],obs['gripper_states'][:]],-1).astype('float32')
                actions=d['actions'][:].astype('float32')
                transformed=action_chunks(actions,state).reshape(-1,7).astype('float64')
                state=state.astype('float64')
                s1+=state.sum(0);s2+=(state**2).sum(0);ns+=len(state)
                a1+=transformed.sum(0);a2+=(transformed**2).sum(0);na+=len(transformed);episodes+=1
        print('processed',filename,flush=True)
    assert ns==manifest['samples'] and episodes==manifest['episodes'] and na==50*ns
    def moments(total,square,n):
        mean=total/n;std=np.sqrt(np.maximum(square/n-mean**2,1e-12))
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        return mean.tolist(),std.tolist()
    manifest['state_mean'],manifest['state_std']=moments(s1,s2,ns)
    manifest['action_mean'],manifest['action_std']=moments(a1,a2,na)
    manifest.update(extra_delta_transform=True,normalization_protocol={
        'state_samples':ns,'action_samples':na,'horizon':50,
        'actions':'all chunk starts and all 50 offsets; repeat-last padding included',
        'transform':'actions[..., :6] -= raw_state_at_chunk_start[:6]; gripper unchanged',
        'accumulation':'float64 population mean/std after float32 transform',
        'source_manifest':args.manifest})
    Path(args.output).write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',args.output,manifest['normalization_protocol'],flush=True)
if __name__=='__main__':main()
