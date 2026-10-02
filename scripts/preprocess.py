"""Precompute raw HDF5 rewards, average RTG, task LOO and global normalization."""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
from arfm.reward import table8_reward

def process_file(path):
    import cv2
    cv2.setNumThreads(1)
    path=Path(path); dest=path.parents[2]/'processed'/path.parent.name
    dest.mkdir(parents=True,exist_ok=True)
    output=dest/(path.stem+'.npz')
    if output.exists():
        return str(output)
    rewards=[]; rtgs=[]; lengths=[]; keys=[]; sums=[]; sums2=[]; components=[]
    with h5py.File(path,'r') as f:
        demos=sorted(f['data'],key=lambda x:int(x.split('_')[-1]))
        if len(demos)!=50:
            raise ValueError(f'{path}: expected 50 demonstrations, got {len(demos)}')
        for demo in demos:
            d=f['data'][demo]; o=d['obs']; a=d['actions'][:].astype(np.float64)
            state=np.concatenate([o['ee_pos'][:],o['ee_ori'][:],o['gripper_states'][:]],axis=-1).astype(np.float64)
            joint=o['joint_states'][:].astype(np.float64)
            r,parts,goals=table8_reward(o['agentview_rgb'][:],o['eye_in_hand_rgb'][:],joint,a,True)
            rtg=np.cumsum(r[::-1])[::-1]/np.arange(len(r),0,-1)
            rewards.append(r); rtgs.append(rtg); lengths.append(len(r)); keys.append(demo)
            both=np.concatenate([state,a],axis=-1)
            sums.append(both.sum(0)); sums2.append((both**2).sum(0)); components.append(parts)
    G=np.concatenate(rtgs); adv=(G-G.mean())*len(G)/(len(G)-1)
    np.savez(output,reward=np.concatenate(rewards),rtg=G,advantage=adv.astype('float32'),
             components=np.concatenate(components),lengths=lengths,episodes=keys,
             sums=np.sum(sums,0),sums2=np.sum(sums2,0),source=str(path.resolve()))
    return str(output)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/raw'); p.add_argument('--workers',type=int,default=16)
    args=p.parse_args(); root=Path(args.root)
    files=sorted(root.glob('*/*.hdf5'))
    expected={'libero_goal','libero_spatial','libero_object','libero_10'}
    files=[f for f in files if f.parent.name in expected]
    if len(files)!=40 or any(sum(f.parent.name==s for f in files)!=10 for s in expected):
        raise ValueError(f'Expected 40 files, 10 per suite; found {len(files)}')
    outputs=[]
    with ProcessPoolExecutor(args.workers) as ex:
        for out in ex.map(process_file,files):
            outputs.append(out); print('processed',out,flush=True)
    stats=[np.load(f) for f in outputs]; count=sum(s['lengths'].sum() for s in stats)
    mean=sum(s['sums'] for s in stats)/count
    std=np.sqrt(np.maximum(sum(s['sums2'] for s in stats)/count-mean**2,1e-12))
    outdir=root.parent/'processed'
    (outdir/'manifest.json').write_text(json.dumps({'files':outputs,'samples':int(count),'episodes':2000,
        'state_mean':mean[:-7].tolist(),'state_std':std[:-7].tolist(),
        'action_mean':mean[-7:].tolist(),'action_std':std[-7:].tolist()},indent=2))
    print('DONE',count,'samples',flush=True)
if __name__=='__main__': main()
