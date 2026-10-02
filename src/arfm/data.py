import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ColorJitter, functional as TF

class LiberoChunks(Dataset):
    def __init__(self, manifest, augment=True):
        self.manifest=json.loads(Path(manifest).read_text())
        self.tasks=[]; self.offsets=[0]; self.handles={}; self.augment=augment
        self.jitter=ColorJitter(brightness=(.8,1.2),contrast=(.8,1.2),saturation=(.5,1.5),hue=(-.05,.05))
        for filename in self.manifest['files']:
            with np.load(filename) as f:
                task={k:f[k].copy() for k in ('lengths','episodes','advantage','source')}
            task['starts']=np.r_[0,np.cumsum(task['lengths'])]
            name=Path(str(task['source'])).stem.removesuffix('_demo')
            # BDDL scene prefixes are not natural-language commands.
            if '_SCENE' in name:
                name=name.split('_SCENE',1)[1].split('_',1)[1]
            task['language']=name.replace('_',' ')
            self.tasks.append(task); self.offsets.append(self.offsets[-1]+int(task['lengths'].sum()))
    def __len__(self): return self.offsets[-1]
    def __getitem__(self,index):
        task_id=int(np.searchsorted(self.offsets,index,side='right')-1)
        task=self.tasks[task_id]; local=index-self.offsets[task_id]
        ep=int(np.searchsorted(task['starts'],local,side='right')-1); t=int(local-task['starts'][ep])
        path=str(task['source'])
        if path not in self.handles: self.handles[path]=h5py.File(path,'r')
        d=self.handles[path]['data'][str(task['episodes'][ep])]; obs=d['obs']; T=int(task['lengths'][ep])
        actions=d['actions'][t:min(t+50,T)].astype('float32')
        pad=np.arange(50)>=len(actions)
        actions=np.pad(actions,((0,50-len(actions)),(0,0)),mode='edge')
        state=np.concatenate([obs['ee_pos'][t],obs['ee_ori'][t],obs['gripper_states'][t]]).astype('float32')
        result={'action':torch.from_numpy(actions),'action_is_pad':torch.from_numpy(pad),
                'observation.state':torch.from_numpy(state),'task':task['language'],
                'advantage':torch.tensor(task['advantage'][local]),'task_id':task_id}
        for name,key in [('agentview_rgb','camera0'),('eye_in_hand_rgb','camera1')]:
            # LIBERO demonstration convention is preserved. Eval preserves the same raw orientation (verified by state replay).
            image=torch.from_numpy(obs[name][t].copy()).permute(2,0,1).float()/255
            if self.augment:
                image=self.jitter(image)
                image=TF.adjust_sharpness(image,float(torch.empty(()).uniform_(.5,1.5)))
            result['observation.images.'+key]=image
        return result

class TaskBalancedSampler(Sampler):
    """Generate identical global batches on all ranks, then take disjoint rank slices."""
    def __init__(self,dataset,steps,batch=16,rank=0,world=1,seed=42,start=0):
        if batch%world: raise ValueError('global batch must divide world size')
        self.dataset=dataset; self.steps=steps; self.batch=batch; self.rank=rank; self.world=world; self.seed=seed; self.start=start
    def __len__(self): return (self.steps-self.start)*self.batch//self.world
    def __iter__(self):
        n=self.batch//self.world
        for step in range(self.start,self.steps):
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
            tasks=rng.integers(len(self.dataset.tasks),size=self.batch)
            indices=[int(rng.integers(self.dataset.offsets[t],self.dataset.offsets[t+1])) for t in tasks]
            yield from indices[self.rank*n:(self.rank+1)*n]
