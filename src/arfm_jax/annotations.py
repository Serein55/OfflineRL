"""Index-aligned offline reward/advantage sidecars. Never reuse old HDF5 indices."""
import hashlib
import json
from pathlib import Path
import numpy as np

REVISION='a4336d589d589045d1c56423ffdf3b88a0e19b1f'

def metadata_hash(root):
    h=hashlib.sha256()
    for name in ('info.json','episodes.jsonl','tasks.jsonl'):
        h.update(name.encode());h.update((Path(root)/'meta'/name).read_bytes())
    return h.hexdigest()

def prepare(reward,episode_index,frame_index,task_index,normalization='none',eps=1e-8):
    reward=np.asarray(reward,dtype=np.float64)
    episode_index,frame_index,task_index=map(np.asarray,(episode_index,frame_index,task_index))
    if reward.ndim!=1 or not reward.size or not np.isfinite(reward).all():raise ValueError('Invalid rewards')
    if any(x.shape!=reward.shape for x in (episode_index,frame_index,task_index)):raise ValueError('Shape mismatch')
    if normalization not in ('none','task_zscore') or eps<=0:raise ValueError('Invalid normalization')
    rtg=np.empty_like(reward)
    for ep in np.unique(episode_index):
        rows=np.flatnonzero(episode_index==ep)
        if not np.array_equal(frame_index[rows],np.arange(len(rows))):raise ValueError('Episode frames must be complete and ordered')
        if len(np.unique(task_index[rows]))!=1:raise ValueError('Episode spans tasks')
        rtg[rows]=np.cumsum(reward[rows][::-1])[::-1]/np.arange(len(rows),0,-1)
    advantage=np.empty_like(rtg)
    for task in np.unique(task_index):
        mask=task_index==task;x=rtg[mask];n=len(x)
        if n<2:raise ValueError('LOO requires at least two samples per task')
        advantage[mask]=(x-x.mean())*n/(n-1)
    raw=advantage.copy()
    if normalization=='task_zscore':
        for task in np.unique(task_index):
            mask=task_index==task;x=advantage[mask]
            advantage[mask]=(x-x.mean())/(x.std()+eps)
    return dict(reward=reward.astype('float32'),rtg=rtg.astype('float32'),raw_advantage=raw.astype('float32'),advantage=advantage.astype('float32'))

def load_sidecar(path,root,length,dataset_id='physical-intelligence/libero'):
    with np.load(path,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    meta=json.loads(str(data.pop('metadata')))
    revisions={'physical-intelligence/libero':REVISION,'local/libero_expert2k':'expert2k-preserved-v1'}
    if dataset_id not in revisions or meta.get('dataset_id', 'physical-intelligence/libero')!=dataset_id or meta['dataset_revision']!=revisions[dataset_id] or meta['metadata_sha256']!=metadata_hash(root):raise ValueError('Sidecar dataset identity mismatch')
    for key in ('index','episode_index','frame_index','task_index','reward','rtg','advantage'):
        if data[key].shape!=(length,) or not np.isfinite(data[key]).all():raise ValueError(f'Invalid {key}')
    if not np.array_equal(data['index'],np.arange(length)):raise ValueError('Sidecar global indices must be exact and ordered')
    return data,meta
