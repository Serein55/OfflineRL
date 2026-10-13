"""Keep official transforms/shuffle, add checked per-index advantage AFTER transforms."""
import numpy as np
from openpi.models import model
from openpi.training import data_loader as official
from arfm_jax.annotations import load_sidecar

class AnnotatedDataset:
    def __init__(self,raw,transformed,annotations):
        self.raw=raw;self.transformed=transformed;self.annotations=annotations
    def __len__(self):return len(self.raw)
    def __getitem__(self,index):
        raw=self.raw[index]
        for key in ('index','episode_index','frame_index','task_index'):
            if int(np.asarray(raw[key]).item())!=int(self.annotations[key][index]):raise ValueError(f'Annotation mismatch: {key} at {index}')
        # Apply exactly the official deterministic transforms once to this same sample.
        result=self.transformed(raw)
        result['advantage']=np.float32(self.annotations['advantage'][index])
        return result

class Loader:
    def __init__(self,config,loader):self.config=config;self.loader=loader
    def data_config(self):return self.config
    def __iter__(self):
        for batch in self.loader:
            yield model.Observation.from_dict(batch),batch['actions'],batch['advantage']

def create_loader(config,*,sidecar,dataset_root,sharding=None,shuffle=True,**kwargs):
    if kwargs:raise ValueError(f'Unsupported loader arguments: {kwargs}')
    dc=config.data.create(config.assets_dirs,config.model)
    if dc.repo_id not in ('physical-intelligence/libero','local/libero_expert2k'):raise ValueError('Unsupported dataset')
    raw=official.create_torch_dataset(dc,config.model.action_horizon,config.model)
    annotations,_=load_sidecar(sidecar,dataset_root,len(raw),dataset_id=dc.repo_id)
    # The official TransformedDataset owns the composed callable; use the same chain.
    transformed=official.transform_dataset(raw,dc)._transform
    dataset=AnnotatedDataset(raw,transformed,annotations)
    loader=official.TorchDataLoader(dataset,config.batch_size,sharding=sharding,shuffle=shuffle,
        num_workers=config.num_workers,seed=config.seed)
    return Loader(dc,loader)
