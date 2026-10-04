"""Thin wrapper around the frozen LeRobot PI0 implementation."""
import os
from pathlib import Path
import torch
from safetensors.torch import load_file
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from arfm.objective import weighted_loss
from arfm.action_transform import ExtraDeltaInference

class ARFMPolicy(ExtraDeltaInference,PI0Policy):
    method='arfm'
    time_sampler='uniform'
    fixed_alpha=.1
    arfm_lambda=5e-4
    def forward(self,batch,noise=None,time=None):
        if time is None and self.time_sampler=='uniform':
            time=torch.rand(len(batch['action']),device=batch['action'].device)
        _,details=super().forward(batch,noise=noise,time=time)
        per_sample=details['losses_after_rm_padding'].float().mean((1,2))
        return weighted_loss(per_sample,batch['advantage'],self.method,self.fixed_alpha,lam=self.arfm_lambda)

def build_policy(checkpoint,stats,method='arfm',time_sampler='uniform',training=True):
    config=PI0Config(input_features={
        'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,)),
        'observation.images.camera0':PolicyFeature(type=FeatureType.VISUAL,shape=(3,128,128)),
        'observation.images.camera1':PolicyFeature(type=FeatureType.VISUAL,shape=(3,128,128))},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))},
        freeze_vision_encoder=False,train_expert_only=False,train_state_proj=True,
        chunk_size=50,n_action_steps=50,device='cpu',push_to_hub=False)
    norm={}
    for name,key in [('observation.state','state'),('action','action')]:
        norm[name]={'mean':torch.tensor(stats[key+'_mean']),'std':torch.tensor(stats[key+'_std'])}
    policy=ARFMPolicy(config,dataset_stats=norm)
    policy.extra_delta_transform=stats.get('extra_delta_transform',False)
    raw=load_file(str(Path(checkpoint)/'model.safetensors'))
    # Check historical and transformed key schemas against this exact runtime.
    target=policy.model.state_dict()
    selected={}
    transformed=PI0Policy._transform_state_dict_keys(raw)
    for source in (raw,transformed):
        for key,value in source.items():
            key=key.removeprefix('model.')
            if key in target and target[key].shape==value.shape:
                selected[key]=value
    # Safetensors may omit the tied token embedding/head weight.
    for key in target:
        if key not in selected and key.endswith('lm_head.weight') and 'paligemma.' in key:
            candidates=[v for k,v in selected.items() if k.endswith('embed_tokens.weight') and 'paligemma.' in k and v.shape==target[key].shape]
            if candidates: selected[key]=candidates[0]
    policy.model.load_state_dict(selected,strict=True)
    policy.method=method; policy.time_sampler=time_sampler
    # FP32 master weights; bf16 autocast only for compute, avoiding tiny-update rounding.
    policy.float()
    if training:
        policy.model.requires_grad_(True)
        # MLP recomputation preserves the frozen forward semantics and reduces activation memory.
        from torch.utils.checkpoint import checkpoint as recompute
        for module in policy.model.modules():
            if module.__class__.__name__ in ('GemmaMLP','SiglipMLP'):
                original=module.forward
                def forward(*args,_original=original,**kwargs):
                    return recompute(_original,*args,use_reentrant=False,**kwargs)
                module.forward=forward
    return policy
