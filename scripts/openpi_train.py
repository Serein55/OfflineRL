"""Resource/path overrides only; official pi0_libero model, data and optimizer recipe."""
import argparse,dataclasses,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from openpi.training import config,weight_loaders

def make_config(steps=30000,experiment='vanilla_seed42',resume=False):
    c=config.get_config('pi0_libero')
    return dataclasses.replace(c,model=dataclasses.replace(c.model,image_augmentation=False),exp_name=experiment,num_train_steps=steps,fsdp_devices=4,
        checkpoint_base_dir=str(ROOT/'artifacts/jax_2k/checkpoints'),
        assets_base_dir=str(ROOT/'artifacts/jax_2k/assets'),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT/'checkpoints/openpi_pi0_base/params')),
        save_interval=1000,keep_period=15000,log_interval=10,wandb_enabled=False,resume=resume)
