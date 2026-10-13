"""The 2k expert control: same JAX recipe, isolated dataset/assets/checkpoints."""
import dataclasses
from pathlib import Path
from openpi_train import make_config as official_config
ROOT=Path(__file__).resolve().parents[2]
REPO_ID='local/libero_expert2k'

def make_config(steps=30000,experiment='vanilla_seed42',resume=False):
    c=official_config(steps,experiment,resume)
    return dataclasses.replace(c,data=dataclasses.replace(c.data,repo_id=REPO_ID),
        checkpoint_base_dir=str(ROOT/'artifacts/jax_2k/checkpoints'),
        assets_base_dir=str(ROOT/'artifacts/jax_2k/assets'))
