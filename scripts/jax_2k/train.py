"""Train original 2k expert data from official pi0_base using OpenPI JAX."""
import argparse,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from jax_2k.config import make_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=30000);p.add_argument('--experiment',default='vanilla_seed42');p.add_argument('--resume',action='store_true');a=p.parse_args()
    import jax
    assert jax.device_count()==4
    c=make_config(a.steps,a.experiment,a.resume)
    assert c.batch_size==32 and c.ema_decay==.99 and not c.model.image_augmentation
    art=ROOT/'artifacts/jax_2k'
    assert all((art/name).exists() for name in ('conversion_complete.json','reader_verified.json','norm_verified.json'))
    if a.experiment=='vanilla_seed42':assert (art/'smoke_pass.json').exists()
    (art/f'{a.experiment}_config.txt').write_text(repr(c)+'\n')
    spec=importlib.util.spec_from_file_location('expert2k_official_train',ROOT/'vendor/openpi_official/scripts/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main(c)
if __name__=='__main__':main()
