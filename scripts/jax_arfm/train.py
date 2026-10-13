"""Independent JAX ARFM entry point; does not patch the running Vanilla process."""
import argparse,dataclasses,functools,importlib.util,json,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from arfm_jax.data import create_loader
from arfm_jax.training import train_step
from arfm_jax.annotations import load_sidecar

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset',choices=['expert2k'],default='expert2k');p.add_argument('--experiment',required=True);p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--method',choices=['arfm','rwr','vanilla'],default='arfm');p.add_argument('--fixed-alpha',type=float,default=.1)
    p.add_argument('--lam',type=float,default=5e-4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.steps<=0 or a.lam<0 or a.fixed_alpha<0:raise ValueError('Invalid training parameters')
    import jax
    if jax.device_count()!=4:raise ValueError('This launcher requires 4 devices, global batch32')
    from jax_2k.config import make_config as selected_config
    dataset_id='local/libero_expert2k'
    root=Path(os.environ['HF_LEROBOT_HOME'])/dataset_id
    n=json.loads((root/'meta/info.json').read_text())['total_frames']
    _,metadata=load_sidecar(a.annotations,root,n,dataset_id=dataset_id)
    c=dataclasses.replace(selected_config(a.steps,a.experiment,a.resume),checkpoint_base_dir=str(ROOT/'artifacts/jax_arfm/checkpoints'))
    run_dir=ROOT/'artifacts/jax_arfm/runs'/a.experiment
    protocol=dict(dataset_id=dataset_id,method=a.method,lam=a.lam,fixed_alpha=a.fixed_alpha,annotations_sha256=__import__('hashlib').sha256(a.annotations.read_bytes()).hexdigest(),annotation_metadata=metadata,batch_size=c.batch_size,image_augmentation=c.model.image_augmentation,ema_decay=c.ema_decay)
    run_dir.mkdir(parents=True,exist_ok=True);record=run_dir/'protocol.json'
    if record.exists():
        if not a.resume or json.loads(record.read_text())!=protocol:raise ValueError('Existing run or incompatible resume protocol')
    record.write_text(json.dumps(protocol,indent=2)+'\n');(run_dir/'config.txt').write_text(repr(c)+'\n')
    spec=importlib.util.spec_from_file_location('arfm_official_train',ROOT/'vendor/openpi_official/scripts/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # These replacements affect only this newly imported module in THIS process.
    module.train_step=functools.partial(train_step,method=a.method,fixed_alpha=a.fixed_alpha,lam=a.lam)
    module._data_loader=types.SimpleNamespace(create_data_loader=functools.partial(create_loader,sidecar=a.annotations,dataset_root=root))
    module.main(c)
if __name__=='__main__':main()
