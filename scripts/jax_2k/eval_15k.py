"""User-requested 15k evaluation only. Does not resume training."""
import sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import openpi_pipeline as pipeline
ART=ROOT/'artifacts/jax_2k'
def state(stage,**extra):
    report=dict(stage=stage,time=datetime.datetime.now().astimezone().isoformat(),auto_resume=False,**extra)
    (ART/'handoff_state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    pipeline.ART=ART;pipeline.ENV.update(ARFM_JAX_DATASET='expert2k',OPENPI_EVAL_RESOLUTION='128')
    try:
        state('expert2k_evaluating_15000_training_stopped')
        pipeline.evaluate(ART/'checkpoints/pi0_libero/vanilla_seed42/15000','expert2k_step_015000')
        state('expert2k_15000_eval_complete_training_stopped')
    except BaseException as e:state('eval_failed_training_stopped',error=repr(e));raise
