"""Explicitly authorized Vanilla resume from saved 15k, then full 30k evaluation."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import openpi_pipeline as pipeline

ART = ROOT / 'artifacts/jax_2k'
CP = ART / 'checkpoints/pi0_libero/vanilla_seed42'


def state(stage, **extra):
    value = dict(stage=stage, time=datetime.datetime.now().astimezone().isoformat(),
                 resume_step=15000, target_step=30000, eval_episodes=2000, **extra)
    temp = ART / 'vanilla_30k_state.tmp'
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(ART / 'vanilla_30k_state.json')
    print(json.dumps(value), flush=True)


def main():
    with (ART / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ART / 'vanilla_30k_state.json').exists():
            raise ValueError('Existing continuation: explicit recovery required')
        try:
            assert max(int(p.name) for p in CP.iterdir() if p.is_dir() and p.name.isdigit()) == 15000
            assert (CP / '15000/train_state').is_dir()
            summary = json.loads((ART / 'eval/expert2k_step_015000/summary.json').read_text())
            assert all(summary[s]['episodes'] == 500 for s in pipeline.SUITES)
            (ART / 'vanilla_seed42_config_before_resume.txt').write_text((ART / 'vanilla_seed42_config.txt').read_text())
            # Preserve the original stop request for audit; this entry point is separately authorized.
            state('training_15000_to_30000', authorization='User requested same-config Vanilla 30k + eval')
            with (ROOT / 'logs/jax_2k_vanilla_30000_train.log').open('w') as log:
                subprocess.run([str(pipeline.PY), 'scripts/jax_2k/train.py', '--steps', '30000',
                                '--experiment', 'vanilla_seed42', '--resume'], cwd=ROOT,
                               env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoint = CP / '30000'
            assert checkpoint.is_dir() and (CP / '15000').is_dir()
            state('evaluating_30000', checkpoint=str(checkpoint))
            pipeline.ART = ART
            pipeline.ENV.update(ARFM_JAX_DATASET='expert2k', OPENPI_EVAL_RESOLUTION='128')
            pipeline.evaluate(checkpoint, 'expert2k_step_030000')
            state('complete', checkpoint=str(checkpoint))
        except BaseException as exc:
            state('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
