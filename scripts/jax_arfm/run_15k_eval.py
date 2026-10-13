"""Original expert2k ARFM from pi0 base, 15k then full matched Vanilla evaluation."""
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

EXPERIMENT = 'arfm_expert2k_noaug_seed42'
ART = ROOT / 'artifacts/jax_arfm'
RUN = ART / 'runs' / EXPERIMENT
CHECKPOINT = ART / 'checkpoints/pi0_libero' / EXPERIMENT / '15000'


def state(stage, **extra):
    value = dict(stage=stage, time=datetime.datetime.now().astimezone().isoformat(),
                 experiment=EXPERIMENT, train_steps=15000, eval_episodes=2000, **extra)
    temp = RUN / 'pipeline_state.tmp'
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(RUN / 'pipeline_state.json')
    print(json.dumps(value), flush=True)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'protocol.json').exists() or (RUN / 'pipeline_state.json').exists():
            raise ValueError('Existing run: inspect state before explicit recovery; no automatic overwrite')
        try:
            state('training')
            with (ROOT / 'logs/jax_arfm_expert2k_15000_train.log').open('w') as log:
                subprocess.run([str(pipeline.PY), 'scripts/jax_arfm/train.py',
                                '--dataset', 'expert2k', '--annotations', 'data/jax_2k/annotations.npz',
                                '--experiment', EXPERIMENT, '--method', 'arfm', '--steps', '15000',
                                '--lam', '0.0005'], cwd=ROOT, env=os.environ.copy(),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            assert CHECKPOINT.is_dir(), 'Completed 15k checkpoint missing'
            state('evaluating', checkpoint=str(CHECKPOINT))
            pipeline.ART = ART
            pipeline.ENV.update(ARFM_JAX_DATASET='expert2k', OPENPI_EVAL_RESOLUTION='128')
            pipeline.evaluate(CHECKPOINT, EXPERIMENT + '_step_015000')
            state('complete', checkpoint=str(CHECKPOINT))
        except BaseException as exc:
            state('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
