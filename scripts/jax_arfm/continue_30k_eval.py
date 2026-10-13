"""Wait for the 15k evaluation to release GPUs, resume to 30k, then evaluate."""
import datetime
import fcntl
import json
import os
import subprocess
import time

from run_15k_eval import ART, CHECKPOINT, EXPERIMENT, ROOT, RUN, pipeline


def state(stage, **extra):
    value = dict(stage=stage, time=datetime.datetime.now().astimezone().isoformat(),
                 experiment=EXPERIMENT, resume_step=15000, target_step=30000,
                 eval_episodes=2000, **extra)
    temp = RUN / 'continuation_state.tmp'
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(RUN / 'continuation_state.json')
    print(json.dumps(value), flush=True)


def validate_eval():
    directory = ART / 'eval' / (EXPERIMENT + '_step_015000')
    assert json.loads((RUN / 'pipeline_state.json').read_text())['stage'] == 'complete', '15k pipeline did not complete'
    summary = json.loads((directory / 'summary.json').read_text())
    for suite in pipeline.SUITES:
        rows = [json.loads(line) for line in (directory / f'{suite}.jsonl').read_text().splitlines()]
        assert len(rows) == 500 and summary[suite]['episodes'] == 500
        assert {(r['task_id'], r['episode']) for r in rows} == {(t, e) for t in range(10) for e in range(50)}
        assert sum(r['success'] for r in rows) == summary[suite]['successes']
    assert CHECKPOINT.is_dir() and (CHECKPOINT / 'train_state').is_dir()
    steps = [int(p.name) for p in CHECKPOINT.parent.iterdir() if p.is_dir() and p.name.isdigit()]
    assert max(steps) == 15000, 'Unexpected latest checkpoint; inspect before resuming'


def main():
    with (RUN / 'continuation.lock').open('w') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'continuation_state.json').exists():
            raise ValueError('Existing continuation: explicit recovery required')
        try:
            state('waiting_for_15k_eval')
            # The original pipeline releases this lock only after evaluate() cleans up its workers.
            with (RUN / 'pipeline.lock').open('a') as previous_lock:
                while True:
                    try:
                        fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(15)
                validate_eval()
                (RUN / 'config_15000.txt').write_text((RUN / 'config.txt').read_text())
                state('training_15000_to_30000')
                with (ROOT / 'logs/jax_arfm_expert2k_30000_train.log').open('w') as log:
                    subprocess.run([str(pipeline.PY), 'scripts/jax_arfm/train.py',
                                    '--dataset', 'expert2k', '--annotations', 'data/jax_2k/annotations.npz',
                                    '--experiment', EXPERIMENT, '--method', 'arfm', '--steps', '30000',
                                    '--lam', '0.0005', '--resume'], cwd=ROOT, env=os.environ.copy(),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                checkpoint = CHECKPOINT.parent / '30000'
                assert checkpoint.is_dir()
                state('evaluating_30000', checkpoint=str(checkpoint))
                pipeline.ART = ART
                pipeline.ENV.update(ARFM_JAX_DATASET='expert2k', OPENPI_EVAL_RESOLUTION='128')
                pipeline.evaluate(checkpoint, EXPERIMENT + '_step_030000')
                state('complete', checkpoint=str(checkpoint))
        except BaseException as exc:
            state('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
