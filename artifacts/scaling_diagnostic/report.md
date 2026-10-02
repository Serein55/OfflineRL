# Advantage scaling experiment — 2026-10-03

Only advantage scaling changes: population z-score over all chunk starts in each of 40 concrete tasks, epsilon=1e-8. Original processed files are unchanged. Adaptive solver, lambda=0.0005, task sampler, images, action transforms, optimizer, time sampler and initialization remain unchanged.

## Offline gate (no training updates)

256 identical global batches of 16 per checkpoint, seed 2026, real per-sample flow-matching losses with training augmentation and uniform time. The same batches, noise and times are replayed at base pi0 and vanilla 40k. Fixed-alpha results are identical across checkpoints because they do not depend on loss.

| Checkpoint | Weighting | Mean ESS | Mean max weight | Mean min weight | ESS < 4 |
|---|---|---:|---:|---:|---:|
| base_pi0 | raw_adaptive | 16.0000 | 0.062513 | 0.062495 | 0.00% |
| base_pi0 | zscore_adaptive | 15.9840 | 0.066572 | 0.061119 | 0.00% |
| base_pi0 | zscore_rwr_0.1 | 15.7949 | 0.080826 | 0.056800 | 0.00% |
| base_pi0 | zscore_rwr_0.5 | 11.4966 | 0.220567 | 0.034774 | 12.11% |
| base_pi0 | zscore_rwr_1 | 7.2473 | 0.398639 | 0.017324 | 32.42% |
| vanilla_40k | raw_adaptive | 16.0000 | 0.062703 | 0.062424 | 0.00% |
| vanilla_40k | zscore_adaptive | 14.2979 | 0.124857 | 0.046905 | 0.00% |
| vanilla_40k | zscore_rwr_0.1 | 15.7949 | 0.080826 | 0.056800 | 0.00% |
| vanilla_40k | zscore_rwr_0.5 | 11.4966 | 0.220567 | 0.034774 | 12.11% |
| vanilla_40k | zscore_rwr_1 | 7.2473 | 0.398639 | 0.017324 | 32.42% |

The adaptive result at vanilla 40k passes the nontrivial-weight gate: mean ESS 14.30 (slightly above the suggested 8–14 band), 26.56% of batches within 8–14, no ESS < 4, minimum ESS 5.05. Initial pi0 still has ESS 15.98; weighting is weak at the start because the solver responds to loss variance. This experiment does not force ESS to a target or retune lambda. Training ESS must be tracked.

RWR alpha=0.1 is a mild control; alpha=0.5 gives mean ESS 11.50 but has a 12.11% ESS<4 tail. Alpha=1 is excluded from training because 32.42% of batches have ESS<4. No clipping is added, to keep the control interpretable.

## Queued experiments

1. ARFM, task z-score, adaptive alpha, 40k steps.
2. RWR, task z-score, fixed alpha=0.1, 40k steps.
3. RWR, task z-score, fixed alpha=0.5, 40k steps.

Each starts fresh from the original base pi0 with seed42, global batch16, uniform time, full parameter updates, original LR schedule. Runs use GPUs0–3 sequentially. Each is followed by 2000 LIBERO rollouts (40 tasks ×50 fixed initial states), action chunk50, replan5. Vanilla is not retrained or modified. Its measured baseline is 1606/2000 = 80.3%; previous raw-advantage ARFM was 1626/2000 = 81.3%. New results remain pending until evaluations complete. A single-seed improvement is not sufficient to establish statistical reliability.

Run `bash scripts/run_scaling_experiments.sh` after the diagnostic. State: `artifacts/scaling_pipeline_state.txt`; logs: `logs/train_arfm_taskz.log`, `logs/train_rwr_taskz_a01.log`, `logs/train_rwr_taskz_a05.log`. `scripts/report_scaling.py` writes `artifacts/scaling_comparison.json` after each completed evaluation.

Validation: 22 tests passed, including per-task normalization, raw-data preservation, constant advantages, and fixed-alpha weighting. Shell syntax and Python compilation passed.
