"""Generate plots and a result table from actual logged measurements only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; out=root/'artifacts/report'; out.mkdir(exist_ok=True)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
results={}
any_data=False
for method in ['vanilla','arfm','rwr']:
    run=root/f'artifacts/{method}_uniform_seed42'
    metrics=run/'metrics.jsonl'
    if metrics.exists():
        rows={}
        for line in metrics.read_text().splitlines():
            try:
                r=json.loads(line); rows[r['step']]=r
            except json.JSONDecodeError: pass  # A currently-appending final line is not a measurement yet.
        data=[rows[k] for k in sorted(rows)]
        if data:
            any_data=True
            for ax,key in zip(axes,['fm_mean','alpha','ess']):
                x=np.array([r['step'] for r in data]); y=np.array([r[key] for r in data])
                n=min(100,len(y)); ax.plot(x[n-1:],np.convolve(y,np.ones(n)/n,mode='valid'),label=method)
                ax.set_ylabel(key); ax.grid(alpha=.2)
    result=run/'results.json'
    if result.exists():
        results[method]=json.loads(result.read_text())
    else:
        results[method]={'status':'pending'}
axes[-1].set_xlabel('Optimizer step'); fig.suptitle('Training diagnostics (up to 100-step moving mean)')
if any_data: axes[0].legend()
fig.tight_layout(); fig.savefig(out/'training.png',dpi=160); plt.close(fig)
files=list((root/'data/processed').glob('*/*.npz'))
if files:
    adv=np.concatenate([np.load(f)['advantage'] for f in files])
    fig,ax=plt.subplots(figsize=(8,4)); ax.hist(adv,bins=100); ax.set_xlabel('Task-centered LOO advantage'); ax.set_ylabel('Chunk count')
    fig.tight_layout(); fig.savefig(out/'advantage.png',dpi=160); plt.close(fig)
(out/'results.json').write_text(json.dumps({'protocol':{'time_sampler':'uniform','seed':42,'rollouts_per_task':50,'tasks':40,'replan_steps':50},'results':results},indent=2)+'\n')
