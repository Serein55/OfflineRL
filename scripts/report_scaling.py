"""Compare completed evaluations against the unchanged replan=5 vanilla baseline."""
import json
from pathlib import Path

root=Path('artifacts')
runs={'vanilla': 'vanilla_uniform_replan5_seed42',
      'ARFM raw': 'arfm_uniform_replan5_seed42',
      'ARFM task z-score': 'arfm_taskz_uniform_seed42',
      'RWR task z-score alpha=0.1': 'rwr_taskz_a01_uniform_seed42',
      'RWR task z-score alpha=0.5': 'rwr_taskz_a05_uniform_seed42'}
baseline=json.loads((root/runs['vanilla']/'results.json').read_text())['average']
rows={}
for name,directory in runs.items():
    path=root/directory/'results.json'
    if path.exists():
        results=json.loads(path.read_text())
        rows[name]={'results':results,'difference_pp':100*(results['average']-baseline)}
    else:
        rows[name]={'status':'pending'}
(root/'scaling_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
