import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
state=root/'artifacts/pipeline_state.txt'
print(state.read_text().strip() if state.exists() else 'Pipeline not started')
for method in ('vanilla','arfm','rwr'):
    out=root/f'artifacts/{method}_uniform_seed42'; log=out/'metrics.jsonl'
    if log.exists():
        with log.open('rb') as f:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-4096)); lines=f.read().decode().splitlines()
        row=json.loads(lines[-1]); print(method,'step',row['step'],'/ 40000','loss',round(row['loss'],5),'alpha',round(row['alpha'],5),'ESS',round(row['ess'],3))
    result=out/'results.json'
    if result.exists(): print(result.read_text())
