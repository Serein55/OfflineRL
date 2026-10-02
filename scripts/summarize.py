import json,sys
from pathlib import Path
root=Path(sys.argv[1]); summaries={}
for suite in ['libero_goal','libero_spatial','libero_object','libero_10']:
    rows=[json.loads(x) for x in (root/f'eval_{suite}.jsonl').read_text().splitlines()]
    expected={(t,e) for t in range(10) for e in range(50)}
    keys=[(r['task_id'],r['episode']) for r in rows]
    if set(keys)!=expected or len(keys)!=500: raise ValueError(f'Incomplete or duplicated evaluation: {suite}')
    summaries[suite]={'successes':sum(r['success'] for r in rows),'rollouts':500,
                      'success_rate':sum(r['success'] for r in rows)/500}
summaries['average']=sum(s['success_rate'] for s in summaries.values())/4
(root/'results.json').write_text(json.dumps(summaries,indent=2)); print(json.dumps(summaries,indent=2))

import subprocess
subprocess.run([sys.executable,str(Path(__file__).resolve().parent/"report.py")],check=True)
