import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
assert torch.cuda.is_available() and torch.cuda.device_count()==4, 'source scripts/env.sh and expose exactly GPU 0–3'
for i in range(4):
    print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory)
for path in ['checkpoints/pi0/download.json','data/raw/download.json']:
    assert json.loads((root/path).read_text())['status']=='complete'
m=json.loads((root/'data/processed/manifest.json').read_text()); assert len(m['files'])==40
count=0; episodes=0
for path in m['files']:
    with np.load(root/path) as f:
        assert len(f['lengths'])==50
        assert sum(f['lengths'])==len(f['advantage'])
        assert np.isfinite(f['advantage']).all() and np.isfinite(f['components']).all()
        assert abs(f['advantage'].mean())<1e-7
        count+=len(f['advantage']); episodes+=len(f['lengths'])
assert count==m['samples'] and episodes==2000
print('PASS preflight',count,'chunks',episodes,'episodes')
