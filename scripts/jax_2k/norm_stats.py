"""Official norm-stat algorithm on extra-delta-transformed 2k data, CPU only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'vendor/openpi_official/scripts'))
from jax_2k.config import make_config
import compute_norm_stats

def main():
    assert (ROOT/'artifacts/jax_2k/conversion_complete.json').exists()
    c=make_config();compute_norm_stats._config.get_config=lambda _:c
    compute_norm_stats.main('pi0_libero')
    path=Path(c.assets_dirs)/'local/libero_expert2k/norm_stats.json'
    stats=json.loads(path.read_text())['norm_stats']
    for key in ('state','actions'):
        for field in ('mean','std','q01','q99'):assert np.isfinite(stats[key][field]).all()
    report=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),extra_delta=True,algorithm='pinned official compute_norm_stats.py',source='local/libero_expert2k',computed_on_cpu=True)
    (ROOT/'artifacts/jax_2k/norm_verified.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
