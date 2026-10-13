"""Build advantages from explicit, index-aligned rewards for the retained expert2k LIBERO dataset."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from arfm_jax.annotations import prepare,metadata_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--rewards',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reward-description',required=True)
    p.add_argument('--normalization',choices=['none','task_zscore'],default='none');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.rewards,allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
    keys=['index','episode_index','frame_index','task_index']
    columns={k:[] for k in keys}
    for file in sorted((a.dataset_root/'data').glob('**/*.parquet')):
        table=pq.read_table(file,columns=keys)
        for key in keys:columns[key].append(table[key].to_numpy())
    columns={k:np.concatenate(v) for k,v in columns.items()}
    n=json.loads((a.dataset_root/'meta/info.json').read_text())['total_frames']
    if not np.array_equal(columns['index'],np.arange(n)):raise ValueError('Dataset index order mismatch')
    for key in keys:
        if not np.array_equal(inputs[key],columns[key]):raise ValueError(f'Reward alignment mismatch: {key}')
    result=prepare(inputs['reward'],inputs['episode_index'],inputs['frame_index'],inputs['task_index'],a.normalization)
    meta=dict(dataset_id='local/libero_expert2k',dataset_revision='expert2k-preserved-v1',metadata_sha256=metadata_hash(a.dataset_root),reward_description=a.reward_description,
        reward_file_sha256=hashlib.sha256(a.rewards.read_bytes()).hexdigest(),normalization=a.normalization,rtg='undiscounted remaining-trajectory average',baseline='leave-one-out over all frames within concrete task',implementation='port of historical reconstruction, not official ARFM author code')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as stream:np.savez_compressed(stream,**columns,**result,metadata=np.array(json.dumps(meta)))
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
