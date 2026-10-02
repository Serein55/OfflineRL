"""Download immutable model and official raw demonstrations; safe to resume."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
ROOT = Path(__file__).resolve().parents[1]
api = HfApi(token=False)
for repo, typ, dest, revision, patterns in [
    ('lerobot/pi0_old', 'model', 'checkpoints/pi0', 'e4ed526af508e58f6008b29e9e48f1098278fdb5', ['config.json','model.safetensors','README.md']),
    ('yifengzhu-hf/LIBERO-datasets','dataset','data/raw','f13aa24a3da8c43c7225569f28c562979fa0e35a',['libero_goal/*','libero_spatial/*','libero_object/*','libero_10/*']),
]:
    info=api.repo_info(repo,repo_type=typ,revision=revision)
    record={'repo':repo,'revision':info.sha,'type':typ,'status':'downloading'}
    manifest=ROOT/dest/'download.json'; manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(record,indent=2))
    print(f'Downloading {repo}@{info.sha}',flush=True)
    snapshot_download(repo,repo_type=typ,revision=info.sha,local_dir=str(ROOT/dest),allow_patterns=patterns,max_workers=8,token=False)
    record['status']='complete'; manifest.write_text(json.dumps(record,indent=2))
    print(f'Completed {repo}',flush=True)
