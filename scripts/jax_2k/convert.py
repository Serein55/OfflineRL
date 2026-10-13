"""Lossless episode/frame-preserving HDF5 -> LeRobot v2 parquet, with reward audit."""
import argparse,hashlib,io,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/openpi_lerobot/local/libero_expert2k'
ART=ROOT/'artifacts/jax_2k'
SCHEMA_FILE=ROOT/'configs/expert2k_schema.arrow'

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def image_record(frame):
    b=io.BytesIO();Image.fromarray(np.ascontiguousarray(frame[::-1,::-1])).save(b,format='PNG',compress_level=1)
    return {'bytes':b.getvalue(),'path':None}

def convert_task(job):
    task_id,npz_path,first_episode,first_frame=job
    with np.load(npz_path,allow_pickle=False) as f:old={k:f[k] for k in f.files}
    source=Path(str(old['source']));lengths=old['lengths'];episodes=old['episodes']
    assert len(lengths)==len(episodes)==50
    schema=pa.ipc.read_schema(pa.BufferReader(SCHEMA_FILE.read_bytes()));offset=0;mapping=[];states=[];actions=[];timesteps=[]
    with h5py.File(source,'r') as raw:
        language=json.loads(raw['data'].attrs['problem_info'])['language_instruction']
        env=json.loads(raw['data'].attrs['env_args']);assert env['env_kwargs']['control_freq']==20
        assert set(raw['data'].keys())==set(map(str,episodes))
        for local,(demo,n) in enumerate(zip(episodes,lengths)):
            n=int(n);d=raw['data'][str(demo)];o=d['obs'];a=d['actions'][:].astype('float32')
            st=np.concatenate([o['ee_pos'][:],o['ee_ori'][:],o['gripper_states'][:]],axis=-1).astype('float32')
            assert len(a)==n and st.shape==(n,8) and a.shape==(n,7)
            assert o['agentview_rgb'].shape==(n,128,128,3) and o['eye_in_hand_rgb'].shape==(n,128,128,3)
            assert np.isfinite(a).all() and np.isfinite(st).all()
            ep=first_episode+local;start=first_frame+offset;dest=OUT/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
            dest.parent.mkdir(parents=True,exist_ok=True)
            values=dict(image=[image_record(x) for x in o['agentview_rgb'][:]],wrist_image=[image_record(x) for x in o['eye_in_hand_rgb'][:]],
                state=st.tolist(),actions=a.tolist(),timestamp=(np.arange(n,dtype=np.float32)/20).tolist(),
                frame_index=np.arange(n),episode_index=np.full(n,ep),index=np.arange(start,start+n),task_index=np.full(n,task_id))
            table=pa.Table.from_pydict(values,schema=schema)
            temp=dest.with_suffix('.tmp');pq.write_table(table,temp,compression='snappy');temp.replace(dest)
            # Verify stored state/actions/indices and every decoded PNG, not just a few samples.
            read=pq.read_table(dest).to_pydict()
            np.testing.assert_array_equal(read['state'],st);np.testing.assert_array_equal(read['actions'],a)
            for key in ('index','episode_index','frame_index','task_index'):np.testing.assert_array_equal(read[key],values[key])
            for key,rawkey in [('image','agentview_rgb'),('wrist_image','eye_in_hand_rgb')]:
                frames=o[rawkey][:]
                for t,img in enumerate(read[key]):np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(img['bytes']))),frames[t,::-1,::-1])
            r=old['reward'][offset:offset+n];rtg=np.cumsum(r[::-1])[::-1]/np.arange(n,0,-1)
            np.testing.assert_allclose(rtg,old['rtg'][offset:offset+n],rtol=1e-12,atol=1e-12)
            mapping.append(dict(episode_index=ep,task_index=task_id,task=language,source=str(source),source_demo=str(demo),
                start_index=start,length=n,source_frame_start=0,frame_mapping='identity',parquet_sha256=digest(dest)))
            states.append(st);actions.append(a);timesteps.append(n);offset+=n
    G=old['rtg'];expected=(G-G.mean())*len(G)/(len(G)-1)
    np.testing.assert_array_equal(expected.astype('float32'),old['advantage'])
    assert offset==int(lengths.sum())
    report=dict(task_index=task_id,task=language,source=str(source),source_sha256=digest(source),reward_source=str(npz_path),reward_sha256=digest(npz_path),episodes=mapping)
    (ART/f'task_{task_id:02d}.json').write_text(json.dumps(report,indent=2))
    np.savez(ART/f'task_{task_id:02d}_arrays.npz',state=np.concatenate(states),actions=np.concatenate(actions),lengths=lengths,
        reward=old['reward'],rtg=old['rtg'],advantage=old['advantage'])
    print('DONE TASK',task_id,offset,flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    if (ART/'conversion_complete.json').exists():raise FileExistsError('Conversion already complete')
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text());jobs=[];ep=frame=0
    for task,path in enumerate(manifest['files']):
        path=ROOT/path
        with np.load(path) as f:n=int(f['lengths'].sum());e=len(f['episodes'])
        jobs.append((task,path,ep,frame));ep+=e;frame+=n
    assert ep==2000 and frame==manifest['samples'] and len(jobs)==40
    with ProcessPoolExecutor(max_workers=a.workers) as pool:reports=list(pool.map(convert_task,jobs))
    meta=OUT/'meta';meta.mkdir(parents=True,exist_ok=True)
    info=json.loads((ROOT/'configs/expert2k_info_template.json').read_text())
    info.update(total_episodes=ep,total_frames=frame,total_tasks=40,total_chunks=2,fps=20,splits={'train':'0:2000'})
    for key in ('image','wrist_image'):info['features'][key]['shape']=[128,128,3]
    (meta/'info.json').write_text(json.dumps(info,indent=2))
    (meta/'tasks.jsonl').write_text(''.join(json.dumps(dict(task_index=r['task_index'],task=r['task']))+'\n' for r in reports))
    all_episodes=[e for r in reports for e in r['episodes']]
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=e['episode_index'],tasks=[e['task']],length=e['length']))+'\n' for e in all_episodes))
    (ART/'episode_mapping.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in all_episodes))
    arrays={key:[] for key in ('state','actions','reward','rtg','advantage')}
    for i in range(40):
        with np.load(ART/f'task_{i:02d}_arrays.npz') as f:
            for key in arrays:arrays[key].append(f[key])
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    stats={}
    for key in ('state','actions'):
        x=arrays[key].astype('float64');stats[key]=dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),count=[len(x)])
    # These are raw reader metadata, NOT model training normalization statistics.
    (meta/'stats.json').write_text(json.dumps(stats,indent=2))
    from arfm_jax.annotations import metadata_hash
    metadata=dict(dataset_id='local/libero_expert2k',dataset_revision='expert2k-preserved-v1',metadata_sha256=metadata_hash(OUT),
        normalization='none',reward_description='Exact historical Table8 reconstruction on original raw observations/actions/joints; no recomputation',
        mapping_sha256=digest(ART/'episode_mapping.jsonl'),manifest_sha256=digest(ROOT/'data/processed/manifest.json'))
    sidecar=ROOT/'data/jax_2k/annotations.npz';sidecar.parent.mkdir(parents=True,exist_ok=True)
    with sidecar.open('wb') as f:np.savez_compressed(f,index=np.arange(frame),episode_index=np.concatenate([np.full(e['length'],e['episode_index']) for e in all_episodes]),
        frame_index=np.concatenate([np.arange(e['length']) for e in all_episodes]),task_index=np.concatenate([np.full(e['length'],e['task_index']) for e in all_episodes]),
        reward=arrays['reward'],rtg=arrays['rtg'],advantage=arrays['advantage'],raw_advantage=arrays['advantage'],metadata=np.array(json.dumps(metadata)))
    result=dict(dataset_id=metadata['dataset_id'],episodes=ep,frames=frame,tasks=40,fps=20,image_shape=[128,128,3],image_transform='rotate180 lossless PNG',
        dropped_frames=0,filtered_noops=False,rtg_recomputed_and_verified=True,advantage_recomputed_and_verified=True,reward_reused_exactly=True,
        metadata_sha256=metadata_hash(OUT),annotations_sha256=digest(sidecar))
    (ART/'conversion_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
