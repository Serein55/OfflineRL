"""Read-only analysis of existing evaluations and exact logged training batches."""
import csv,json,runpy,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/arfm_analysis';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
names=runpy.run_path(str(ROOT/'vendor/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']
runs={'vanilla':'vanilla_uniform_replan5_seed42','arfm_raw':'arfm_uniform_replan5_seed42','arfm_z':'arfm_taskz_uniform_seed42','rwr_01':'rwr_taskz_a01_uniform_seed42','rwr_05':'rwr_taskz_a05_uniform_seed42','vanilla_delta':'vanilla_extra_delta_uniform_seed42'}
suites=['libero_spatial','libero_goal','libero_object','libero_10']
evals={};table=[];paired={}
for method,run in runs.items():
 evals[method]={}
 for suite in suites:
  rows=[json.loads(s) for s in (ROOT/'artifacts'/run/f'eval_{suite}.jsonl').read_text().splitlines()]
  assert len(rows)==500 and {(r['task_id'],r['episode']) for r in rows}=={(t,e) for t in range(10) for e in range(50)}
  assert all(r['seed']==42 and r['replan_steps']==5 for r in rows)
  evals[method][suite]=np.array([[next(r['success'] for r in rows if r['task_id']==t and r['episode']==e) for e in range(50)] for t in range(10)],dtype=int)
for suite in suites:
 for t,name in enumerate(names[suite]):
  row={'suite':suite,'task_id':t,'task':name}
  for method in runs:row[method]=int(evals[method][suite][t].sum())
  table.append(row)
 for method in runs:
  if method=='vanilla':continue
  a=evals['vanilla'][suite];b=evals[method][suite]
  wins=int(((a==0)&(b==1)).sum());losses=int(((a==1)&(b==0)).sum());n=wins+losses
  p=min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
  paired[f'{suite}/{method}']={'gained':wins,'lost':losses,'net':wins-losses,'mcnemar_exact_unadjusted_p':p,'note':'descriptive single-seed paired episodes; clustered by task, multiple comparisons not corrected'}
with (out/'task_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
# Build task-local arrays exactly matching dataset's chunk indices.
tasks=[];taskstats=[]
weights=np.full(13,.1/13);weights[10:12]=.01/13
for file in manifest['files']:
 with np.load(ROOT/file) as d:
  adv=d['advantage'].astype(float);lengths=d['lengths'];parts=d['components'].astype(float)
  pos=np.concatenate([np.arange(T)/max(1,T-1) for T in lengths]);remain=np.concatenate([np.arange(T,0,-1) for T in lengths])
  rtgparts=[];start=0
  for T in lengths:
   c=parts[start:start+T]*weights;rtgparts.append(np.cumsum(c[::-1],axis=0)[::-1]/np.arange(T,0,-1)[:,None]);start+=T
  c=np.concatenate(rtgparts);g=c.sum(1);center=c-c.mean(0)
  assert np.allclose((g-g.mean())*len(g)/(len(g)-1),adv,atol=1e-7)
  z=((adv-adv.mean())/(adv.std()+1e-8)).astype('float32')
  pad=np.maximum(50-remain,0)/50
  contribution=(center*(g-g.mean())[:,None]).mean(0)/g.var()
  top=z>=np.quantile(z,.95)
  taskstats.append({'file':file,'suite':Path(file).parent.name,'task':Path(file).stem.removesuffix('_demo'),'std':float(adv.std()),'zmax':float(z.max()),'correlation_z_time':float(np.corrcoef(z,pos)[0,1]),'top5_mean_time':float(pos[top].mean()),'top5_mean_padding':float(pad[top].mean()),'all_mean_padding':float(pad.mean()),'rtg_variance_contribution':contribution.tolist()})
  tasks.append({'raw':adv.astype('float32'),'z':z,'time':pos,'pad':pad,'terminal':remain==1,'late':pos>=.8,'early':pos<.2})
# Exact replay of seed42 task-balanced sampler and alpha from each actual step.
training={'arfm_raw':'arfm_uniform_seed42','arfm_z':'arfm_taskz_uniform_seed42','rwr_01':'rwr_taskz_a01_uniform_seed42','rwr_05':'rwr_taskz_a05_uniform_seed42'}
replay={}
for method,run in training.items():
 rows={r['step']:r for r in map(json.loads,(ROOT/'artifacts'/run/'metrics.jsonl').read_text().splitlines())}
 mass=np.zeros(40);counts=np.zeros(40);metrics={k:0. for k in ['time','pad','terminal','late','early']};uniform=dict(metrics);err=0.;ess=[]
 for step,row in sorted(rows.items()):
  rng=np.random.default_rng(np.random.SeedSequence([42,step-1]));ids=rng.integers(40,size=16)
  ix=[int(rng.integers(len(tasks[t]['z']))) for t in ids]
  a=np.array([tasks[t]['raw' if method=='arfm_raw' else 'z'][j] for t,j in zip(ids,ix)],dtype='float32')
  logits=row['alpha']*a;w=np.exp(logits-logits.max());w/=w.sum()
  e=float(1/(w*w).sum());ess.append(e);err=max(err,abs(e-row['ess']))
  np.add.at(mass,ids,w);np.add.at(counts,ids,1/16)
  for k in metrics:
   v=np.array([tasks[t][k][j] for t,j in zip(ids,ix)]);metrics[k]+=float(w@v);uniform[k]+=float(v.mean())
 n=len(rows);assert n==40000 and err<1e-4,(method,err)
 replay[method]={'steps':n,'max_ess_reconstruction_error':err,'ess_mean':float(np.mean(ess)),'weighted':{k:v/n for k,v in metrics.items()},'uniform':{k:v/n for k,v in uniform.items()},'tasks':[{'task':taskstats[t]['task'],'suite':taskstats[t]['suite'],'weight_share':float(mass[t]/n),'sample_share':float(counts[t]/n),'weight_ratio':float(mass[t]/counts[t])} for t in range(40)]}
 print('replayed',method,flush=True)
result={'task_comparison':table,'paired':paired,'advantage_diagnostics':taskstats,'actual_training_weight_replay':replay,'component_order':['image_mse','image_ssim','image_orb','wrist_mse','wrist_ssim','wrist_orb','joint_mse','progress','joint_velocity','joint_acceleration','action_velocity','action_acceleration','terminal_success']}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('SPATIAL')
for row in table[:10]:print(row)
print('REPLAY')
for k,v in replay.items():print(k,{key:val for key,val in v.items() if key!='tasks'})
print('ADVANTAGE SUITES')
for suite in suites:
 rows=[r for r in taskstats if r['suite']==suite]
 print(suite,{k:float(np.mean([r[k] for r in rows])) for k in ['correlation_z_time','top5_mean_time','top5_mean_padding','all_mean_padding']},'components',np.mean([r['rtg_variance_contribution'] for r in rows],0).round(3))
