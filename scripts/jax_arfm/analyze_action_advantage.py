"""Within-state ranking; bootstrap INITIAL EPISODES, not action pairs."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from arfm_jax.action_quality import geometric_prefix_score,fit_pairwise_ranker,prefix_features


def summarize(rows,method,natural_only):
    grouped={}
    for row in rows:
        if natural_only and row['variant'] not in ('policy_a','policy_b'):continue
        grouped.setdefault((row['task'],row['episode'],row['anchor']),[]).append(row)
    wins=pairs=ties=0;clusters={};informative=0
    for (task,ep,anchor),group in grouped.items():
        w=n=t=0
        for a,b in itertools.combinations(group,2):
            if a['success']==b['success']:continue
            good,bad=(a,b) if a['success'] else (b,a)
            d=good[method]-bad[method];tie=abs(d)<1e-10
            w+=.5 if tie else float(d>0);n+=1;t+=int(tie)
        informative+=int(n>0);wins+=w;pairs+=n;ties+=t
        k=(task,ep);prev=clusters.get(k,(0.,0));clusters[k]=(prev[0]+w,prev[1]+n)
    values=np.array(list(clusters.values()));rng=np.random.default_rng(42);boot=[]
    if len(values):
        for _ in range(2000):
            v=values[rng.integers(len(values),size=len(values))].sum(axis=0)
            if v[1]>0:boot.append(v[0]/v[1])
    return dict(states=len(grouped),informative_states=informative,initial_episodes=len(clusters),discordant_pairs=pairs,
                concordance=wins/pairs if pairs else None,tie_fraction=ties/pairs if pairs else None,
                episode_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist() if boot else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='+');p.add_argument('--fit-ranker',action='store_true');args=p.parse_args()
    rows=[json.loads(f.read_text()) for directory in args.directory for f in sorted(directory.glob('task*.json'))]
    keys=[(r['task'],r['episode'],r['anchor'],r['variant']) for r in rows];assert len(keys)==len(set(keys)), 'Duplicate branches'
    for r in rows:r['geometry_v2']=geometric_prefix_score(r['before'],r['after5'],r['event_h5'])
    methods=['event_h5','event_h50','geometry_v2'];ranker=None
    if args.fit_ranker:
        beta,ranker=fit_pairwise_ranker([r for r in rows if r['task'] in [0,4,5]])
        ranker['weights']=beta.tolist();methods.append('pairwise_v3')
        for r in rows:r['pairwise_v3']=float(prefix_features(r)@beta)
    result=dict(ranker=ranker,branches=len(rows),max_state_error=max((r['state_error'] for r in rows),default=None),completed=all((directory/'complete.json').exists() for directory in args.directory),results={})
    for split,tasks in [('development',[0,4,5]),('heldout',[1,2,3,6,7,8,9]),('all',list(range(10)))]:
        rr=[r for r in rows if r['task'] in tasks];result['results'][split]={}
        for mode,natural in [('natural',True),('with_controls',False)]:
            result['results'][split][mode]={m:summarize(rr,m,natural) for m in methods}
    result['per_task']={t:dict(branches=sum(r['task']==t for r in rows),successes=sum(r['success'] for r in rows if r['task']==t)) for t in range(10)}
    if args.fit_ranker:
        groups={}
        for r in rows:groups.setdefault((r['task'],r['episode'],r['anchor']),[]).append(r)
        scores=[]
        for group in groups.values():
            base=float(np.mean([r['pairwise_v3'] for r in group]))
            for r in group:scores.append({**{k:r[k] for k in ['task','episode','anchor','variant','success','event_h5','geometry_v2','pairwise_v3']},'advantage_v3':r['pairwise_v3']-base})
        (args.directory[0]/'candidate_scores.json').write_text(json.dumps(scores,indent=2))
    (args.directory[0]/('analysis.json' if len(args.directory)==1 else 'combined_analysis.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
