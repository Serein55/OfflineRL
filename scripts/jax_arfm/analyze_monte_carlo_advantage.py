"""Independent-seed validation of a same-state Monte Carlo advantage target."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from arfm_jax.action_quality import paired_monte_carlo_advantage


def paired_stats(outcomes):
    d=outcomes[1]-outcomes[0];rng=np.random.default_rng(42)
    boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1)
    wins=int((d>0).sum());losses=int((d<0).sum());n=wins+losses
    return dict(successes=outcomes.sum(axis=1).astype(int).tolist(),seeds=outcomes.shape[1],
                q=outcomes.mean(axis=1).tolist(),b_minus_a=float(d.mean()),
                paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                discordant_wins_b=wins,discordant_wins_a=losses,
                exact_paired_sign_test_p=binomtest(wins,n,.5).pvalue if n else 1.)


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    variants=['policy_a','policy_b'];ys=[];reference={}
    for replica in range(16):
        row=[]
        for v in variants:
            path=args.directory/f'replica_{replica}/task04_ep01_t040_{v}.json'
            r=json.loads(path.read_text());assert r['continuation_replica']==replica
            if v in reference:
                for key in ['prefix_actions','candidate_chunk']:
                    np.testing.assert_array_equal(r[key],reference[v][key])
                assert r['before']==reference[v]['before']
                # Identical action prefix must also produce identical first five outcomes.
                assert r['features'][:5]==reference[v]['features'][:5]
            else:reference[v]=r
            assert r['state_error']==0 and r['duplicate_prefix_error']<=1e-10
            row.append(int(r['success']))
        ys.append(row)
    outcomes=np.asarray(ys,float).T;q,adv=paired_monte_carlo_advantage(outcomes[:,:8])
    train=paired_stats(outcomes[:,:8]);validation=paired_stats(outcomes[:,8:]);all_stats=paired_stats(outcomes)
    same_direction=bool(np.sign(train['b_minus_a'])==np.sign(validation['b_minus_a']) and train['b_minus_a']!=0)
    supported=bool(same_direction and validation['exact_paired_sign_test_p']<.05)
    result=dict(scope='One development state only; no generalization claim',candidate_order=variants,
                all_states_candidates_and_first5_equal=True,success_by_seed=outcomes.astype(int).tolist(),
                estimation=train,validation=validation,pooled_descriptive_only=all_stats,
                estimated_advantage=adv.tolist(),validation_same_direction=same_direction,
                validated_difference=supported,conservative_advantage=(adv if supported else np.zeros(2)).tolist(),
                note='Bootstrap intervals can be degenerate at small n; exact paired test is required. Pooled outcomes are not independent validation.')
    (args.directory/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
