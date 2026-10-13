"""Experimental Spatial prefix scores. Not wired into policy training."""
import numpy as np


def event_rewards(features):
    """One-time causal contact/loss/goal rewards from online simulator labels."""
    held=False;on=off=0;paid_contact=paid_loss=paid_goal=False;out=[]
    for f in features:
        value=0.
        if f['goal'] and not paid_goal:value+=1.;paid_goal=True
        on=on+1 if f['contact'] else 0;off=off+1 if not f['contact'] else 0
        if not held and on>=5:
            held=True
            if not paid_contact and not paid_goal:value+=.2;paid_contact=True
        elif held and off>=5:
            held=False
            if not paid_loss and not paid_goal:value-=.2;paid_loss=True
        out.append(value)
    return np.asarray(out)


def geometric_prefix_score(before,after,event_h5):
    """v2: initial-state-gated distance progress; no future success used.

    Fixed before heldout inspection. 10cm is the distance scale. Gate is taken
    ONLY from the shared initial state so a candidate cannot improve its score
    merely by changing the phase definition. Contact remains a proxy for grasp.
    """
    if before['contact']:
        progress=(before['goal_xy_distance']-after['goal_xy_distance'])/.1
    else:
        progress=.2*(before['reach_distance']-after['reach_distance'])/.1
    return float(event_h5+progress)


def paired_monte_carlo_advantage(outcomes):
    """Q minus mean candidate Q, evaluated at EXACTLY the same state.

    outcomes shape [candidates, independent continuation seeds]. All candidates
    must be evaluated with the same set of seeds. This is a rollout-based target,
    not an event predictor; it cannot serve as its own validation criterion.
    """
    y=np.asarray(outcomes,float)
    if y.ndim!=2 or min(y.shape)<1 or not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError('Expected finite success outcomes [candidate, seed] in [0,1]')
    q=y.mean(axis=1)
    return q,q-q.mean()


PREFIX_FEATURES = ['event_h5','reach_progress_unheld','goal_progress_held',
                   'goal_progress_unheld','contact_change','height_change_held',
                   'height_change_unheld','goal_within_prefix']


def prefix_features(row):
    """Only initial state and first <=5 action outcomes; no continuation labels."""
    a,b=row['before'],row['after5'];held=float(a['contact'])
    reach=(a['reach_distance']-b['reach_distance'])/.1
    goal=(a['goal_xy_distance']-b['goal_xy_distance'])/.1
    dz=(b['height']-a['height'])/.05
    return np.clip([row['event_h5'],(1-held)*reach,held*goal,(1-held)*goal,
                    float(b['contact'])-held,held*dz,(1-held)*dz,float(b['goal'])],-5.,5.)


def fit_pairwise_ranker(rows,regularization=1.):
    """Development data only. Equal weight per informative initial state.

    Fit success-vs-failure feature DIFFERENCES at the same simulator state, so
    easy-vs-hard states cannot be used as a substitute for action quality.
    Includes explicit L2 shrinkage; weights are not calibrated Q probabilities.
    """
    from itertools import combinations
    from scipy.optimize import minimize
    from scipy.special import expit
    groups={}
    for row in rows:groups.setdefault((row['task'],row['episode'],row['anchor']),[]).append(row)
    xs=[];ws=[];states=0
    for group in groups.values():
        pairs=[]
        for a,b in combinations(group,2):
            if a['success']==b['success']:continue
            good,bad=(a,b) if a['success'] else (b,a)
            pairs.append(prefix_features(good)-prefix_features(bad))
        if pairs:xs.extend(pairs);ws.extend([1/len(pairs)]*len(pairs));states+=1
    if not xs:raise ValueError('No within-state success/failure contrasts in development data')
    x=np.asarray(xs);weight=np.asarray(ws)/states
    def objective(beta):
        z=x@beta
        loss=np.dot(weight,np.logaddexp(0,-z))+.5*regularization*np.dot(beta,beta)
        grad=-(x.T@(weight*expit(-z)))+regularization*beta
        return loss,grad
    result=minimize(objective,np.zeros(len(PREFIX_FEATURES)),jac=True,method='L-BFGS-B')
    if not result.success:raise RuntimeError(result.message)
    return result.x,dict(informative_states=states,pairs=len(xs),regularization=regularization,features=PREFIX_FEATURES)
