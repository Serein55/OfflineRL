import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from arfm_jax.action_quality import event_rewards,geometric_prefix_score,paired_monte_carlo_advantage,prefix_features,fit_pairwise_ranker
class QualityTests(unittest.TestCase):
    def test_causal_event_once(self):
        fs=[dict(contact=c,goal=False) for c in [True]*5+[False]*5+[True]*5]
        x=event_rewards(fs);self.assertEqual(x[4],.2);self.assertEqual(x[9],-.2);self.assertEqual(x[14],0.)
        self.assertEqual(x[:4].sum(),0.)
    def test_goal_release(self):
        fs=[dict(contact=True,goal=False)]*5+[dict(contact=False,goal=True)]*5
        x=event_rewards(fs);self.assertAlmostEqual(x.sum(),1.2)
    def test_prefix_gate_does_not_switch(self):
        before=dict(contact=False,reach_distance=.1,goal_xy_distance=.5)
        after=dict(contact=True,reach_distance=.05,goal_xy_distance=.9)
        self.assertAlmostEqual(geometric_prefix_score(before,after,0),.1)
    def test_same_state_baseline(self):
        q,a=paired_monte_carlo_advantage([[1,1,0,1],[0,0,1,0]])
        np.testing.assert_allclose(q,[.75,.25]);np.testing.assert_allclose(a,[.25,-.25])
        with self.assertRaises(ValueError):paired_monte_carlo_advantage([[1,np.nan]])
    def test_ranker_no_future_features(self):
        before=dict(contact=False,reach_distance=.1,goal_xy_distance=.5,height=.05,goal=False)
        after={**before,'reach_distance':.05}
        good=dict(task=0,episode=0,anchor=40,success=True,before=before,after5=after,event_h5=0)
        bad={**good,'success':False,'after5':{**before,'reach_distance':.15}}
        np.testing.assert_array_equal(prefix_features(good),prefix_features({**good,'success':False,'steps':220}))
        beta,meta=fit_pairwise_ranker([good,bad]);self.assertGreater(prefix_features(good)@beta,prefix_features(bad)@beta)
        self.assertEqual(meta['pairs'],1)
if __name__=='__main__':unittest.main()
