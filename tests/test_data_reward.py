import numpy as np
import torch
from arfm.reward import table8_reward
from arfm.data import TaskBalancedSampler

def test_reward_components_and_terminal_success():
    image=np.zeros((5,16,16,3),dtype='uint8')
    joints=np.zeros((5,7)); actions=np.zeros((5,7))
    reward,parts,goals=table8_reward(image,image,joints,actions)
    weights=np.full(13,.1/13); weights[10:12]=.01/13
    assert parts.shape==(5,13)
    assert np.allclose(reward,parts@weights)
    assert parts[:-1,12].sum()==0 and parts[-1,12]==1
    assert np.isfinite(reward).all() and goals[-1]==4

def test_four_rank_sampler_matches_single_global_batch():
    class D:
        tasks=list(range(40)); offsets=np.arange(41)*100
    one=list(TaskBalancedSampler(D(),3,seed=42))
    ranks=[list(TaskBalancedSampler(D(),3,rank=r,world=4,seed=42)) for r in range(4)]
    for step in range(3):
        assert [i for rank in ranks for i in rank[step*4:(step+1)*4]]==one[step*16:(step+1)*16]
    assert list(TaskBalancedSampler(D(),3,seed=42,start=2))==one[32:]
