import numpy as np
import torch
from arfm.action_transform import extra_delta,ExtraDeltaInference
from scripts.compute_delta_stats import action_chunks
from scripts.evaluate import select_replanned_action
from test_evaluator import QueueHarness

def test_chunk_start_anchor_and_padding():
    actions=np.arange(21,dtype=np.float32).reshape(3,7)
    states=np.arange(24,dtype=np.float32).reshape(3,8)
    chunks=action_chunks(actions,states,horizon=4)
    expected=np.stack([actions[np.minimum(np.arange(4)+t,2)] for t in range(3)])
    np.testing.assert_array_equal(chunks[...,-1],expected[...,-1])
    restored=chunks.copy();restored[...,:6]+=states[:,None,:6]
    np.testing.assert_array_equal(restored,expected)
    np.testing.assert_array_equal(actions,np.arange(21).reshape(3,7))
    np.testing.assert_array_equal(extra_delta(expected[1],states[1]),chunks[1])

class DeltaHarness(ExtraDeltaInference,QueueHarness):
    extra_delta_transform=True
    def prepare_state(self,batch):return batch['observation.state'][:,:1]
    def unnormalize_outputs(self,batch):return {'action':batch['action']*2+3}

def test_inverse_after_unnormalization_uses_frozen_anchor_and_replan():
    policy=DeltaHarness()
    for t in range(12):
        state=torch.full((1,8),float(t))
        action=select_replanned_action(policy,{'observation.state':state},t,5)
        start=t//5*5
        expected=(100*start+t%5)*2+3
        torch.testing.assert_close(action[0,:6],torch.full((6,),float(expected+start)))
        assert action[0,6]==expected
        torch.testing.assert_close(state,torch.full((1,8),float(t)))
    assert policy.calls==[0,5,10]

def test_disabled_preserves_raw_behavior():
    policy=DeltaHarness();policy.extra_delta_transform=False
    action=select_replanned_action(policy,{'observation.state':torch.ones(1,8)},0,5)
    torch.testing.assert_close(action,torch.full((1,7),203.))
