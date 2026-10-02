"""Exercise the frozen LeRobot action queue, without loading model weights or GPUs."""
import argparse
from collections import deque
from types import SimpleNamespace
import json
import pytest
import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from scripts.evaluate import select_replanned_action,load_existing_results,replan_steps

class QueueHarness:
    select_action=PI0Policy.select_action
    def __init__(self):
        self.config=SimpleNamespace(adapt_to_pi_aloha=False,action_feature=SimpleNamespace(shape=(7,)),
                                    chunk_size=50,n_action_steps=50)
        self._action_queue=deque(maxlen=50)
        self.calls=[]
        self.model=SimpleNamespace(sample_actions=self.sample_actions)
    def eval(self): pass
    def reset(self): self._action_queue.clear()
    def normalize_inputs(self,batch): return batch
    def unnormalize_outputs(self,batch): return batch
    def prepare_images(self,batch): return [],[]
    def prepare_state(self,batch): return batch['observation.state']
    def prepare_language(self,batch): return None,None
    def sample_actions(self,images,masks,tokens,lang_masks,state,noise=None):
        observation=int(state.item()); self.calls.append(observation)
        return (100*observation+torch.arange(50)).view(1,50,1).expand(1,50,7).float()

def execute(policy,step,interval):
    batch={'observation.state':torch.tensor([[step]])}
    return select_replanned_action(policy,batch,step,interval)[0,0].item()

def test_five_steps_uses_new_observation_and_discards_tail():
    policy=QueueHarness()
    actions=[execute(policy,t,5) for t in range(12)]
    assert policy.calls==[0,5,10]
    assert actions==[0,1,2,3,4,500,501,502,503,504,1000,1001]
    assert policy.config.chunk_size==policy.config.n_action_steps==50

def test_default_fifty_matches_original_queue():
    policy=QueueHarness(); original=QueueHarness()
    for t in range(103):
        expected=original.select_action({'observation.state':torch.tensor([[t]])})[0,0].item()
        assert execute(policy,t,50)==expected
    assert policy.calls==[0,50,100]

def test_episode_reset_prevents_stale_actions():
    policy=QueueHarness(); execute(policy,0,5); execute(policy,1,5)
    execute(policy,0,5)
    assert policy.calls==[0,0]

def test_result_protocol_guard(tmp_path):
    output=tmp_path/'eval.jsonl'
    output.write_text(json.dumps({'seed':42,'success':False})+'\n')
    assert len(load_existing_results(output,50,42))==1
    with pytest.raises(ValueError,match='new --output'): load_existing_results(output,5,42)
    output.write_text(json.dumps({'seed':42,'replan_steps':5,'success':True})+'\n')
    assert load_existing_results(output,5,42)[0]['success']
    with pytest.raises(ValueError): load_existing_results(output,50,42)
    with pytest.raises(ValueError): load_existing_results(output,5,43)

@pytest.mark.parametrize('value',['0','-1','51'])
def test_invalid_intervals(value):
    with pytest.raises(argparse.ArgumentTypeError): replan_steps(value)
