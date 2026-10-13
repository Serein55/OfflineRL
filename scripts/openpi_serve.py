"""Serve EMA inference weights using official OpenPI policy transforms."""
import argparse,json,os
from pathlib import Path
import numpy as np
from jax_2k.config import make_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,default=8104);p.add_argument('--check-only',action='store_true');p.add_argument('--paired-rng',action='store_true');a=p.parse_args()
policy=policy_config.create_trained_policy(make_config(),a.checkpoint)
if a.paired_rng:
    import jax
    class PairedPolicy:
        """Diagnostic-only episode RNG reset; defaults preserve official inference."""
        def __init__(self,wrapped):self.wrapped=wrapped
        @property
        def metadata(self):return {**self.wrapped.metadata,'paired_episode_rng':True}
        def infer(self,obs):
            obs=dict(obs);seed=obs.pop('_diagnostic_episode_seed',None)
            if seed is not None:self.wrapped._rng=jax.random.key(int(seed))
            return self.wrapped.infer(obs)
    policy=PairedPolicy(policy)
if a.check_only:
    result=policy.infer({'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),
        'observation/state':np.zeros(8,np.float32),'prompt':'pick up the black bowl'})
    assert result['actions'].shape==(50,7) and np.isfinite(result['actions']).all()
    print('PASS checkpoint reload and policy inference',result['actions'].shape,flush=True)
else:websocket_policy_server.WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()
