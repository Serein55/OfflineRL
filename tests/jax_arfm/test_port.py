import importlib.util
import unittest
from pathlib import Path
import numpy as np
import math
import jax
import jax.numpy as jnp
from arfm_jax.objective import weighted_loss,solve_alpha
from arfm_jax.annotations import prepare
ROOT=Path(__file__).resolve().parents[2]
def reference_weights(losses, advantage, method):
    a=np.asarray(advantage,dtype=np.float64);l=np.asarray(losses,dtype=np.float64)
    alpha=0. if method=='vanilla' else .1
    if method=='arfm':
        r2=np.mean(a*a)
        alpha=.01
        if r2>=1e-12:
            sigma=math.sqrt(r2);c=5e-4*sigma/max(float(np.var(l)),1e-12)
            def f(x):
                return math.inf if x>350 else 2*math.sqrt(x)*math.exp(x)*(2*math.exp(x)-1)-c
            lo,hi=r2*.01**2,r2*5.**2
            if f(hi)<=0:alpha=5.
            elif f(lo)<0:
                for _ in range(20):
                    mid=(lo+hi)/2;value=f(mid)
                    if abs(value)<1e-5:lo=hi=mid;break
                    if value>0:hi=mid
                    else:lo=mid
                alpha=min(5.,max(.01,math.sqrt((lo+hi)/2)/sigma))
    z=alpha*a;w=np.exp(z-z.max());return w/w.sum(),alpha

class PortTests(unittest.TestCase):
    def test_numpy_reference_and_detached_gradient(self):
        rng=np.random.default_rng(42)
        for adv in [np.zeros(32),np.ones(32)*.1,rng.normal(size=32)*.003,rng.normal(size=32),rng.normal(size=32)*100]:
            for losses in [np.ones(32),rng.uniform(.01,2,32)]:
                for method in ['arfm','rwr','vanilla']:
                    a=adv.astype('float32');l=losses.astype('float32')
                    w,alpha=reference_weights(l,a,method);expected=float(w@l)
                    got,m=jax.jit(lambda l,a:weighted_loss(l,a,method))(jnp.array(l),jnp.array(a))
                    np.testing.assert_allclose(got,expected,rtol=3e-5,atol=2e-6)
                    np.testing.assert_allclose(m['alpha'],alpha,rtol=3e-4,atol=3e-6)
                    np.testing.assert_allclose(m['ess'],1/(w*w).sum(),rtol=3e-4)
                    grad=jax.grad(lambda x:weighted_loss(x,jnp.array(a),method)[0])(jnp.array(l))
                    np.testing.assert_allclose(grad,w,rtol=3e-4,atol=2e-6)
                    adv_grad=jax.grad(lambda x:weighted_loss(jnp.array(l),x,method)[0])(jnp.array(a))
                    np.testing.assert_array_equal(adv_grad,np.zeros(32))
    def test_global_sharding(self):
        from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
        self.assertEqual(jax.device_count(),4)
        sh=NamedSharding(Mesh(np.array(jax.devices()),('batch',)),P('batch'))
        a=np.linspace(-2,3,32,dtype='float32');l=np.linspace(.1,1,32,dtype='float32')
        f=jax.jit(lambda l,a:weighted_loss(l,a),in_shardings=(sh,sh))
        got,_=f(jax.device_put(l,sh),jax.device_put(a,sh))
        w,_=reference_weights(l,a,'arfm');expected=w@l
        np.testing.assert_allclose(got,expected,rtol=3e-5)
        grad=jax.jit(jax.grad(lambda x:weighted_loss(x,jnp.array(a))[0]),in_shardings=sh)(jax.device_put(l,sh))
        np.testing.assert_allclose(grad,w,rtol=3e-4,atol=1e-6)
    def test_rtg_loo_and_zscore(self):
        r=np.array([1.,0.,2.,3.,1.]);ep=np.array([0,0,0,1,1]);fr=np.array([0,1,2,0,1]);task=np.zeros(5,dtype=int)
        got=prepare(r,ep,fr,task)
        rtg=np.array([1.,1.,2.,2.,1.]);adv=(rtg-rtg.mean())*len(rtg)/(len(rtg)-1)
        np.testing.assert_allclose(got['rtg'],rtg,rtol=1e-6)
        np.testing.assert_allclose(got['advantage'],adv,atol=1e-7)
        z=prepare(r,ep,fr,task,'task_zscore')['advantage']
        np.testing.assert_allclose(z,(adv-adv.mean())/(adv.std()+1e-8),rtol=1e-6)
        with self.assertRaises(ValueError):prepare(r,ep,np.zeros(5),task)
    def test_train_step_and_ema(self):
        import dataclasses
        from types import SimpleNamespace
        import flax.nnx as nnx
        import optax
        from openpi.training.utils import TrainState
        from arfm_jax.training import train_step
        class TinyModel(nnx.Module):
            def __init__(self):self.linear=nnx.Linear(3,2,rngs=nnx.Rngs(42))
            def compute_loss(self,rng,obs,actions,train=True):return (self.linear(obs)-actions)**2
        model=TinyModel();graph,params=nnx.split(model);tx=optax.adam(1e-3)
        state=TrainState(step=jnp.array(0),params=params,model_def=graph,opt_state=tx.init(params),tx=tx,ema_decay=.99,ema_params=params)
        config=SimpleNamespace(trainable_filter=nnx.Param)
        obs=jnp.arange(96,dtype=jnp.float32).reshape(32,3)/96;actions=jnp.zeros((32,2));adv=jnp.linspace(-1,1,32)
        f=jax.jit(lambda st:train_step(config,jax.random.key(7),st,(obs,actions,adv)))
        new,metrics=f(state)
        self.assertEqual(int(new.step),1);self.assertTrue(np.isfinite(float(metrics['loss'])))
        self.assertGreater(float(metrics['ess']),0)
        old_leaves=jax.tree.leaves(params);new_leaves=jax.tree.leaves(new.params)
        self.assertTrue(any(not np.array_equal(x,y) for x,y in zip(old_leaves,new_leaves)))
        for old,updated,ema in zip(old_leaves,new_leaves,jax.tree.leaves(new.ema_params)):
            np.testing.assert_allclose(ema,.99*np.asarray(old)+.01*np.asarray(updated),rtol=1e-6,atol=1e-7)
        # Vanilla adapter must produce the same gradient/update as unweighted FM.
        base=nnx.merge(graph,params)
        _,g=nnx.value_and_grad(lambda m:jnp.mean(m.compute_loss(None,obs,actions)),argnums=nnx.DiffState(0,nnx.Param))(base)
        updates,_=tx.update(g,state.opt_state,params);expected=optax.apply_updates(params,updates)
        vanilla,_=train_step(config,jax.random.key(7),state,(obs,actions,adv),method='vanilla')
        for got,want in zip(jax.tree.leaves(vanilla.params),jax.tree.leaves(expected)):
            np.testing.assert_allclose(got,want,atol=1e-7)

    def test_annotation_identity_and_sample_alignment(self):
        import tempfile,json
        from arfm_jax.annotations import load_sidecar,metadata_hash,REVISION
        from arfm_jax.data import AnnotatedDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meta').mkdir()
            for name in ('info.json','episodes.jsonl','tasks.jsonl'):(root/'meta'/name).write_text('{}')
            fields=dict(index=np.arange(2),episode_index=np.zeros(2,dtype=int),frame_index=np.arange(2),task_index=np.zeros(2,dtype=int))
            values=prepare(np.array([0.,1.]),fields['episode_index'],fields['frame_index'],fields['task_index'])
            metadata=dict(dataset_revision=REVISION,metadata_sha256=metadata_hash(root))
            path=root/'ann.npz';np.savez(path,**fields,**values,metadata=np.array(json.dumps(metadata)))
            annotations,_=load_sidecar(path,root,2)
            raw=[{k:v[i] for k,v in fields.items()} for i in range(2)]
            dataset=AnnotatedDataset(raw,lambda x:dict(x),annotations)
            self.assertEqual(float(dataset[1]['advantage']),float(values['advantage'][1]))
            raw[1]['frame_index']=999
            with self.assertRaises(ValueError):dataset[1]
            metadata.update(dataset_id='local/libero_expert2k',dataset_revision='expert2k-preserved-v1')
            np.savez(path,**fields,**values,metadata=np.array(json.dumps(metadata)))
            load_sidecar(path,root,2,dataset_id='local/libero_expert2k')
            with self.assertRaises(ValueError):load_sidecar(path,root,2)
            (root/'meta/info.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):load_sidecar(path,root,2)

    def test_shapes_and_parameters(self):
        with self.assertRaises(ValueError):solve_alpha(jnp.ones(2),jnp.ones(3))
        with self.assertRaises(ValueError):solve_alpha(jnp.ones(2),jnp.ones(2),lam=-1)
        with self.assertRaises(ValueError):weighted_loss(jnp.ones(2),jnp.ones(2),'invalid')

if __name__=='__main__':unittest.main()
