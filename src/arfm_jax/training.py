"""OpenPI train step with detached global ARFM weights; optimizer/EMA unchanged."""
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from openpi.shared import nnx_utils
from arfm_jax.objective import weighted_loss

def train_step(config,rng,state,batch,*,method='arfm',fixed_alpha=.1,lam=5e-4):
    model=nnx.merge(state.model_def,state.params);model.train()
    observation,actions,advantage=batch
    def loss_fn(model):
        chunked=model.compute_loss(jax.random.fold_in(rng,state.step),observation,actions,train=True)
        per_sample=jnp.mean(chunked,axis=tuple(range(1,chunked.ndim)))
        return weighted_loss(per_sample,advantage,method,fixed_alpha,lam=lam)
    (loss,info),grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,config.trainable_filter),has_aux=True)(model)
    params=state.params.filter(config.trainable_filter)
    updates,opt_state=state.tx.update(grads,state.opt_state,params)
    nnx.update(model,optax.apply_updates(params,updates));new_params=nnx.state(model)
    new_state=dataclasses.replace(state,step=state.step+1,params=new_params,opt_state=opt_state)
    if state.ema_decay is not None:
        new_state=dataclasses.replace(new_state,ema_params=jax.tree.map(lambda old,new:state.ema_decay*old+(1-state.ema_decay)*new,state.ema_params,new_params))
    kernels=nnx.state(model,nnx.All(nnx.Param,nnx.Not(nnx_utils.PathRegex('.*/(bias|scale|pos_embedding|input_embedding)')),lambda _,x:x.value.ndim>1))
    info.update(grad_norm=optax.global_norm(grads),param_norm=optax.global_norm(kernels))
    return new_state,info
