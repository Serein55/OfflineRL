"""ARFM objective on globally sharded JAX arrays.

The batch axis is GLOBAL under OpenPI jit/NamedSharding. Do not multiply by
world size (that correction belongs only to PyTorch DDP's gradient averaging).
"""
import jax
import jax.numpy as jnp


def solve_alpha(advantage, losses, lam=5e-4, alpha_min=.01, alpha_max=5., iterations=20, tol=1e-5):
    if not (0 <= alpha_min <= alpha_max and lam >= 0 and iterations > 0):
        raise ValueError('Invalid solver parameters')
    r, l = jax.lax.stop_gradient(jnp.asarray(advantage)), jax.lax.stop_gradient(jnp.asarray(losses))
    if r.ndim != 1 or r.shape != l.shape or not r.size:
        raise ValueError('Expected equally sized nonempty vectors')
    r2 = jnp.mean(r*r)
    sigma = jnp.sqrt(jnp.maximum(r2, 1e-30))
    c = lam*sigma/jnp.maximum(jnp.var(l), 1e-12)
    def residual(x):
        # Beyond this point the positive term dominates for float32 inputs.
        e = jnp.exp(jnp.minimum(x, 40.))
        return jnp.where(x > 40, jnp.inf, 2*jnp.sqrt(x)*e*(2*e-1)-c)
    low, high = r2*alpha_min**2, r2*alpha_max**2
    def body(_, bounds):
        lo, hi, done = bounds
        mid = (lo+hi)/2
        f = residual(mid)
        hit = jnp.abs(f)<tol
        new_lo = jnp.where(hit | (f<=0), mid, lo)
        new_hi = jnp.where(hit | (f>0), mid, hi)
        return jnp.where(done,lo,new_lo), jnp.where(done,hi,new_hi), done|hit
    lo,hi,_ = jax.lax.fori_loop(0,iterations,body,(low,high,jnp.array(False)))
    alpha = jnp.clip(jnp.sqrt((lo+hi)/2)/sigma,alpha_min,alpha_max)
    alpha = jnp.where(residual(high)<=0,alpha_max,alpha)
    alpha = jnp.where((r2<1e-12)|(residual(low)>=0),alpha_min,alpha)
    return jax.lax.stop_gradient(alpha)


def weighted_loss(per_sample, advantage, method='arfm', fixed_alpha=.1, **solver):
    if method not in ('vanilla','rwr','arfm'):raise ValueError(method)
    losses=jnp.asarray(per_sample,dtype=jnp.float32)
    adv=jax.lax.stop_gradient(jnp.asarray(advantage,dtype=jnp.float32))
    if losses.ndim!=1 or losses.shape!=adv.shape or not losses.size:raise ValueError('Expected matching vectors')
    alpha=jnp.asarray(0. if method=='vanilla' else fixed_alpha) if method!='arfm' else solve_alpha(adv,losses,**solver)
    weights=jax.lax.stop_gradient(jax.nn.softmax(alpha*adv))
    loss=jnp.sum(weights*losses)
    metrics=dict(loss=loss,fm_mean=jnp.mean(losses),fm_std=jnp.std(losses),
        adv_mean=jnp.mean(adv),adv_std=jnp.std(adv),alpha=alpha,
        ess=1/jnp.sum(weights**2),weight_max=jnp.max(weights),weight_min=jnp.min(weights),
        entropy=-jnp.sum(weights*jnp.log(jnp.maximum(weights,1e-30))))
    return loss,metrics
