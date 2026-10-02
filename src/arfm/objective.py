"""ARFM Eq. (7) solver and global-batch loss, including DDP semantics."""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def solve_alpha(advantage, losses, lam=5e-4, alpha_min=.01, alpha_max=5., iterations=20, tol=1e-5):
    if not (0 <= alpha_min <= alpha_max and lam >= 0 and iterations > 0):
        raise ValueError('Invalid solver parameters')
    r, l = advantage.detach().double().flatten(), losses.detach().double().flatten()
    if r.numel() != l.numel() or not r.numel() or not (torch.isfinite(r).all() and torch.isfinite(l).all()):
        raise ValueError('Expected equally sized finite nonempty vectors')
    r2 = r.square().mean().item()  # Paper E[R²]; task-level centering, no batch recentering.
    if r2 < 1e-12:
        return alpha_min
    sigma = math.sqrt(r2)
    variance = max(l.var(unbiased=False).item(), 1e-12)
    c = lam * sigma / variance
    def residual(x):
        if x > 350:  # The positive exponential term dominates; avoid overflow.
            return math.inf
        return 2 * math.sqrt(x) * math.exp(x) * (2 * math.exp(x) - 1) - c
    low, high = r2 * alpha_min**2, r2 * alpha_max**2
    if residual(low) >= 0:
        return alpha_min
    if residual(high) <= 0:
        return alpha_max
    for _ in range(iterations):
        mid = (low + high) / 2
        f = residual(mid)
        if abs(f) < tol:
            low = high = mid
            break
        if f > 0:
            high = mid
        else:
            low = mid
    return min(alpha_max, max(alpha_min, math.sqrt((low + high)/2)/sigma))

@torch.no_grad()
def gather_detached(x):
    if not dist.is_initialized():
        return x.detach()
    pieces = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, x.detach().contiguous())
    return torch.cat(pieces)

def weighted_loss(per_sample, advantage, method='arfm', fixed_alpha=.1, **solver):
    """Use all 16 samples across GPUs for statistics AND softmax, not four local softmaxes.

    DDP averages gradients, hence multiply each rank's partial sum by world size.
    This assumes equal local batch sizes and no gradient accumulation.
    """
    losses, adv = gather_detached(per_sample.float()), gather_detached(advantage.float())
    alpha = 0. if method == 'vanilla' else fixed_alpha if method == 'rwr' else solve_alpha(adv, losses, **solver)
    if method not in ('vanilla', 'rwr', 'arfm'):
        raise ValueError(method)
    weights = torch.softmax(alpha * adv, dim=0)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local = weights[rank*len(per_sample):(rank+1)*len(per_sample)]
    loss = world * (local * per_sample).sum()
    metrics = dict(loss=(weights*losses).sum().item(), fm_mean=losses.mean().item(),
                   fm_std=losses.std(unbiased=False).item(), adv_mean=adv.mean().item(),
                   adv_std=adv.std(unbiased=False).item(), alpha=alpha,
                   ess=weights.square().sum().reciprocal().item(), weight_max=weights.max().item(),
                   weight_min=weights.min().item(), entropy=-(weights*weights.clamp_min(1e-30).log()).sum().item())
    return loss, metrics

def average_rtg(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float64)
    if r.ndim != 1 or not r.numel() or not torch.isfinite(r).all():
        raise ValueError('Rewards must be a finite nonempty vector')
    return r.flip(0).cumsum(0).flip(0) / torch.arange(len(r), 0, -1, dtype=r.dtype, device=r.device)

def leave_one_out(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        raise ValueError('LOO needs at least two samples')
    return (x-x.mean()) * (x.numel()/(x.numel()-1))

def learning_rate(step, warmup=1000, decay=30000, peak=2.5e-5, floor=2.5e-6):
    if step < warmup:
        return peak * (step+1)/warmup
    progress = min((step-warmup)/decay, 1.)
    return floor + .5*(peak-floor)*(1+math.cos(math.pi*progress))
