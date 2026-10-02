import math
import pytest
import torch
from arfm.objective import solve_alpha,weighted_loss,average_rtg,leave_one_out,learning_rate

def test_zero_advantage_loss_and_grad():
    x=torch.randn(16,50,32,requires_grad=True); per=x.square().mean((1,2))
    out,m=weighted_loss(per,torch.zeros(16))
    assert torch.allclose(out,x.square().mean())
    grad=torch.autograd.grad(out,x,retain_graph=True)[0]
    assert torch.allclose(grad,torch.autograd.grad(x.square().mean(),x)[0])
    assert m['ess']==16

def test_shift_fixed_alpha_and_order():
    adv=torch.arange(16,dtype=torch.float32); losses=torch.randn(16,requires_grad=True)
    a,_=weighted_loss(losses,adv,'rwr',.5); b,_=weighted_loss(losses,adv+20,'rwr',.5)
    assert torch.allclose(a,b)
    weights=torch.softmax(adv*.5,0)
    assert (weights[1:]>weights[:-1]).all()

def test_residual_and_bounds():
    adv=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64)
    losses=torch.tensor([.05,.06,.04,.07],dtype=torch.float64)
    alpha=solve_alpha(adv,losses)
    r2=adv.square().mean().item(); x=alpha**2*r2
    residual=4*math.sqrt(x)*math.exp(2*x)-2*math.sqrt(x)*math.exp(x)-5e-4*math.sqrt(r2)/losses.var(unbiased=False).item()
    assert abs(residual)<1e-4
    assert solve_alpha(adv,losses,lam=0)==.01
    assert solve_alpha(adv*.001,torch.ones(4))==5.
    assert math.isfinite(solve_alpha(adv*1e3,losses))

def test_rtg_loo():
    assert torch.allclose(average_rtg([1,2,3]),torch.tensor([2,2.5,3],dtype=torch.float64))
    assert torch.allclose(leave_one_out([1,2,3]),torch.tensor([-1.5,0,1.5],dtype=torch.float64))
    with pytest.raises(ValueError): leave_one_out([1])

def test_padding_fixed_denominator():
    losses=torch.ones(2,50,32); losses[:,25:]=0
    weighted,_=weighted_loss(losses.mean((1,2)),torch.zeros(2))
    assert weighted==.5

def test_schedule():
    assert learning_rate(999)==2.5e-5
    assert learning_rate(1000)==2.5e-5
    assert learning_rate(31000)==2.5e-6
    assert learning_rate(39999)==2.5e-6
