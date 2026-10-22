"""Numerical central differences along normalized model-parameter gradients."""
import math
import torch


def unit_loss_gradient(loss, parameters):
    params=list(parameters)
    grads=torch.autograd.grad(loss, params, allow_unused=True)
    norm=math.sqrt(sum(float(g.detach().double().square().sum())
                       for g in grads if g is not None))
    if not math.isfinite(norm) or norm<=0:
        raise ValueError("zero or nonfinite gradient")
    return [torch.zeros_like(p) if g is None else (g.detach()/norm)
            for p,g in zip(params,grads)], norm


def symmetric_scores(parameters, direction, radius, evaluate):
    """Call evaluate at +/- radius * unit direction, restoring exact weights."""
    params=list(parameters)
    if len(params)!=len(direction) or not params or radius<=0:
        raise ValueError("invalid parameter perturbation")
    original=[p.detach().clone() for p in params]
    def assign(scale):
        with torch.no_grad():
            for p,base,d in zip(params,original,direction):
                p.copy_(base + scale*d)
    try:
        assign(+radius)
        plus=evaluate()
        assign(-radius)
        minus=evaluate()
        return plus,minus
    finally:
        assign(0.0)


def comparison(base,plus,minus,h):
    derivative=(plus-minus)/(2*h)
    predicted=-h*derivative
    observed=minus-base
    return dict(directional_derivative=derivative,
                delta_predicted=predicted,
                delta_observed=observed,
                nonlinear_remainder=observed-predicted,
                symmetry_residual=plus+minus-2*base)
