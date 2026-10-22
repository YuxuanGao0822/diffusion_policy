"""CPU numerical sanity tests; no checkpoint or GPU required."""
import pytest
import torch
from diffusion_policy.model.priming.finite_difference import (
    unit_loss_gradient,symmetric_scores,comparison)


def test_quadratic_noise_sensitivity_directional_derivative():
    # Toy generator G(z)=theta*z -> Jacobian score S(theta)=theta**2.
    theta=torch.nn.Parameter(torch.tensor(0.7))
    saved=theta.detach().clone()
    d,norm=unit_loss_gradient(theta,[theta])
    assert norm==pytest.approx(1.0)
    score=lambda:float(theta.detach().square())
    plus,minus=symmetric_scores([theta],d,0.01,score)
    assert torch.equal(theta.detach(),saved)
    values=comparison(score(),plus,minus,0.01)
    assert values["directional_derivative"]==pytest.approx(1.4,abs=1e-5)
    assert values["delta_predicted"]==pytest.approx(-0.014,abs=1e-6)
    assert values["delta_observed"]==pytest.approx(-0.0139,abs=1e-6)
    assert values["nonlinear_remainder"]==pytest.approx(0.0001,abs=1e-6)


def test_reverse_loss_reverses_direction():
    p=torch.nn.Parameter(torch.tensor(0.7))
    d,_=unit_loss_gradient(-p,[p])
    plus,minus=symmetric_scores([p],d,0.01,
                                lambda:float(p.detach().square()))
    assert comparison(0.49,plus,minus,0.01)["delta_predicted"]>0


def test_weights_restored_after_exception():
    p=torch.nn.Parameter(torch.tensor(1.0))
    def crash():
        raise RuntimeError("injected")
    with pytest.raises(RuntimeError,match="injected"):
        symmetric_scores([p],[torch.ones_like(p)],0.1,crash)
    assert float(p.detach())==1.


def test_zero_gradient_is_invalid():
    p=torch.nn.Parameter(torch.tensor(2.0))
    with pytest.raises(ValueError,match="zero"):
        unit_loss_gradient(p*0,[p])
