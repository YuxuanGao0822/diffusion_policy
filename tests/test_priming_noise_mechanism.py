"""CPU regression checks for latent-input Jacobians and SGD interventions."""
import pytest
import torch
from scripts.diagnose_priming_noise_mechanism import (
    sensitivity_from_function, take_sgd_step, window_slice,
)


def test_identity_jacobian_full_and_executed_are_exact():
    z = torch.randn(8,16,2)
    full = sensitivity_from_function(lambda x:x, z, slice(None),
                                     probes=3,seed=17)
    short = sensitivity_from_function(lambda x:x, z, slice(1,9),
                                      probes=3,seed=17)
    assert full["jacobian_sq_per_noise_dim"] == pytest.approx(1.,abs=1e-6)
    assert short["jacobian_sq_per_noise_dim"] == pytest.approx(0.5,abs=1e-6)


def test_zero_noise_sensitivity_is_zero():
    z = torch.randn(8,16,2)
    output = sensitivity_from_function(lambda x:x*0+0.2,z,slice(None))
    assert output["jacobian_sq_per_noise_dim"]==0.
    assert output["noise_std"]==pytest.approx(0.,abs=1e-7)


def test_matched_parameter_step_size():
    p = torch.nn.Parameter(torch.ones(4))
    before = p.detach().clone()
    metric = take_sgd_step([p],(p**2).sum(),lr=1e-5,
                           matched_step_norm=0.2)
    assert metric["gradient_norm"]==pytest.approx(4.)
    assert metric["parameter_update_l2"]==pytest.approx(0.2)
    assert (p-before).norm().item()==pytest.approx(0.2)


def test_zero_gradient_cannot_match_nonzero_step():
    p = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError,match="zero"):
        take_sgd_step([p],(p*0).sum(),lr=1e-5,matched_step_norm=0.2)


def test_executed_window_respects_oa_convention():
    class Fake:
        n_obs_steps=2
        n_action_steps=8
        oa_step_convention=True
    assert window_slice(Fake(),"executed")==slice(1,9)
    assert window_slice(Fake(),"full")==slice(None)
