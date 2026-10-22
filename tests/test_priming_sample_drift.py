"""CPU numerical sanity checks for reference drift measurements."""
import pytest
import torch
from diffusion_policy.model.priming.sample_drift import (
    half_pairwise_squared,sample_variance,mean_shift_rms)


def test_mean_shift_does_not_change_diversity():
    x=torch.tensor([[[-1.]],[[1.]]])
    y=x+4
    assert sample_variance(y)==pytest.approx(sample_variance(x))
    assert mean_shift_rms(y,x)==pytest.approx(4.)
    assert half_pairwise_squared(y,y)==pytest.approx(1.)
    assert half_pairwise_squared(y,x)==pytest.approx(9.)


def test_fresh_reference_cancels_shared_translation():
    x=torch.tensor([[[-0.003]],[[0.003]]])
    y=x+0.5
    fresh=half_pairwise_squared(y,y)
    stale=half_pairwise_squared(y,x)
    original=half_pairwise_squared(x,x)
    assert fresh==pytest.approx(original)
    assert stale>1000*original


def test_shapes_are_enforced():
    with pytest.raises(ValueError):
        half_pairwise_squared(torch.zeros(4,2,2),torch.zeros(4,3,2))
