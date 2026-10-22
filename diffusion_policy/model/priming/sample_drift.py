"""High-precision measures separating shifts in policy mean from sample spread."""
import torch


def half_pairwise_squared(live, reference):
    """Mean 0.5||live_i-ref_j||² in float64, to avoid cancellation."""
    if live.ndim != 3 or reference.ndim != 3 or live.shape[1:] != reference.shape[1:]:
        raise ValueError("expected live/ref [K,H,Da] with same H,Da")
    a=live.detach().reshape(live.shape[0],-1).double()
    b=reference.detach().reshape(reference.shape[0],-1).double()
    return float((a[:,None,:]-b[None,:,:]).square().sum(-1).mean()*0.5)


def sample_variance(live):
    """Biased per-coordinate sample variance (same convention as prior audit)."""
    x=live.detach().double()
    return float((x-x.mean(0,keepdim=True)).square().mean())


def mean_shift_rms(current, initial):
    """RMS displacement of mean action chunk, in normalized action units."""
    a=current.detach().double().mean(0)
    b=initial.detach().double().mean(0)
    return float((a-b).square().mean().sqrt())
