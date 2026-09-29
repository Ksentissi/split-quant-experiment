"""
Compression methods for the cut-layer activation `a`.

All methods share the SAME clipping range (l, u) and the SAME number of
quantization levels for a given bit budget, so the comparison between
deterministic and stochastic quantization isolates the effect of
stochasticity only (Control #3 in the experimental design).

Notation (per scalar element x of the activation tensor):
    levels = 2**bits
    delta  = (u - l) / (levels - 1)      # step size of the uniform grid
    grid points: l, l+delta, l+2*delta, ..., u

Deterministic quantizer:
    Q_d(x) = l + round( (clip(x,l,u) - l) / delta ) * delta

Stochastic quantizer (unbiased "stochastic rounding"):
    Let idx = (clip(x,l,u) - l) / delta, lo = floor(idx), frac = idx - lo.
    Round to lo with probability (1 - frac), to lo+1 with probability frac.
    E[Q_s(x) | x] = l + (lo*(1-frac) + (lo+1)*frac) * delta
                  = l + (lo + frac) * delta = l + idx * delta = clip(x,l,u)
    => exactly unbiased for any x inside [l, u].
Values outside [l, u] are clipped identically for both methods, so any
bias from clipping affects both methods equally and is not a
stochastic-vs-deterministic artifact.
"""
import torch


def compute_range(a: torch.Tensor):
    """
    Per-sample clipping range (l, u), computed once from the clean
    activation `a` itself and reused, unchanged, by every compressor and
    every bit budget. a: [N, C, H, W]. Returns l, u of shape [N, 1, 1, 1].
    """
    flat = a.flatten(1)
    l = flat.min(dim=1).values.view(-1, 1, 1, 1)
    u = flat.max(dim=1).values.view(-1, 1, 1, 1)
    eps = 1e-8
    u = torch.where(u - l < eps, l + eps, u)  # guard against a degenerate (constant) activation
    return l, u


def quantize_deterministic(a: torch.Tensor, bits: int, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Deterministic uniform quantization: round-to-nearest-grid-point."""
    levels = 2 ** bits
    delta = (u - l) / (levels - 1)
    x = torch.clamp(a, min=l, max=u)
    idx = torch.round((x - l) / delta)
    return l + idx * delta


def quantize_stochastic(
    a: torch.Tensor, bits: int, l: torch.Tensor, u: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """
    Unbiased stochastic uniform quantization on the SAME grid as
    quantize_deterministic. `generator` must be an independent
    torch.Generator per replica/seed so that noise is not correlated
    across replicas (Control #2).
    """
    levels = 2 ** bits
    delta = (u - l) / (levels - 1)
    x = torch.clamp(a, min=l, max=u)
    idx_float = (x - l) / delta
    lo = torch.floor(idx_float)
    frac = idx_float - lo
    hi = torch.clamp(lo + 1, max=levels - 1)
    lo = torch.clamp(lo, max=levels - 1)

    noise = torch.rand(a.shape, generator=generator, device=a.device)
    idx = torch.where(noise < frac, hi, lo)
    return l + idx * delta


def make_generator(device: torch.device, base_seed: int, *tags: int) -> torch.Generator:
    """
    Builds a torch.Generator with a seed deterministically derived from
    (base_seed, *tags) -- e.g. (base_seed, bits, replica_idx) -- so that
    every replica/seed combination gets its own independent random stream
    (Control #2: independent randomness across stochastic replicas).
    """
    seed = base_seed
    for t in tags:
        seed = (seed * 1_000_003 + int(t)) % (2 ** 31 - 1)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
