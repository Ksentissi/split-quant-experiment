"""Small numerical helpers for activation-level metrics and simple stats."""
import math
import torch


def mse_per_image(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample, averaged over (C,H,W). Shape [N]."""
    return (a - b).flatten(1).pow(2).mean(dim=1)


def rel_error_per_image(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Relative L2 error ||a-b||/||a|| per sample. Shape [N]."""
    return (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp(min=1e-12)


def pairwise_divergence_matrix(replicas: torch.Tensor) -> torch.Tensor:
    """
    replicas: [R, N, D] (D = flattened feature dim).
    Returns the [N, R, R] matrix of pairwise L2 distances between replicas,
    for every sample.
    """
    x = replicas.permute(1, 0, 2)  # [N, R, D]
    return torch.cdist(x, x)


def divergence_from_matrix(mat_RxR: torch.Tensor, R: int) -> torch.Tensor:
    """
    mat_RxR: [N, R, R] pairwise distance matrix (diagonal is 0).
    Returns, per sample, (1/(R*(R-1))) * sum_{i!=j} ||q_i - q_j||_2 -- the
    average pairwise inter-replica divergence defined in the experiment.
    """
    n = mat_RxR.shape[0]
    if R < 2:
        return torch.zeros(n)
    total = mat_RxR.sum(dim=(-2, -1))
    return total / (R * (R - 1))


def mean_std_ci95(values):
    """Mean, sample std (ddof=1) and an approximate 95% CI half-width
    (normal approximation: 1.96 * std / sqrt(n)) over a 1D list/array."""
    values = list(values)
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
    else:
        var = 0.0
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return mean, std, ci95


def paired_t_stat(diffs):
    """
    Paired t-statistic for a list of per-image (or per-seed) differences
    (e.g. stochastic_metric - deterministic_metric). Used to check whether
    an observed improvement is unlikely to be pure noise. Returns
    (t_stat, approx_p_two_sided) using a normal approximation to the
    t-distribution (adequate for the sample sizes used here and avoids an
    extra dependency on scipy).
    """
    diffs = list(diffs)
    n = len(diffs)
    mean, std, _ = mean_std_ci95(diffs)
    if std == 0 or n < 2:
        return 0.0, 1.0
    t = mean / (std / math.sqrt(n))
    # Normal approximation to the two-sided p-value.
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p
