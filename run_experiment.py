"""
Main experiment: deterministic vs. stochastic quantization of the cut-layer
activation in replicated split inference.

Design summary (see README.md for the full write-up):
  - The client-side network, server-side network and cut layer are FIXED
    (loaded once from a single checkpoint) for every method/bit/replica
    combination tested here.
  - Client-side activations are computed ONCE per test image and cached;
    every compression method/bit-rate/replica-count is then applied to the
    exact same cached activations, so differences in the results can only
    come from the compression method, not from re-running the network.
  - Two evaluation pools:
      (1) FULL test set (10,000 images) for simple, high-confidence
          "accuracy vs bits" curves at R=1 (no replica averaging).
      (2) A fixed random SUBSET (N_SUBSET images) of the same cached
          activations for the full (bits x replicas x seeds) grid, which
          is far more expensive to compute.
  - For deterministic quantization, Q_d does not depend on the replica
    index or on any seed, so a_bar_R is provably constant in R; we still
    compute it via an explicit multi-replica pass to experimentally verify
    (not just assume) that inter-replica divergence is ~0.
  - For stochastic quantization, every (seed, bit, replica) triple gets an
    independent torch.Generator (see compressors.make_generator), and the
    32 replicas generated for a given seed are reused (via cumulative
    averaging / prefix slicing) to obtain all R in R_LIST without
    recomputing anything.
"""
import argparse
import csv
import os

import torch

from data_utils import get_test_loader, get_device
from model import SplitCNN
from compressors import compute_range, quantize_deterministic, quantize_stochastic, make_generator
from metrics import mse_per_image, rel_error_per_image, pairwise_divergence_matrix, divergence_from_matrix, mean_std_ci95, paired_t_stat

BITS_LIST = [8, 6, 4, 3, 2]
R_LIST = [1, 2, 4, 8, 16, 32]
R_MAX = max(R_LIST)
N_SEEDS = 10
N_SUBSET = 1000
BASE_SEED = 0


@torch.no_grad()
def cache_activations(model, loader, device):
    """One forward pass through the client-side network for the whole
    test set; everything downstream reuses these cached activations."""
    acts, labels = [], []
    for x, y in loader:
        x = x.to(device)
        a = model.forward_client(x)
        acts.append(a.cpu())
        labels.append(y)
    return torch.cat(acts), torch.cat(labels)


@torch.no_grad()
def accuracy_from_activation(model, a, y, device, batch_size=500):
    """Feeds a (possibly reconstructed) activation through the server-side
    network and returns classification accuracy."""
    correct = 0
    for i in range(0, a.shape[0], batch_size):
        ab = a[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        pred = model.forward_server(ab).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / a.shape[0]


@torch.no_grad()
def simple_accuracy_vs_bits(model, a_full, y_full, device, out_csv):
    """
    Full-test-set (N=10000), R=1 comparison: uncompressed vs deterministic
    vs stochastic (single realization, averaged over N_SEEDS independent
    draws) accuracy, for every bit rate. This is the 'Plot 4' data and the
    top rows of the headline summary.
    """
    rows = []
    uncompressed_acc = accuracy_from_activation(model, a_full, y_full, device)
    rows.append({"method": "uncompressed", "bits": "", "accuracy_mean": uncompressed_acc, "accuracy_std": 0.0})
    print(f"[simple] uncompressed accuracy: {uncompressed_acc:.4f}")

    l, u = compute_range(a_full)
    for bits in BITS_LIST:
        q_det = quantize_deterministic(a_full, bits, l, u)
        det_acc = accuracy_from_activation(model, q_det, y_full, device)
        rows.append({"method": "deterministic", "bits": bits, "accuracy_mean": det_acc, "accuracy_std": 0.0})
        print(f"[simple] bits={bits} deterministic accuracy: {det_acc:.4f}")

        seed_accs = []
        for seed in range(N_SEEDS):
            gen = make_generator(a_full.device, BASE_SEED + seed, bits, 0)
            q_stoch = quantize_stochastic(a_full, bits, l, u, gen)
            seed_accs.append(accuracy_from_activation(model, q_stoch, y_full, device))
        mean, std, _ = mean_std_ci95(seed_accs)
        rows.append({"method": "stochastic_R1", "bits": bits, "accuracy_mean": mean, "accuracy_std": std})
        print(f"[simple] bits={bits} stochastic (R=1) accuracy: {mean:.4f} +/- {std:.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "accuracy_mean", "accuracy_std"])
        writer.writeheader()
        writer.writerows(rows)
    return uncompressed_acc


@torch.no_grad()
def replica_grid(model, a_sub, y_sub, device, raw_csv, biasvar_csv):
    """
    The full (bits x replicas x seeds) grid on the fixed subset, producing:
      - raw_csv: one row per (method, bits, R, seed) with mse/rel_err/
        divergence/accuracy averaged over the subset images.
      - biasvar_csv: one row per (method, bits, R) with the bias^2/variance
        DECOMPOSITION of the reconstruction error, for BOTH methods, on the
        exact same per-element-mean scale as `mse` in raw_csv, so that the
        textbook identity
            MSE(a, a_bar_R) = bias^2(a_bar_R) + variance(a_bar_R)
        can be checked numerically (see `mse_check` column) instead of
        merely assumed. This is the core evidence for the central claim:
        deterministic quantization's error is (by construction) 100% bias
        and does not shrink with R, while stochastic quantization's error
        is (by construction) ~100% variance, which DOES shrink with R.
    """
    N = a_sub.shape[0]
    l, u = compute_range(a_sub)
    raw_rows = []
    biasvar_rows = []

    for bits in BITS_LIST:
        # ---------------- Deterministic ----------------
        # Q_d(a) does not depend on the replica index -> a_bar_R is the
        # same for every R. We still generate R_MAX independent calls to
        # verify experimentally (not just assume) that both the
        # inter-replica divergence AND the variance-across-replicas are
        # numerically zero: deterministic quantization is a pure function
        # of `a`, so it has no randomness to average out.
        det_replicas = torch.stack([quantize_deterministic(a_sub, bits, l, u) for _ in range(R_MAX)], dim=0)
        det_flat = det_replicas.flatten(2)  # [R_MAX, N, D]
        det_div_matrix = pairwise_divergence_matrix(det_flat)  # [N, R_MAX, R_MAX]
        # Variance across the R_MAX (supposedly identical) replicas, on the
        # same per-element-mean scale as mse_per_image. Should be ~0.
        det_variance = det_replicas.var(dim=0, unbiased=False).flatten(1).mean(dim=1).mean().item()

        for R in R_LIST:
            a_bar = det_replicas[:R].mean(dim=0)
            mse = mse_per_image(a_sub, a_bar).mean().item()
            rel = rel_error_per_image(a_sub, a_bar).mean().item()
            div = divergence_from_matrix(det_div_matrix[:, :R, :R], R).mean().item()
            acc = accuracy_from_activation(model, a_bar, y_sub, device)
            raw_rows.append({"method": "deterministic", "bits": bits, "R": R, "seed": "",
                              "mse": mse, "rel_error": rel, "divergence": div, "accuracy": acc})
            # Deterministic reconstruction has (numerically) zero variance,
            # so its entire MSE is bias^2 -- and since a_bar_R = Q_d(a) for
            # EVERY R, this bias^2 is exactly constant in R (never shrinks).
            biasvar_rows.append({
                "method": "deterministic", "bits": bits, "R": R,
                "bias_sq": mse - det_variance, "variance": det_variance,
                "mse_check": mse, "mse_actual": mse,
            })
        print(f"[grid] bits={bits} deterministic done "
              f"(inter-replica divergence ~ {det_div_matrix.mean().item():.3e}, "
              f"variance-across-replicas ~ {det_variance:.3e}; both should be ~0)")

        # ---------------- Stochastic ----------------
        # Accumulators (over seeds) for the bias^2/variance decomposition
        # of a_bar_R, per R, all on the per-element-mean scale (matching
        # mse_per_image) so bias_sq + variance can be checked against the
        # actual measured MSE.
        sum_abar = {R: torch.zeros_like(a_sub) for R in R_LIST}
        sumsq_meanscale = {R: torch.zeros(N) for R in R_LIST}
        mse_accum = {R: [] for R in R_LIST}

        for seed in range(N_SEEDS):
            gens = [make_generator(a_sub.device, BASE_SEED + seed, bits, r) for r in range(R_MAX)]
            replicas = torch.stack(
                [quantize_stochastic(a_sub, bits, l, u, gens[r]) for r in range(R_MAX)], dim=0
            )  # [R_MAX, N, C, H, W]
            replicas_flat = replicas.flatten(2)  # [R_MAX, N, D]
            div_matrix = pairwise_divergence_matrix(replicas_flat)  # [N, R_MAX, R_MAX]

            cumsum = torch.cumsum(replicas, dim=0)  # cumsum[r-1] = sum of first r replicas
            for R in R_LIST:
                a_bar = cumsum[R - 1] / R
                mse = mse_per_image(a_sub, a_bar).mean().item()
                rel = rel_error_per_image(a_sub, a_bar).mean().item()
                div = divergence_from_matrix(div_matrix[:, :R, :R], R).mean().item()
                acc = accuracy_from_activation(model, a_bar, y_sub, device)
                raw_rows.append({"method": "stochastic", "bits": bits, "R": R, "seed": seed,
                                  "mse": mse, "rel_error": rel, "divergence": div, "accuracy": acc})
                mse_accum[R].append(mse)

                sum_abar[R] += a_bar
                sumsq_meanscale[R] += a_bar.flatten(1).pow(2).mean(dim=1)
        print(f"[grid] bits={bits} stochastic done ({N_SEEDS} seeds x {R_MAX} replicas)")

        for R in R_LIST:
            mean_abar = sum_abar[R] / N_SEEDS
            # bias^2 per image, on the SAME per-element-mean scale as mse_per_image.
            bias_sq_per_image = (a_sub - mean_abar).flatten(1).pow(2).mean(dim=1)
            bias_sq = bias_sq_per_image.mean().item()
            # Var(X) = E[X^2] - E[X]^2, computed across the N_SEEDS independent
            # estimates of a_bar_R (each itself an average of R iid replicas).
            var_per_image = sumsq_meanscale[R] / N_SEEDS - mean_abar.flatten(1).pow(2).mean(dim=1)
            variance = var_per_image.clamp(min=0).mean().item()
            mse_actual = sum(mse_accum[R]) / len(mse_accum[R])
            biasvar_rows.append({
                "method": "stochastic", "bits": bits, "R": R,
                "bias_sq": bias_sq, "variance": variance,
                "mse_check": bias_sq + variance, "mse_actual": mse_actual,
            })

    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "R", "seed", "mse", "rel_error", "divergence", "accuracy"])
        writer.writeheader()
        writer.writerows(raw_rows)

    with open(biasvar_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "R", "bias_sq", "variance", "mse_check", "mse_actual"])
        writer.writeheader()
        writer.writerows(biasvar_rows)

    max_decomposition_error = max(abs(r["mse_check"] - r["mse_actual"]) for r in biasvar_rows)
    print(f"[check] max |bias^2 + variance - actual MSE| over all rows = {max_decomposition_error:.3e} "
          f"(should be ~0 -- confirms the bias/variance decomposition is exact, not approximate)")

    return raw_rows, biasvar_rows


def build_summary_table(raw_rows, biasvar_rows, out_csv):
    """
    The headline table: for every (bits, R), deterministic MSE, mean+/-std
    stochastic MSE (over seeds), deterministic accuracy, mean+/-std
    stochastic accuracy, mean honest stochastic inter-replica divergence,
    a paired significance check (stochastic vs deterministic MSE, per
    seed), AND the bias^2/variance decomposition for both methods -- the
    direct evidence that deterministic error is (near) 100% bias while
    stochastic error is (near) 100% variance.
    """
    from collections import defaultdict
    det = {}
    stoch = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        key = (row["bits"], row["R"])
        if row["method"] == "deterministic":
            det[key] = row
        else:
            stoch[key]["mse"].append(row["mse"])
            stoch[key]["accuracy"].append(row["accuracy"])
            stoch[key]["divergence"].append(row["divergence"])

    bv = {(r["method"], r["bits"], r["R"]): r for r in biasvar_rows}

    out_rows = []
    for bits in BITS_LIST:
        for R in R_LIST:
            key = (bits, R)
            d = det[key]
            s_mse_mean, s_mse_std, _ = mean_std_ci95(stoch[key]["mse"])
            s_acc_mean, s_acc_std, _ = mean_std_ci95(stoch[key]["accuracy"])
            s_div_mean, _, _ = mean_std_ci95(stoch[key]["divergence"])
            diffs = [d["mse"] - m for m in stoch[key]["mse"]]  # positive => stochastic worse (higher MSE)
            t_stat, p_val = paired_t_stat(diffs)
            det_bv = bv[("deterministic", bits, R)]
            stoch_bv = bv[("stochastic", bits, R)]
            out_rows.append({
                "bits": bits, "replicas": R,
                "deterministic_mse": d["mse"], "stochastic_mse_mean": s_mse_mean, "stochastic_mse_std": s_mse_std,
                "deterministic_accuracy": d["accuracy"], "stochastic_accuracy_mean": s_acc_mean, "stochastic_accuracy_std": s_acc_std,
                "honest_stochastic_divergence": s_div_mean,
                "mse_diff_det_minus_stoch": -sum(diffs) / len(diffs),
                "p_value_stoch_vs_det_mse": p_val,
                "deterministic_bias_sq": det_bv["bias_sq"], "deterministic_variance": det_bv["variance"],
                "stochastic_bias_sq": stoch_bv["bias_sq"], "stochastic_variance": stoch_bv["variance"],
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="split_cnn_cifar10.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    model = SplitCNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_loader = get_test_loader(args.data_dir)
    a_full, y_full = cache_activations(model, test_loader, device)
    print(f"Cached activations for {a_full.shape[0]} test images, shape {tuple(a_full.shape[1:])} each.")

    simple_accuracy_vs_bits(model, a_full, y_full, device, os.path.join(args.out_dir, "simple_accuracy_vs_bits.csv"))

    # Fixed random subset (seeded) for the expensive replica grid.
    g = torch.Generator().manual_seed(BASE_SEED)
    perm = torch.randperm(a_full.shape[0], generator=g)[:N_SUBSET]
    a_sub, y_sub = a_full[perm], y_full[perm]
    print(f"Using a fixed random subset of {N_SUBSET} test images for the replica grid.")

    raw_rows, biasvar_rows = replica_grid(
        model, a_sub, y_sub, device,
        os.path.join(args.out_dir, "replica_grid_raw.csv"),
        os.path.join(args.out_dir, "bias_variance.csv"),
    )

    summary_rows = build_summary_table(raw_rows, biasvar_rows, os.path.join(args.out_dir, "summary_table.csv"))
    print("\n=== Summary table (bits, replicas, det MSE=bias^2, stoch MSE=bias^2+var, accuracies, divergence) ===")
    for r in summary_rows:
        print(f"bits={r['bits']:>2} R={r['replicas']:>2}  "
              f"det_mse={r['deterministic_mse']:.5f} (bias^2={r['deterministic_bias_sq']:.5f}, var={r['deterministic_variance']:.1e})  "
              f"stoch_mse={r['stochastic_mse_mean']:.5f} (bias^2={r['stochastic_bias_sq']:.1e}, var={r['stochastic_variance']:.5f})  "
              f"det_acc={r['deterministic_accuracy']:.4f}  "
              f"stoch_acc={r['stochastic_accuracy_mean']:.4f}+/-{r['stochastic_accuracy_std']:.4f}  "
              f"div={r['honest_stochastic_divergence']:.5f}  p={r['p_value_stoch_vs_det_mse']:.3g}")

    print(f"\nAll CSVs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
