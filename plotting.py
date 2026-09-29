"""Reads the CSVs produced by run_experiment.py and produces the 5 requested plots."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BITS_LIST = [8, 6, 4, 3, 2]
R_LIST = [1, 2, 4, 8, 16, 32]


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            r = {}
            for k, v in row.items():
                if v == "" or v is None:
                    r[k] = None
                else:
                    try:
                        r[k] = int(v)
                    except ValueError:
                        try:
                            r[k] = float(v)
                        except ValueError:
                            r[k] = v
            rows.append(r)
        return rows


def mean_std(values):
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / max(n - 1, 1)
    return m, var ** 0.5


def plot1_mse_vs_replicas(raw_rows, out_dir):
    for bits in BITS_LIST:
        det = sorted([r for r in raw_rows if r["method"] == "deterministic" and r["bits"] == bits], key=lambda r: r["R"])
        det_x = [r["R"] for r in det]
        det_y = [r["mse"] for r in det]

        stoch_by_R = defaultdict(list)
        for r in raw_rows:
            if r["method"] == "stochastic" and r["bits"] == bits:
                stoch_by_R[r["R"]].append(r["mse"])
        stoch_x = sorted(stoch_by_R.keys())
        stoch_mean = [mean_std(stoch_by_R[R])[0] for R in stoch_x]
        stoch_std = [mean_std(stoch_by_R[R])[1] for R in stoch_x]

        plt.figure(figsize=(5, 4))
        plt.plot(det_x, det_y, "o-", label="Deterministic")
        plt.errorbar(stoch_x, stoch_mean, yerr=stoch_std, fmt="s-", label="Stochastic (mean ± std over seeds)")
        plt.xlabel("Number of replicas R")
        plt.ylabel("Activation reconstruction MSE")
        plt.title(f"MSE vs. replicas — {bits}-bit quantization")
        plt.xscale("log", base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot1_mse_vs_replicas_bits{bits}.png"), dpi=150)
        plt.close()


def plot2_accuracy_vs_replicas(raw_rows, out_dir):
    for bits in BITS_LIST:
        det = sorted([r for r in raw_rows if r["method"] == "deterministic" and r["bits"] == bits], key=lambda r: r["R"])
        det_x = [r["R"] for r in det]
        det_y = [r["accuracy"] for r in det]

        stoch_by_R = defaultdict(list)
        for r in raw_rows:
            if r["method"] == "stochastic" and r["bits"] == bits:
                stoch_by_R[r["R"]].append(r["accuracy"])
        stoch_x = sorted(stoch_by_R.keys())
        stoch_mean = [mean_std(stoch_by_R[R])[0] for R in stoch_x]
        stoch_std = [mean_std(stoch_by_R[R])[1] for R in stoch_x]

        plt.figure(figsize=(5, 4))
        plt.plot(det_x, det_y, "o-", label="Deterministic")
        plt.errorbar(stoch_x, stoch_mean, yerr=stoch_std, fmt="s-", label="Stochastic (mean ± std over seeds)")
        plt.xlabel("Number of replicas R")
        plt.ylabel("Classification accuracy")
        plt.title(f"Accuracy vs. replicas — {bits}-bit quantization")
        plt.xscale("log", base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot2_accuracy_vs_replicas_bits{bits}.png"), dpi=150)
        plt.close()


def plot3_divergence_vs_bits(raw_rows, out_dir, R_fixed=32):
    det_y, stoch_mean_y, stoch_std_y = [], [], []
    for bits in BITS_LIST:
        det_val = [r["divergence"] for r in raw_rows if r["method"] == "deterministic" and r["bits"] == bits and r["R"] == R_fixed]
        det_y.append(det_val[0] if det_val else 0.0)
        stoch_vals = [r["divergence"] for r in raw_rows if r["method"] == "stochastic" and r["bits"] == bits and r["R"] == R_fixed]
        m, s = mean_std(stoch_vals)
        stoch_mean_y.append(m)
        stoch_std_y.append(s)

    plt.figure(figsize=(5, 4))
    plt.plot(BITS_LIST, det_y, "o-", label="Deterministic (should be ≈ 0)")
    plt.errorbar(BITS_LIST, stoch_mean_y, yerr=stoch_std_y, fmt="s-", label="Stochastic (honest divergence)")
    plt.xlabel("Quantization bits")
    plt.ylabel(f"Inter-replica divergence (R={R_fixed})")
    plt.title("Inter-replica divergence vs. bit rate")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_divergence_vs_bits.png"), dpi=150)
    plt.close()


def plot4_accuracy_vs_bits(simple_rows, out_dir):
    uncompressed = [r["accuracy_mean"] for r in simple_rows if r["method"] == "uncompressed"][0]
    det = sorted([r for r in simple_rows if r["method"] == "deterministic"], key=lambda r: -r["bits"])
    stoch = sorted([r for r in simple_rows if r["method"] == "stochastic_R1"], key=lambda r: -r["bits"])

    plt.figure(figsize=(5, 4))
    plt.axhline(uncompressed, color="gray", linestyle="--", label="Uncompressed (no split-inference loss)")
    plt.plot([r["bits"] for r in det], [r["accuracy_mean"] for r in det], "o-", label="Deterministic")
    plt.errorbar([r["bits"] for r in stoch], [r["accuracy_mean"] for r in stoch],
                 yerr=[r["accuracy_std"] for r in stoch], fmt="s-", label="Stochastic (R=1, mean ± std over seeds)")
    plt.xlabel("Quantization bits")
    plt.ylabel("Classification accuracy")
    plt.title("Clean task accuracy vs. compression level (R=1)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_accuracy_vs_bits.png"), dpi=150)
    plt.close()


def plot5_variance_loglog(biasvar_rows, out_dir):
    plt.figure(figsize=(5.5, 4.5))
    for bits in BITS_LIST:
        rows = sorted([r for r in biasvar_rows if r["bits"] == bits], key=lambda r: r["R"])
        Rs = [r["R"] for r in rows]
        var = [max(r["variance"], 1e-12) for r in rows]
        plt.plot(Rs, var, "o-", label=f"{bits} bits")

    # Reference 1/R slope, anchored at R=1 of the 8-bit curve.
    ref_rows = sorted([r for r in biasvar_rows if r["bits"] == 8], key=lambda r: r["R"])
    anchor = max(ref_rows[0]["variance"], 1e-12)
    ref_R = [r["R"] for r in ref_rows]
    ref_y = [anchor / R for R in ref_R]
    plt.plot(ref_R, ref_y, "k--", label="∝ 1/R reference")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Number of replicas R")
    plt.ylabel("Variance of stochastic reconstruction  Var(ā_R)")
    plt.title("Variance vs. replicas (log-log)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_variance_loglog.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    raw_rows = read_csv(os.path.join(args.results_dir, "replica_grid_raw.csv"))
    simple_rows = read_csv(os.path.join(args.results_dir, "simple_accuracy_vs_bits.csv"))
    biasvar_rows = read_csv(os.path.join(args.results_dir, "bias_variance.csv"))

    plot1_mse_vs_replicas(raw_rows, args.results_dir)
    plot2_accuracy_vs_replicas(raw_rows, args.results_dir)
    plot3_divergence_vs_bits(raw_rows, args.results_dir)
    plot4_accuracy_vs_bits(simple_rows, args.results_dir)
    plot5_variance_loglog(biasvar_rows, args.results_dir)
    print(f"Plots written to {args.results_dir}/")


if __name__ == "__main__":
    main()
