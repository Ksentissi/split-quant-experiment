# Deterministic Quantization Creates an Aggregation Bias That Averaging Cannot Remove — Stochastic Quantization Does Not

**Research question:** when several honest replicas compress the same activation before sending it to a server, does *deterministic* quantization introduce an aggregation bias that persists no matter how many replicas are averaged, while *unbiased stochastic* quantization introduces (almost) none?

**Short answer: yes, and this is demonstrated numerically, not just assumed.**

## Method (in brief)

- **Model:** a small CNN trained on CIFAR-10 (86.6% clean accuracy), cut after the 2nd convolutional block → activation `a` of shape 64×8×8.
- **Compressors:** deterministic uniform quantization (round-to-nearest) vs. unbiased stochastic quantization (stochastic rounding, `E[Q_s(a)] = a` by construction), on exactly the same grid (same bounds, same number of levels) for 8, 6, 4, 3, 2 bits.
- **Replicas:** R = 1, 2, 4, 8, 16, 32, each with its own independent random generator in the stochastic case.
- **Bias/variance decomposition:** for every (bits, R, method), the **bias²** and **variance** of the averaged reconstruction `ā_R` are computed separately, on the same scale as the MSE, and the identity `MSE = bias² + variance` is checked numerically.
- Source code and raw CSVs: [github.com/Ksentissi/split-quant-experiment](https://github.com/Ksentissi/split-quant-experiment).

## Main proof: the bias² / variance decomposition

**Verification of the identity `MSE = bias² + variance`:** computed across 30 configurations (5 bit rates × 6 values of R), the maximum discrepancy between `bias² + variance` and the actually measured MSE is **4.16 × 10⁻⁹** — i.e. zero up to floating-point rounding error. The decomposition is therefore not an approximation; it is an identity verified on real data.

**Central result (example at 2 bits — see `results/plot6_bias_variance_decomposition_bits{2,3,4,6,8}.png` for every bit rate):**

| R | Deterministic bias² | Deterministic variance | Stochastic bias² | Stochastic variance |
|---|---|---|---|---|
| 1  | 0.1001 | **0.0 (exactly)** | 0.0200 | 0.1793 |
| 2  | 0.1001 | **0.0 (exactly)** | 0.0100 | 0.0897 |
| 4  | 0.1001 | **0.0 (exactly)** | 0.0050 | 0.0448 |
| 8  | 0.1001 | **0.0 (exactly)** | 0.0025 | 0.0224 |
| 16 | 0.1001 | **0.0 (exactly)** | 0.0012 | 0.0112 |
| 32 | 0.1001 | **0.0 (exactly)** | 0.0006 | 0.0056 |

Two facts, measured rather than assumed:

1. **The variance across deterministic replicas is exactly 0.000 × 10⁰ at every bit rate tested** (verified by generating 32 independent calls of `Q_d(a)` and measuring their empirical variance — see logs `[grid] ... variance-across-replicas ~ 0.000e+00`). This makes sense: `Q_d` is a deterministic function of `a`, so there is literally no randomness to average out. Hence `ā_R = Q_d(a)` for **every** R, and its total error (`MSE = bias² + 0`) is **strictly constant in R**: 0.1001 at R=1, still 0.1001 at R=32, to the digit.
2. **For stochastic quantization, the error is almost entirely variance, not bias** — and this variance decreases as 1/R with averaging. The small residual bias² measured (0.02 at R=1, shrinking toward 0 as R grows) is not a true bias: `E[Q_s(a)] = a` is an exact algebraic identity (see `compressors.py`, proof in the comments), so the theoretical bias is zero at *any* R. What is measured here is simply the Monte Carlo estimation noise from using only 10 seeds to estimate an expectation — and this residual noise also shrinks with R, which **confirms** convergence toward zero bias rather than contradicting it.

**Direct consequence:** because deterministic error is 100% (non-removable) bias and stochastic error is ~100% (removable) variance, the gap between the two methods can only widen in favor of stochastic quantization as R increases — exactly what is observed (`plot6_bias_variance_decomposition_bits2.png`: the deterministic blue line is perfectly flat; the stochastic curves — bias², variance, total MSE — all fall in a straight line on the log-log scale).

## Measured consequences on error and accuracy

| Bits | R | Deterministic MSE | Stochastic MSE | Deterministic accuracy | Stochastic accuracy |
|---|---|---|---|---|---|
| 2 | 1  | 0.1001 | 0.1992 | 0.842 | 0.829 ± 0.008 |
| 2 | 32 | 0.1001 | **0.0062** | 0.842 | **0.856** ± 0.003 |
| 4 | 1  | 0.0040 | 0.0079 | 0.852 | 0.857 ± 0.003 |
| 4 | 32 | 0.0040 | **0.0002** | 0.852 | **0.857** ± 0.001 |
| 8 | 1  | 0.00001 | 0.00003 | 0.856 | 0.857 |
| 8 | 32 | 0.00001 | 0.00000 | 0.856 | 0.857 |

(full table with bias/variance: `results/summary_table.csv`; raw decomposition: `results/bias_variance.csv`)

- Since the deterministic bias never moves, its MSE stays stuck at 0.1001 (at 2 bits) regardless of R.
- Stochastic MSE starts higher (noise without averaging) but collapses with R, since it has (almost) only variance to eliminate: **16× better than deterministic at R=32** (2 bits).
- This translates into real accuracy gains: +1.4 points at 2 bits, +0.5 points at 4 bits; negligible at 6-8 bits, where compression is already fine enough that the error (from either method) barely affects predictions.

## Limitations

- Only one cut layer tested (8×8×64) on one small CNN — not yet generalized to other architectures/depths.
- The replica × seed grid was computed on a fixed subset of 1,000 images (not the full 10,000), for compute-cost reasons.
- The reported stochastic "bias²" is a Monte Carlo estimate over only 10 seeds (not the true theoretical bias, which is exactly zero by construction) — see discussion above.

## Natural next step for the project

Deterministic quantization has no variance to exploit: its replicas are identical, so a Byzantine attacker cannot hide behind "legitimate disagreement" since there is none. Stochastic quantization, on the other hand, introduces an honest divergence between replicas (measured in `replica_grid_raw.csv`, `divergence` column) that grows with compression — this is exactly the tolerance margin a robust consensus protocol will have to accept between honest replicas, and therefore the potential stealth budget for a Byzantine attacker. This is the subject of the next step (not covered here).
