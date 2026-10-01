# Deterministic Quantization Creates an Aggregation Bias That Averaging Cannot Remove — Stochastic Quantization Does Not

**Research question:** when several honest replicas compress the same activation before sending it to a server, does *deterministic* quantization introduce an aggregation bias that persists no matter how many replicas are averaged, while *unbiased stochastic* quantization introduces (almost) none?

**Short answer: yes, demonstrated numerically, not just assumed.**

## Method (in brief)

Small CNN trained on CIFAR-10 (86.6% clean accuracy), cut after the 2nd convolutional block (activation `a`, shape 64×8×8). Deterministic vs. unbiased stochastic uniform quantization, same grid for both, tested at 8/6/4/3/2 bits with 1–32 replicas averaged, each stochastic point repeated over 10 seeds. For every (bits, R, method), I compute the **bias²** and **variance** of the averaged reconstruction separately and check the identity `MSE = bias² + variance` numerically. Code + raw CSVs: [github.com/Ksentissi/split-quant-experiment](https://github.com/Ksentissi/split-quant-experiment).

## Main proof: bias² / variance decomposition

Checked across 30 configurations, `bias² + variance` matches the actually measured MSE to within **4.16 × 10⁻⁹** — an exact identity, not an approximation.

**Example at 2 bits:**

| R | Det. bias² | Det. variance | Stoch. bias² | Stoch. variance |
|---|---|---|---|---|
| 1  | 0.1001 | **0.0 (exactly)** | 0.0200 | 0.1793 |
| 32 | 0.1001 | **0.0 (exactly)** | 0.0006 | 0.0056 |

(full table for all bit rates: `results/bias_variance.csv`, `plot6_bias_variance_decomposition_bits{2,3,4,6,8}.png`)

Two facts, measured rather than assumed:

1. **Variance across deterministic replicas is exactly 0.000 × 10⁰ at every bit rate** (verified empirically, not assumed — `Q_d` is a pure function of `a`, so there is no randomness to average out). So `ā_R = Q_d(a)` for every R, and its error is **strictly constant in R**.
2. **Stochastic error is almost entirely variance, not bias** — the true bias is exactly 0 by construction (`E[Q_s(a)] = a`); the tiny residual bias² measured is Monte Carlo noise from only 10 seeds, and it shrinks with R too, confirming convergence toward zero rather than contradicting it. Variance itself shrinks as 1/R with averaging.

**Consequence:** deterministic error is 100% non-removable bias; stochastic error is ~100% removable variance. The gap can only widen in stochastic's favor as R grows — confirmed on `plot1`/`plot2` (2 bits): stochastic+averaging goes from 2× worse (R=1) to **16× better in MSE and +1.4 accuracy points** (R=32) than deterministic. The advantage is negligible at 6–8 bits, where quantization error is already small enough not to matter.

## Limitation

Tested on one architecture / one cut layer, and the replica×seed grid uses a fixed 1,000-image subset for compute reasons — happy to extend this if needed before we build on it.

## Why I'm choosing stochastic compression going forward

I'm aware this decomposition specifically isolates the *aggregation* step, not the end-to-end system. Still, this is exactly why stochastic compression is the right choice for the next phase of the project: it is what lets me actually study the central phenomenon we care about — the legitimate, honest inter-replica divergence that a robust consensus protocol has to tolerate, and which later defines the stealth budget available to a Byzantine attacker. Deterministic compression has no such divergence to study (replicas are identical), so it can't serve as a basis for that analysis. For that reason, I'll move forward with stochastic compression.
