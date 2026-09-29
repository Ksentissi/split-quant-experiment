# Stochastic vs. deterministic quantization in replicated split inference

Preliminary (pre-Byzantine) experiment: does averaging independent
stochastic quantizations across honest replicas ever beat deterministic
quantization, at the same per-replica bit budget?

## Files

- `model.py` — `SplitCNN`, a small CNN split into `client_layers` (client
  replicas) and `server_layers` + `classifier` (server), cut at a fixed
  8x8x64 activation.
- `compressors.py` — no-compression baseline is implicit; deterministic
  uniform quantizer, unbiased stochastic uniform quantizer, and
  `make_generator` for independent per-replica randomness.
- `data_utils.py` — CIFAR-10 loaders and device selection.
- `metrics.py` — MSE, relative error, pairwise inter-replica divergence,
  mean/std/CI, paired significance test.
- `train_model.py` — trains `SplitCNN` once on CIFAR-10 and saves a
  checkpoint. This is the only training run; every compression experiment
  reuses these exact weights.
- `run_experiment.py` — runs the full grid (5 bit rates × 6 replica
  counts × 10 seeds) and writes CSVs to `results/`.
- `plotting.py` — reads the CSVs and produces the 5 requested plots.

## How to run

```bash
cd split_quant_experiment
pip install torch torchvision matplotlib

# 1. Train the model once (≈10-20 min on CPU, faster on GPU/MPS)
python train_model.py --epochs 18

# 2. Run the full compression experiment (a few minutes)
python run_experiment.py --checkpoint split_cnn_cifar10.pt

# 3. Generate the plots
python plotting.py --results-dir results
```

Outputs land in `results/`:
- `simple_accuracy_vs_bits.csv` — uncompressed / deterministic / stochastic
  (R=1) accuracy on the full 10,000-image test set.
- `replica_grid_raw.csv` — per (method, bits, R, seed) MSE / relative
  error / divergence / accuracy, on a fixed random subset of 1000 test
  images (the full grid is too expensive to run on all 10,000 images with
  10 seeds x 32 replicas; the subset is fixed and identical for every
  method/bit/replica so the comparison stays fair).
- `bias_variance.csv` — per (bits, R) bias and variance of the stochastic
  reconstruction, estimated across the 10 seeds.
- `summary_table.csv` — the headline table (bits, replicas, deterministic
  vs stochastic MSE/accuracy, honest divergence, paired significance test).
- `plot1..plot5_*.png` — the 5 requested figures.

## Reproducibility

All randomness is seeded: `BASE_SEED = 0` in `run_experiment.py`, and every
stochastic quantization call gets its own `torch.Generator` derived from
`(seed, bits, replica_index)` (see `compressors.make_generator`), so no two
replicas or seeds ever share the same random stream.

## After running

Once `results/summary_table.csv` exists, the short interpretation
(`REPORT.md`) is filled in from the *actual* numbers in that file — no
numbers are invented ahead of running the code.
