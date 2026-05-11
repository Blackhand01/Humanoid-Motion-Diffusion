# Embodied-Motion-Flow

Embodied-Motion-Flow is a research-grade diffusion pipeline for generating and reconstructing 12-DOF humanoid joint trajectories with explicit biomechanical regularization.

The repository includes:

- Synthetic trajectory engine with nominal and anomalous motions.
- 1D temporal Transformer denoiser with timestep and positional encodings.
- DDPM forward/reverse scheduler with deterministic reconstruction path.
- Biomechanical consistency loss (joint limits, acceleration, temporal jitter).
- End-to-end training, checkpointing, evaluation, plotting, and animation outputs.
- Colab/Kaggle runnable notebook (`main_colab.ipynb`).

## Quick Start

```bash
pip install -r requirements.txt
python -m embodied_motion_flow.cli.train --config config.yaml
python -m embodied_motion_flow.cli.evaluate --config config.yaml --checkpoint outputs/checkpoints/model.pt
```

## Outputs

After training/evaluation, artifacts are written to:

```txt
outputs/
├── checkpoints/
│   ├── model.pt
│   └── model_epoch_*.pt
├── plots/
│   ├── training_loss.png
│   ├── biomechanical_loss.png
│   └── smoothness.png
├── animations/
│   ├── denoising.gif
│   └── trajectory.mp4
├── metrics/
│   └── evaluation_metrics.json
└── logs/
    ├── train.log
    └── evaluate.log
```

## Evaluation Metrics

- MSE Reconstruction
- Temporal Smoothness
- AUROC for anomaly classification
- Physical constraint violation ratio

No mAP metric is used.

## Notebook (Colab + Kaggle)

Run `main_colab.ipynb` from top to bottom. It:

1. Installs dependencies.
2. Detects GPU (CUDA/CPU fallback).
3. Loads config and sets deterministic seeds.
4. Trains the diffusion model.
5. Saves checkpoint and plots.
6. Runs evaluation.
7. Displays generated PNG plots, GIF, and MP4 inline.

The notebook is designed for:

- Google Colab T4.
- Kaggle P100 or T4x2.

## Reproducibility

All executable entrypoints:

- Load `config.yaml`.
- Set deterministic seeds for Python, NumPy, and PyTorch.
- Log active configuration and selected device.

Deterministic PyTorch settings are enabled by default. This can reduce throughput versus nondeterministic kernels.

## Tests

```bash
pytest -q
```

Included tests cover:

- Diffusion scheduler construction.
- Forward noising step.
- Reverse denoising step determinism and shape.
- Biomechanical loss components.

## Sim-to-Real Notes

See [`docs/sim_to_real.md`](docs/sim_to_real.md) for deployment safety constraints and sim-to-real limitations.
