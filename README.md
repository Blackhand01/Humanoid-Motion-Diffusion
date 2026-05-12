# Embodied-Motion-Flow

Embodied-Motion-Flow is a research-grade diffusion pipeline for generating and reconstructing humanoid joint trajectories with explicit biomechanical regularization. The current research path targets AIST++ SMPL motion with 24 joints and 72 axis-angle channels.

The repository includes:

- Synthetic trajectory engine with nominal and anomalous motions for quick tests.
- AIST++ SMPL loader for 24-joint / 72D real motion clips.
- Audio feature extraction for tempo, beat positions, chroma conditioning, and strict cross-modal coverage checks.
- 1D temporal Transformer denoiser and audio-conditioned cross-attention denoiser.
- DDPM forward/reverse scheduler with deterministic reconstruction path, classifier-free guidance, and EMA inference.
- Biomechanical consistency loss (joint limits, acceleration, temporal jitter).
- End-to-end training with AMP, gradient clipping, accumulation, warmup+cosine LR scheduling, checkpoint resume, evaluation, plotting, and animation outputs.
- Colab/Kaggle runnable notebook (`main_colab.ipynb`).

## Quick Start

```bash
pip install -r requirements.txt
python setup_local_data.py --max-motion-files 5
export AISTPP_ROOT="$(pwd)/data/aist_plusplus/motions"
export AISTPP_SPLIT_ROOT="$(pwd)/data/aist_plusplus/splits"
python download_aist_audio_official.py --csv-path data/all_music_wav_url.csv --agree-terms
python -m embodied_motion_flow.cli.check_audio_conditioning --config config.yaml --fail-under 0.95
python -m embodied_motion_flow.cli.train --config config.yaml
python -m embodied_motion_flow.cli.evaluate --config config.yaml --checkpoint outputs/checkpoints/model.pt
```

On Apple Silicon, `device.preference: auto` selects MPS when available.

## Local-To-Cloud Workflow

Local toy dataset:

```bash
python setup_local_data.py --max-motion-files 5
pytest -q
```

Kaggle full dataset:

1. Open `kaggle_bridge.ipynb`.
2. Run all cells.
3. The notebook clones the repo, downloads official AIST++ motions/splits, verifies files under `/kaggle/working/data/aist_plusplus/motions`, writes a Kaggle config, and launches training.

Set `EMF_REPO_URL` in Kaggle if you need to override the default repository URL.

Long-form Kaggle showcase:

```bash
python kaggle_showcase_main.py --config config.kaggle.full.yaml --fresh-start
```

The Kaggle notebook writes `config.kaggle.full.yaml` with `sequence_length=120`, dense AIST++ windows, no file cap, and a fresh output directory. It trains the model, slices the Stardust track from `0:46` to `1:01`, generates `450` frames with EMA + classifier-free guidance, and writes viral/research MP4 renders under `/kaggle/working/outputs/showcase/`. Use `--skip-train --checkpoint outputs/checkpoints/model.pt` to render from an existing checkpoint.

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
│   ├── evaluation_metrics.json
│   └── evaluation_report.csv
├── previews/
│   └── hero_validation.mp4
├── showcase/
│   ├── stardust_0046_0101_viral.mp4
│   └── stardust_0046_0101_research.mp4
└── logs/
    ├── train.log
    └── evaluate.log
```

## Evaluation Metrics

- MSE Reconstruction
- Temporal Smoothness
- AUROC for anomaly classification
- Physical constraint violation ratio
- Beat Alignment Score for generated motion and reference motion

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
- CFG sampling, EMA weight swapping, and sliding-window generation.

## Sim-to-Real Notes

See [`docs/sim_to_real.md`](docs/sim_to_real.md) for deployment safety constraints and sim-to-real limitations.
