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
- Production Kaggle tutorial notebook (`training_tutorial.ipynb`).

## Quick Start

```bash
pip install -e .
python scripts/utils/setup_local_data.py --max-motion-files 5
export AISTPP_ROOT="$(pwd)/data/aist_plusplus/motions"
export AISTPP_SPLIT_ROOT="$(pwd)/data/aist_plusplus/splits"
python scripts/data/download_aist_audio_official.py --csv-path data/all_music_wav_url.csv --agree-terms
python -m embodied_motion_flow.cli.check_audio_conditioning --config configs/base.yaml --fail-under 0.95
python run_pipeline.py train --config configs/base.yaml
python run_pipeline.py showcase --config configs/base.yaml --checkpoint outputs/checkpoints/model.pt
```

On Apple Silicon, `device.preference: auto` selects MPS when available.

## Local-To-Cloud Workflow

Local toy dataset:

```bash
python scripts/utils/setup_local_data.py --max-motion-files 5
pytest -q
```

Kaggle full dataset:

1. Open `training_tutorial.ipynb`.
2. Run all cells.
3. The notebook clones the repo, maps the Kaggle AIST++ dataset paths, runs `configs/kaggle_prod.yaml`, and writes one downloadable ZIP.

Set `EMF_REPO_URL` in Kaggle if you need to override the default repository URL.

Long-form Kaggle showcase:

```bash
python run_pipeline.py full --config configs/kaggle_prod.yaml --fresh-start --zip-path /kaggle/working/embodied_motion_flow_showcase.zip
```

The production Kaggle profile uses dense AIST++ windows, no file cap, AMP, EMA, classifier-free guidance, and sliding-window 450-frame generation. It slices the Stardust track from `0:46` to `1:01`, embeds that audio in the Viral MP4, writes the Research MP4, and packages checkpoint, videos, metrics, plots, logs, and manifest into one ZIP.

Config profiles:

- `configs/base.yaml`: default local research configuration.
- `configs/kaggle_prod.yaml`: full Kaggle training/showcase profile.
- `configs/testing.yaml`: small deterministic profile for fast checks.

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

## Notebook

Run `training_tutorial.ipynb` on Kaggle from top to bottom. It installs runtime dependencies, maps the AIST++ dataset paths, trains with `configs/kaggle_prod.yaml`, renders the 15-second Stardust showcase, and exposes a single downloadable ZIP.

## Reproducibility

All executable entrypoints:

- Load a profile from `configs/`.
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
