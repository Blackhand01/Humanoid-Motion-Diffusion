#!/usr/bin/env bash
set -euo pipefail

python -m embodied_motion_flow.cli.train --config configs/base.yaml
