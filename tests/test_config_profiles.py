"""Tests for profile-based configuration loading."""

from __future__ import annotations

from embodied_motion_flow.config import load_config, resolve_config_path


def test_named_profile_resolves_from_configs_directory() -> None:
    path = resolve_config_path("testing")
    assert path.as_posix().endswith("configs/testing.yaml")


def test_testing_profile_inherits_base_values() -> None:
    config = load_config("testing")
    assert config.project.output_dir == "outputs/testing"
    assert config.data.input_dim == 72
    assert config.model.audio_dim == 14
    assert config.training.epochs == 1
