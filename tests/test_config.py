"""Tests for the typed configuration schema."""

from pathlib import Path

import pytest

from cvttcn.config import Config, DataConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_match_locked_decisions():
    """Defaults must encode the chosen task: binary imagined L/R fist, pooled split."""
    cfg = Config()
    assert cfg.data.runs == [4, 8, 12]
    assert cfg.data.excluded_subjects == [88, 89, 92, 100]
    assert cfg.data.sfreq == 160.0
    assert cfg.data.n_channels == 64
    assert cfg.model.num_classes == 2
    assert cfg.model.in_channels == 1


def test_default_yaml_matches_dataclass_defaults():
    """configs/default.yaml must stay in sync with the dataclass defaults."""
    loaded = Config.from_yaml(CONFIG_DIR / "default.yaml")
    assert loaded.to_dict() == Config().to_dict()


def test_roundtrip_to_and_from_dict():
    cfg = Config()
    assert Config.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_partial_override_only_touches_given_fields():
    cfg = Config.from_dict({"train": {"lr": 0.01, "batch_size": 32}})
    assert cfg.train.lr == 0.01
    assert cfg.train.batch_size == 32
    # Untouched fields keep their defaults.
    assert cfg.train.epochs == Config().train.epochs
    assert cfg.data.runs == [4, 8, 12]


def test_nested_dataclass_is_built_from_dict():
    cfg = Config.from_dict({"model": {"cvt": {"depth": 6, "num_heads": 4}}})
    assert cfg.model.cvt.depth == 6
    assert cfg.model.cvt.num_heads == 4
    # Sibling nested branch keeps defaults.
    assert cfg.model.tcn.kernel_size == 3


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"data": {"not_a_real_field": 1}})


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"nonsense": {}})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"data": {"val_size": 0.6, "test_size": 0.6}}, "leave room"),
        ({"data": {"normalize": "minmax"}}, "normalize"),
        ({"data": {"l_freq": 30.0, "h_freq": 8.0}}, "h_freq"),
        ({"data": {"tmin": 4.0, "tmax": 0.0}}, "tmax"),
        ({"model": {"num_classes": 1}}, "num_classes"),
        ({"train": {"device": "tpu"}}, "device"),
    ],
)
def test_validate_rejects_bad_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        Config.from_dict(overrides).validate()


def test_load_config_without_path_returns_validated_defaults():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.data == DataConfig()


def test_yaml_roundtrip_on_disk(tmp_path):
    out = tmp_path / "cfg.yaml"
    Config().to_yaml(out)
    assert Config.from_yaml(out).to_dict() == Config().to_dict()
