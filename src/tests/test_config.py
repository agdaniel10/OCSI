"""Tests for configuration loading and ablation presets."""
import pytest

from ocsi.config import OCSIConfig, apply_ablation


def test_defaults():
    cfg = OCSIConfig()
    assert cfg.memory.appearance_ema_alpha == 0.9
    assert cfg.association.w_app == 0.30
    assert cfg.behaviour.enabled is False
    assert cfg.perception.reid_input_hw == (256, 128)


def test_from_dict_nested_override_keeps_other_defaults():
    cfg = OCSIConfig.from_dict({"memory": {"min_hits": 5}, "behaviour": {"enabled": True}})
    assert cfg.memory.min_hits == 5
    assert cfg.memory.appearance_ema_alpha == 0.9  # untouched
    assert cfg.behaviour.enabled is True


def test_from_dict_coerces_list_to_tuple():
    cfg = OCSIConfig.from_dict({"perception": {"reid_input_hw": [128, 64]}})
    assert cfg.perception.reid_input_hw == (128, 64)


def test_dict_roundtrip():
    cfg = OCSIConfig()
    assert OCSIConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_ablation_stages():
    base = apply_ablation(OCSIConfig(), "baseline")
    assert base.association.use_memory is False and base.behaviour.enabled is False
    mem = apply_ablation(OCSIConfig(), "memory")
    assert mem.association.use_memory is True and mem.behaviour.enabled is False
    fb = apply_ablation(OCSIConfig(), "feedback")
    assert fb.association.use_memory is True and fb.behaviour.enabled is True


def test_ablation_does_not_mutate_source():
    cfg = OCSIConfig()
    apply_ablation(cfg, "feedback")
    assert cfg.behaviour.enabled is False  # original untouched


def test_ablation_invalid_stage():
    with pytest.raises(ValueError):
        apply_ablation(OCSIConfig(), "nonsense")
