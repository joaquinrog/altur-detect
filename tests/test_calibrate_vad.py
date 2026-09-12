from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import numpy as np
import pytest

from altur.types import AudioExample, DatasetRecord, Segmentation, Turn


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_vad.py"
    spec = importlib.util.spec_from_file_location("calibrate_vad", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self) -> None:
        oracle = Segmentation(
            (Turn(0, 0.1, 0.3), Turn(1, 0.5, 0.7)), source="oracle@1"
        )
        self.examples = {
            "train-call": AudioExample(np.zeros(8000), np.zeros(8000), seg=oracle),
            "val-call": AudioExample(np.zeros(8000), np.zeros(8000), seg=oracle),
        }
        self.records_by_id = {
            "train-call": DatasetRecord("train-call", split="train"),
            "val-call": DatasetRecord("val-call", split="val"),
        }
        self.loaded: list[str] = []

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.examples)

    def record(self, example_id: str) -> DatasetRecord:
        return self.records_by_id[example_id]

    def load(self, example_id: str, *, with_turns: bool = False) -> AudioExample:
        assert with_turns
        self.loaded.append(example_id)
        return self.examples[example_id]


def test_compare_vads_reports_train_aggregate_without_ids() -> None:
    module = _script_module()
    dataset = FakeDataset()

    def energy(ex: AudioExample) -> Segmentation:
        return Segmentation((Turn(0, 0.1, 0.3), Turn(1, 0.5, 0.6)), source="energy@1")

    def webrtc(ex: AudioExample) -> Segmentation:
        return Segmentation((), source="webrtc@1")

    report = module.compare_vads(dataset, segmenters={"energy@1": energy, "webrtc@1": webrtc})

    assert dataset.loaded == ["train-call"]
    assert report["split"] == "train"
    assert report["calls"] == 1
    assert report["segmenters"]["energy@1"] == {
        "frame_iou": {"ch0": 1.0, "ch1": 0.5, "total": 0.75},
        "segment_count_error_mae": 0.0,
        "behavioral_lat_med_mae_s": 0.0,
        "behavioral_lat_med_calls": 1,
    }
    assert report["segmenters"]["webrtc@1"]["frame_iou"] == {
        "ch0": 0.0,
        "ch1": 0.0,
        "total": 0.0,
    }
    assert report["segmenters"]["webrtc@1"]["segment_count_error_mae"] == 2.0
    assert report["segmenters"]["webrtc@1"]["behavioral_lat_med_mae_s"] == 0.2
    assert "train-call" not in str(report)
    assert "val-call" not in str(report)


@pytest.mark.parametrize("split", ["val", "all"])
def test_compare_vads_explicitly_rejects_val_and_all(split: str) -> None:
    module = _script_module()

    with pytest.raises(ValueError, match="solo permite split='train'"):
        module.compare_vads(FakeDataset(), split=split)


def test_vad_config_is_pre_registered_and_contains_no_ids() -> None:
    import json

    config_path = Path(__file__).parents[1] / "configs" / "vad_v1.json"
    config = json.loads(config_path.read_text())

    assert config["version"] == 1
    assert config["origin"] == "pre_registered"
    assert config["tuned_on_current_train"] is False
    assert config["segmenters"]["energy@1"] == {
        "threshold_db_above_noise_floor": 12,
        "pad_ms": 30,
        "merge_gap_ms": 450,
        "min_length_ms": 200,
    }
    assert config["segmenters"]["webrtc@1"] == {
        "mode": 2,
        "frame_ms": 20,
        "padding_ms": 300,
        "hysteresis": 0.9,
    }
    assert "anon_id" not in config_path.read_text()


def test_train_dependencies_include_webrtcvad_runtime() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    assert "setuptools>=68,<81" in pyproject["project"]["optional-dependencies"]["train"]
