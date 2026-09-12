"""Compara VADs pre-registrados contra el oráculo, exclusivamente en train.

No selecciona parámetros ni escribe detalles por llamada: esos artefactos pueden contener
identificadores del dataset y pertenecen a un flujo privado separado.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from altur.features.behavioral import extract as behavioral_extract
from altur.seg import energy, webrtc
from altur.types import AudioExample, DatasetRecord, Segmentation


class VadDataset(Protocol):
    @property
    def ids(self) -> tuple[str, ...]: ...

    def record(self, example_id: str) -> DatasetRecord: ...

    def load(self, example_id: str, *, with_turns: bool = False) -> AudioExample: ...


def _lat_med(ex: AudioExample, segmentation: Segmentation) -> tuple[float, int]:
    features, diagnostics = behavioral_extract(replace(ex, seg=segmentation))
    return features["behavioral.lat_med"], int(diagnostics["n_responses"])


def compare_vads(
    dataset: VadDataset,
    *,
    split: str = "train",
    segmenters: dict[str, Callable[[AudioExample], Segmentation]] | None = None,
) -> dict[str, object]:
    """Return only aggregate agreement metrics for the frozen VAD baselines."""
    if split != "train":
        raise ValueError("calibrate_vad solo permite split='train'; val/all estan prohibidos")
    segmenters = segmenters or {"energy@1": energy, "webrtc@1": webrtc}
    expected_refs = {"energy@1", "webrtc@1"}
    if set(segmenters) != expected_refs:
        raise ValueError(f"segmenters deben ser exactamente {sorted(expected_refs)}")

    totals = {
        ref: {
            "intersection": {"ch0": 0, "ch1": 0, "total": 0},
            "union": {"ch0": 0, "ch1": 0, "total": 0},
            "count_errors": [],
            "lat_errors": [],
        }
        for ref in segmenters
    }
    calls = 0
    for example_id in dataset.ids:
        if dataset.record(example_id).split != "train":
            continue
        ex = dataset.load(example_id, with_turns=True)
        if ex.seg is None or not ex.seg.is_oracle:
            raise ValueError("cada llamada de train requiere segmentacion oracle@1")
        calls += 1
        oracle_lat_med, oracle_responses = _lat_med(ex, ex.seg)
        for ref, segment in segmenters.items():
            predicted = segment(ex)
            metrics = totals[ref]
            for channel, key in ((0, "ch0"), (1, "ch1")):
                if channel == 1 and ex.is_mono:
                    continue
                oracle_mask = ex.seg.speech_mask(channel, ex.n_samples, ex.sr)
                predicted_mask = predicted.speech_mask(channel, ex.n_samples, ex.sr)
                intersection = int((oracle_mask & predicted_mask).sum())
                union = int((oracle_mask | predicted_mask).sum())
                metrics["intersection"][key] += intersection
                metrics["union"][key] += union
                metrics["intersection"]["total"] += intersection
                metrics["union"]["total"] += union
            metrics["count_errors"].append(
                abs(len(predicted.turns) - len(ex.seg.turns))
            )
            predicted_lat_med, _ = _lat_med(ex, predicted)
            if oracle_responses:
                metrics["lat_errors"].append(abs(predicted_lat_med - oracle_lat_med))

    report: dict[str, object] = {"split": "train", "calls": calls, "segmenters": {}}
    for ref, metrics in totals.items():
        intersections = metrics["intersection"]
        unions = metrics["union"]
        lat_errors = metrics["lat_errors"]
        report["segmenters"][ref] = {
            "frame_iou": {
                key: (intersections[key] / unions[key] if unions[key] else None)
                for key in ("ch0", "ch1", "total")
            },
            "segment_count_error_mae": (
                sum(metrics["count_errors"]) / calls if calls else None
            ),
            "behavioral_lat_med_mae_s": (
                sum(lat_errors) / len(lat_errors) if lat_errors else None
            ),
            "behavioral_lat_med_calls": len(lat_errors),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="solo train; val/all estan prohibidos")
    args = parser.parse_args()
    from altur.dataset import Dataset

    print(json.dumps(compare_vads(Dataset(), split=args.split), sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
