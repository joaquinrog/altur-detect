from __future__ import annotations

import numpy as np

from altur.types import AudioExample, Segmentation, Turn


def example(*turns: Turn, seconds: float = 6.0) -> AudioExample:
    return AudioExample(
        ch0=np.zeros(int(seconds * 8000), dtype=np.float32),
        ch1=np.zeros(int(seconds * 8000), dtype=np.float32),
        seg=Segmentation(tuple(turns), source="vad@fixture"),
    )


def test_latency_uses_merged_turns_and_median_response_delay() -> None:
    from altur.features.behavioral import extract

    features, diagnostics = extract(
        example(
            Turn(0, 0.0, 1.0),
            Turn(0, 1.0, 2.0),  # contiguous caller turns are one turn
            Turn(1, 2.5, 3.0),
            Turn(0, 4.0, 5.0),
            Turn(1, 5.5, 6.0),
        )
    )

    assert features == {"behavioral.lat_med": 0.5, "behavioral.caller_bargein_rate": 0.0}
    assert diagnostics["n_caller_turns"] == 2
    assert diagnostics["n_responses"] == 2


def test_barge_in_is_caller_start_during_agent_turn_and_not_first_intervention() -> None:
    from altur.features.behavioral import extract

    features, diagnostics = extract(
        example(
            Turn(0, 0.0, 1.0),
            Turn(1, 1.2, 2.0),
            Turn(0, 1.5, 2.5),  # starts during agent speech: interruption
            Turn(1, 3.0, 4.0),  # response to caller after its end
        )
    )

    assert features["behavioral.caller_bargein_rate"] == 1.0
    # Mediana de las respuestas 0.2 s (primer caller) y 0.5 s (barge-in).
    assert features["behavioral.lat_med"] == 0.35
    assert diagnostics["n_bargeins"] == 1


def test_no_response_and_missing_segmentation_are_finite_and_diagnosed() -> None:
    from altur.features.behavioral import extract

    no_response, no_response_diag = extract(example(Turn(0, 0.0, 1.0)))
    missing, missing_diag = extract(
        AudioExample(np.zeros(8000, dtype=np.float32), np.zeros(8000, dtype=np.float32))
    )

    assert no_response["behavioral.lat_med"] == 0.0
    assert no_response["behavioral.caller_bargein_rate"] == 0.0
    assert no_response_diag["n_responses"] == 0
    assert missing == {
        "behavioral.lat_med": 0.0,
        "behavioral.caller_bargein_rate": 0.0,
    }
    assert missing_diag["segmentation_missing"] is True


def test_module_registers_extractor_with_segmentation_requirement() -> None:
    import altur.features.behavioral  # noqa: F401
    from altur.registry import extractors

    meta = extractors.meta("behavioral@1")
    assert meta["needs_seg"] is True
    assert meta["channels"] == (0, 1)
