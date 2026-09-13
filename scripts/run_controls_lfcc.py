"""A3.6 sobre LFCC (D-A7.6): ¿qué separa las clases cuando no debería?

La familia se declara aquí y se commitea ANTES de correr. Tres controles con el mismo pipeline
LFCC que C2, cada uno sobre otra región del audio: el canal del agente, el silencio del caller y,
como referencia, el habla del caller. Solo `train`, `official_v1`, cross-fitting anidado con la
regresión de C2. No lee `val`.

    python scripts/run_controls_lfcc.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from altur.controls import Hypothesis, HypothesisFamily, format_controls, run_controls
from altur.crossfit import Branch, FeatureTable, make_nested_fit_predict
from altur.dataset import Dataset
from altur.models.adapters import logreg
from altur.models.calibration import PlattCalibrator
from altur.protocol import load_groups, nested_cv, official_v1
from altur.provenance import effective_code_digest
from build_bundle import _commit, feature_table
from run_experiment import load_spec

EXPERIMENTS = ROOT / "configs" / "experiments"
REPORT = ROOT / "experiments" / "reports" / "a7_controls_lfcc_v1.json"
LR_PARAMS = {"class_weight": "balanced", "max_iter": 1000}

FAMILY = HypothesisFamily(
    name="A3.6_confound_lfcc_v1",
    hypotheses=(
        Hypothesis(
            "ch1_only_lfcc",
            "¿El LFCC del canal del AGENTE (mismo TTS en ambas clases) separa las clases?",
            (
                "C2 mide propiedades de la voz del caller.",
                "La separación de C2 es atribuible a la síntesis y no a la cadena de grabación.",
            ),
        ),
        Hypothesis(
            "silence_only_lfcc",
            "¿El LFCC de los tramos SIN habla del caller separa las clases?",
            (
                "La separación de C2 viene del habla del caller.",
                "C2 transfiere a un set con la cadena de grabación normalizada.",
            ),
        ),
        Hypothesis(
            "clean_speech_lfcc",
            "Referencia: ¿el LFCC del habla del caller (C2) separa las clases?",
            (),
        ),
    ),
)
SPECS = {
    "ch1_only_lfcc": EXPERIMENTS / "a7_lfcc_ch1_v1.yaml",
    "silence_only_lfcc": EXPERIMENTS / "a7_lfcc_ch0_silence_v1.yaml",
    "clean_speech_lfcc": EXPERIMENTS / "a7_spectral_factory_lfcc_v1.yaml",
}
# D-A3.6, para leer lado a lado. No entra a la corrección: es otra familia.
BASELINE_D_A3_6 = {
    "ch1_only": {"auc": 0.6409, "scale_auc": 0.5000},
    "silence_only": {"auc": 0.9662, "scale_auc": 0.0157},
    "clean_speech": {"auc": 0.9791, "scale_auc": 0.0367},
}
DIAG_FLAGS = ("vad_fallback_full_channel", "insufficient_audio")


def diagnostic_rates(run_id: str, ids: list[str]) -> dict[str, dict[str, float]]:
    """Tasa de cada bandera de diagnóstico por clase (1 = sintética). Sin ids en la salida."""
    art = ROOT / ".private" / "artifacts" / run_id
    counts = {flag: {0: [0, 0], 1: [0, 0]} for flag in DIAG_FLAGS}
    for example_id in ids:
        name = hashlib.sha256(example_id.encode()).hexdigest()
        payload = json.loads((art / f"{name}.json").read_text(encoding="utf-8"))
        label = int(payload["label"])
        for flag in DIAG_FLAGS:
            counts[flag][label][0] += int(bool(payload["diagnostics"].get(flag)))
            counts[flag][label][1] += 1
    return {
        flag: {"human": c[0][0] / max(c[0][1], 1), "synthetic": c[1][0] / max(c[1][1], 1)}
        for flag, c in counts.items()
    }


def main() -> int:
    commit = _commit()
    code_digest = effective_code_digest(ROOT)
    ds = Dataset()
    protocol = official_v1(ds)
    groups = load_groups()

    observations, units, rates = {}, {}, {}
    for name in FAMILY.names:
        spec = load_spec(SPECS[name])
        order = tuple(spec["feature_order"])
        features, labels, run_id = feature_table(spec, ds, code_digest, commit)
        table = FeatureTable(features, labels)
        branch = Branch(name, order, lambda o: logreg(o, **LR_PARAMS))
        nested = nested_cv(
            ds, protocol,
            make_nested_fit_predict(table, [branch], calibrator_factory=PlattCalibrator, inner_folds=3),
            groups,
        )
        observations[name] = (np.asarray(nested.y_true), np.asarray(nested.y_score))
        units[name] = list(nested.ids)
        rates[name] = diagnostic_rates(run_id, list(features))

    results = run_controls(FAMILY, observations, units=units)
    print(format_controls(results))
    print("\nTasas de diagnóstico por clase (humana / sintética):")
    for name, flags in rates.items():
        texto = "  ".join(f"{f}: {v['human']:.3f} / {v['synthetic']:.3f}" for f, v in flags.items())
        print(f"  {name:<20} {texto}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "family": FAMILY.name,
        "declared_at": FAMILY.declared_at,
        "decision": "D-A7.6",
        "commit": commit,
        "protocol": protocol.name,
        "split": "train",
        "model": {"logreg": LR_PARAMS, "calibrator": "platt@1", "inner_folds": 3},
        "results": [r.to_dict() for r in results],
        "diagnostic_rates_by_class": rates,
        "baseline_acoustic_D-A3.6": BASELINE_D_A3_6,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
