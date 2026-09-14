"""Corre los controles ch1-only y silence-only de C3 sobre el corpus combinado quemado."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_bundle_c3_bc import feature_order_without_static_means, fit_c3_corpus
from build_c3_bc_corpus import PRIVATE_ROOT, build_corpus_index, extract_corpus
from run_experiment import load_spec

EXPERIMENTS = ROOT / "configs" / "experiments"
CONTROL_SPECS = {
    "ch1_only_lfcc": EXPERIMENTS / "a7_lfcc_ch1_v1.yaml",
    "silence_only_lfcc": EXPERIMENTS / "a7_lfcc_ch0_silence_v1.yaml",
}


def run_controls() -> dict:
    index = build_corpus_index()
    results = {}
    for name, spec_path in CONTROL_SPECS.items():
        spec = load_spec(spec_path)
        full_order = tuple(spec["feature_order"])
        corpus = extract_corpus(
            index,
            spec,
            private_root=PRIVATE_ROOT / "controls" / name,
            feature_order=full_order,
        )
        fitted = fit_c3_corpus(corpus, feature_order_without_static_means(full_order))
        results[name] = {
            "n": len(fitted["y"]),
            "n_features": len(fitted["export"].feature_order),
            "oof_auc": fitted["oof_auc"],
            "oof_brier": fitted["oof_brier"],
            "threshold": fitted["threshold"],
            "score_sha256": hashlib.sha256(fitted["calibrated"].tobytes()).hexdigest(),
            "folds": fitted["folds"],
        }
    return {
        "phase": "diagnostic_controls_burned",
        "corpus_fingerprint": index.fingerprint,
        "results": results,
        "limitations": [
            "Controles OOF sobre el corpus de ajuste; no son evidencia confirmatoria.",
            "B+C no ataca explícitamente el ruido aditivo; un AUC alto debe declararse.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIVATE_ROOT / "controls.json")
    args = parser.parse_args(argv)
    if "models/current" in args.output.resolve().as_posix():
        raise SystemExit("D-A8.3 no permite escribir controles en models/current")
    report = run_controls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
