"""
Tarea 6 del prompt de Persona 1 (Regi): el contrato de datos con el endpoint
de Joaquin. El reto en si ya fija el formato exacto en su README:

    POST /detect
    -> {"is_synthetic": true, "confidence": 0.87}

Este script es la pieza que CIERRA ese contrato: toma el score calibrado
(Platt) de la rama que el equipo elija como final y lo convierte en ese JSON
exacto, sin que Joaquin tenga que adivinar nada del lado de calibracion.

"confidence" se define como la probabilidad calibrada de la clase PREDICHA
(no siempre P(synthetic)) -- si is_synthetic=True, confidence=P(synthetic);
si is_synthetic=False, confidence=P(human)=1-P(synthetic). Es la convencion
mas comun para este tipo de campo y la que mejor "premia calibracion" como
pide el README del reto.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PARAMS_PATH = DATA_DIR / "platt_params_production.json"
DEFAULT_BRANCH = "fusion"  # cual score usar por default para el endpoint final


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_platt_params(branch=DEFAULT_BRANCH):
    if not PARAMS_PATH.exists():
        raise SystemExit(f"No existe {PARAMS_PATH}. Corre primero calibrate.py")
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    if branch not in params:
        raise SystemExit(f"No hay parametros de Platt para la rama '{branch}' en {PARAMS_PATH}")
    return params[branch]


def raw_score_to_calibrated(raw_score, platt_params, eps=1e-6):
    """Aplica la MISMA transformacion que calibrate.py: logit del score crudo,
    luego sigmoid(A*logit + B). Debe coincidir exactamente con fit_platt_oof /
    fit_platt_production en calibrate.py -- si se toca uno, se toca el otro."""
    p = np.clip(raw_score, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return float(sigmoid(platt_params["A"] * logit + platt_params["B"]))


def to_detect_response(raw_score, platt_params, threshold=0.5):
    """raw_score: score crudo (probabilidad, no calibrada) de la rama elegida,
    para UNA llamada. Devuelve el dict exacto que espera POST /detect."""
    p_synthetic = raw_score_to_calibrated(raw_score, platt_params)
    is_synthetic = p_synthetic >= threshold
    confidence = p_synthetic if is_synthetic else (1.0 - p_synthetic)
    return {"is_synthetic": bool(is_synthetic), "confidence": round(confidence, 4)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    platt_params = load_platt_params(args.branch)
    scores_path = DATA_DIR / f"scores_{args.branch}_calibrated.csv"
    if not scores_path.exists():
        raise SystemExit(f"No existe {scores_path}. Corre primero calibrate.py")

    df = pd.read_csv(scores_path)
    responses = []
    for _, row in df.iterrows():
        resp = to_detect_response(row["score"], platt_params, threshold=args.threshold)
        responses.append({"anon_id": row["anon_id"], "label_real": row["label"], **resp})

    out_path = DATA_DIR / f"detect_responses_{args.branch}.jsonl"
    with open(out_path, "w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")

    resp_df = pd.DataFrame(responses)
    correct = (resp_df["is_synthetic"] == (resp_df["label_real"] == "synthetic")).mean()
    print(f"Ejemplos ({args.branch}, threshold={args.threshold}):")
    print(resp_df.head(5).to_string(index=False))
    print(f"\nExactitud @ threshold={args.threshold}: {correct*100:.2f}%")
    print(f"Guardado en {out_path} ({len(resp_df)} respuestas en formato POST /detect)")


if __name__ == "__main__":
    main()
