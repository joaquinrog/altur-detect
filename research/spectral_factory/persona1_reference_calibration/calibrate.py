"""
Calibracion de Platt -- implementacion de REFERENCIA para la Tarea 3 de Persona
1 (Regi). Ver README.md de esta carpeta: esto es un placeholder documentado
para no bloquear al equipo mientras Persona 1 no este disponible, no su
entregable final.

Que problema resuelve calibrar probabilidades: un clasificador puede "decir"
90% de confianza y estar mal la mitad de las veces si sus scores no estan
calibrados -- el score crudo de una regresion logistica o de un backbone no
es automaticamente una probabilidad verdadera, solo un numero que ordena bien
los casos (bueno para AUC/EER, no necesariamente para "confidence").

Que hace la escala de Platt: ajusta una regresion logistica de 1 variable
(el score crudo) contra el label verdadero, aprendiendo una curva sigmoide
(A, B) que mapea score -> probabilidad calibrada = sigmoid(A*score + B). Es
literalmente "la misma regresion logistica" que ya se usa en las ramas, pero
aplicada como una capa de post-procesamiento sobre 1 sola variable en vez de
sobre el vector de features original.

Contrato de entrada (ver persona3_prosodic/CONTRATOS.md seccion 4):
    anon_id,label,fold,branch,score
Contrato de salida (mismo archivo + una columna nueva):
    anon_id,label,fold,branch,score,score_calibrated
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "persona3_prosodic" / "data"
OUT_DIR = Path(__file__).parent / "data"

BRANCH_SCORE_FILES = {
    "prosodic": DATA_DIR / "scores_prosodic.csv",
    "spectral": DATA_DIR / "scores_spectral.csv",
    "fusion": DATA_DIR / "scores_fusion.csv",
}


def _to_logit(p, eps=1e-6):
    """Los scores ya son probabilidades (salen de predict_proba), no logits.
    Para que Platt tenga margen donde aprender algo (en vez de una identidad
    casi perfecta cuando el score ya esta en [0,1]), se ajusta sobre el logit
    del score crudo -- es la variante estandar de Platt scaling para inputs que
    ya vienen en escala de probabilidad, no en escala de logit sin acotar."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt_oof(df):
    """Ajusta Platt scaling RESPETANDO los folds ya asignados (mismos folds de
    data/splits.csv): para cada fold, entrena el calibrador SOLO con las otras
    4, y calibra el fold restante -- asi el Brier score reportado es
    out-of-fold, no inflado por reusar los mismos puntos para ajustar y evaluar
    la curva de calibracion."""
    scores = df["score"].values
    labels = (df["label"] == "synthetic").astype(int).values
    folds = df["fold"].values

    logits = _to_logit(scores).reshape(-1, 1)
    calibrated = np.full(len(df), np.nan)

    for fold in sorted(np.unique(folds)):
        train_mask, val_mask = folds != fold, folds == fold
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        platt = LogisticRegression(max_iter=1000)
        platt.fit(logits[train_mask], labels[train_mask])
        calibrated[val_mask] = platt.predict_proba(logits[val_mask])[:, 1]

    return calibrated, labels


def fit_platt_production(df):
    """El calibrador final que se usaria en produccion (POST /detect): entrenado
    sobre TODOS los scores out-of-fold disponibles, no solo un fold -- en este
    punto ya no hay fuga porque los `score` de entrada mismos ya fueron
    generados out-of-fold por cada rama (ver CONTRATOS.md seccion 4)."""
    scores = df["score"].values
    labels = (df["label"] == "synthetic").astype(int).values
    logits = _to_logit(scores).reshape(-1, 1)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(logits, labels)
    A = float(platt.coef_[0][0])
    B = float(platt.intercept_[0])
    return {"A": A, "B": B}


def reliability_curve(labels, probs, n_bins=10):
    """Para el grafico de calibracion: bins de probabilidad predicha vs.
    fraccion real de positivos en cada bin (mismo concepto que
    sklearn.calibration.calibration_curve, implementado a mano para no atarse
    a una version especifica de sklearn)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(probs, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "mean_predicted": float(np.mean(probs[mask])),
            "fraction_positive": float(np.mean(labels[mask])),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def plot_calibration(branch, labels, raw_scores, calibrated_scores, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, scores, title in [
        (axes[0], raw_scores, "Antes de calibrar (score crudo)"),
        (axes[1], calibrated_scores, "Despues de Platt (score_calibrated)"),
    ]:
        curve = reliability_curve(labels, scores)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Calibracion perfecta")
        ax.plot(curve["mean_predicted"], curve["fraction_positive"], "o-", label=branch)
        ax.set_xlabel("Probabilidad media predicha")
        ax.set_ylabel("Fraccion real de synthetic")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Curva de calibracion -- rama {branch}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def calibrate_branch(branch, path):
    if not path.exists():
        print(f"[SKIP] {branch}: no existe {path}")
        return None

    df = pd.read_csv(path)
    calibrated, labels = fit_platt_oof(df)
    valid = ~np.isnan(calibrated)

    brier_before = brier_score_loss(labels[valid], df["score"].values[valid])
    brier_after = brier_score_loss(labels[valid], calibrated[valid])

    df["score_calibrated"] = calibrated
    out_path = OUT_DIR / f"scores_{branch}_calibrated.csv"
    df.to_csv(out_path, index=False)

    plot_path = OUT_DIR / f"calibration_curve_{branch}.png"
    plot_calibration(branch, labels[valid], df["score"].values[valid], calibrated[valid], plot_path)

    production_params = fit_platt_production(df)

    print(f"[{branch}] Brier crudo={brier_before:.4f}  Brier calibrado={brier_after:.4f}  "
          f"({'mejora' if brier_after < brier_before else 'empeora'})")
    return {
        "branch": branch,
        "n": int(valid.sum()),
        "brier_raw": brier_before,
        "brier_calibrated": brier_after,
        "platt_A": production_params["A"],
        "platt_B": production_params["B"],
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    summary = []
    for branch, path in BRANCH_SCORE_FILES.items():
        result = calibrate_branch(branch, path)
        if result is not None:
            summary.append(result)

    if not summary:
        raise SystemExit(
            "No se encontro ningun archivo scores_<branch>.csv en "
            f"{DATA_DIR}. Corre primero los scripts de cada rama "
            "(train_isolated_branch.py, train_isolated_spectral.py, "
            "fusion_ablation.py)."
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))

    # parametros de produccion listos para que Joaquin (o quien sirva el
    # endpoint) los use sin tener que re-entrenar nada: score_calibrado =
    # sigmoid(A * logit(score_crudo) + B)
    params_path = OUT_DIR / "platt_params_production.json"
    with open(params_path, "w") as f:
        json.dump(
            {row["branch"]: {"A": row["platt_A"], "B": row["platt_B"]} for row in summary},
            f, indent=2,
        )
    print(f"\nParametros de Platt (produccion) guardados en {params_path}")
    print(f"Resumen guardado en {OUT_DIR / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
