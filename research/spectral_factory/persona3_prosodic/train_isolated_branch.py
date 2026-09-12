"""
Tarea 5: clasificador simple (logistic regression / random forest, no red
profunda -- dataset chico) sobre el vector latente prosodico, para poder correr
la parte de Persona 3 de la matriz de ablacion (Rama Prosodica sola) de forma
independiente antes de la fusion tardia del Dia 5.

Entrena con validacion cruzada respetando los folds de data/splits.csv (los
mismos que va a usar el resto del equipo -- Persona 1 los entrega, mientras
tanto splits.py genera un fallback identico en formato). Genera predicciones
OUT-OF-FOLD (nunca evalua un clip con un modelo que lo vio en train) para poder
entregarle a Persona 1 un score sin fuga de datos, listo para su calibracion de
Platt -- ver CONTRATOS.md seccion 4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = Path(__file__).parent / "data"
LATENTS_PATH = DATA_DIR / "latents_prosodic.csv"
SCORES_PATH = DATA_DIR / "scores_prosodic.csv"

N_FOLDS = 5


def compute_eer(y_true, y_score):
    """Equal Error Rate: el punto de la curva ROC donde FPR == FNR (1-TPR).
    Se reporta junto con el umbral que lo logra, que es el que en produccion
    definiria el corte is_synthetic=True/False antes de cualquier calibracion."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def run_cv(X, y, groups_fold, model_builder, model_name):
    """CV manual respetando los folds YA asignados en data/splits.csv (no se
    vuelve a correr GroupKFold aqui -- los folds son la fuente de verdad
    compartida con el resto del equipo, ver CONTRATOS.md)."""
    oof_scores = np.full(len(y), np.nan)
    for fold in sorted(np.unique(groups_fold)):
        train_mask = groups_fold != fold
        val_mask = groups_fold == fold
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_val = scaler.transform(X[val_mask])
        model = model_builder()
        model.fit(X_train, y[train_mask])
        oof_scores[val_mask] = model.predict_proba(X_val)[:, 1]

    valid = ~np.isnan(oof_scores)
    auc = roc_auc_score(y[valid], oof_scores[valid])
    eer, eer_threshold = compute_eer(y[valid], oof_scores[valid])
    print(f"[{model_name}] AUC={auc:.4f}  EER={eer*100:.2f}%  (umbral={eer_threshold:.4f})")
    return oof_scores, auc, eer


def main():
    if not LATENTS_PATH.exists():
        raise SystemExit(
            f"No existe {LATENTS_PATH}. Corre primero: "
            "python -m persona3_prosodic.build_prosodic_dataset"
        )

    df = pd.read_csv(LATENTS_PATH)
    feature_cols = [c for c in df.columns if c not in ("anon_id", "label", "fold")]
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < n_before:
        print(f"[WARN] se descartaron {n_before - len(df)} llamadas con features NaN "
              "(voz insuficiente para estimar shimmer)")

    X = df[feature_cols].values
    y = (df["label"] == "synthetic").astype(int).values
    folds = df["fold"].values

    print(f"Entrenando sobre {len(df)} llamadas, {len(feature_cols)} features, {N_FOLDS} folds")

    scores_lr, auc_lr, eer_lr = run_cv(
        X, y, folds,
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LogisticRegression",
    )
    scores_rf, auc_rf, eer_rf = run_cv(
        X, y, folds,
        lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"),
        "RandomForest",
    )

    # nos quedamos con el mejor de los dos por EER (metrica principal del equipo,
    # ver matriz de metricas del plan) para entregar como "el" score prosodico
    if eer_lr <= eer_rf:
        best_scores, best_name = scores_lr, "LogisticRegression"
    else:
        best_scores, best_name = scores_rf, "RandomForest"
    print(f"\nMejor modelo por EER: {best_name}")

    out = pd.DataFrame({
        "anon_id": df["anon_id"].values,
        "label": df["label"].values,
        "fold": df["fold"].values,
        "branch": "prosodic",
        "score": best_scores,
    })
    out.to_csv(SCORES_PATH, index=False)
    print(f"Scores out-of-fold guardados en {SCORES_PATH} (branch=prosodic)")
    print("Este archivo ya respeta el contrato de CONTRATOS.md seccion 4 para "
          "que Persona 1 lo calibre sin cambios.")


if __name__ == "__main__":
    main()
