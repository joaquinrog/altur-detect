"""
Tarea 3.3: bloque de fusion tardia que une el vector latente espectral (Persona
2) con el prosodico (Persona 3), y matriz de ablacion de 3 escenarios (Fase 3 /
Punto de Sincronizacion 2 del plan): Rama Espectral sola vs Rama Prosodica sola
vs Fusion de ambas, sobre el MISMO protocolo de folds para que sea comparable.

Fusion tardia = concatenar los vectores latentes de cada rama y dejar que una
sola capa lineal (aqui: regresion logistica) aprenda el peso relativo de cada
uno -- "sin fijar ponderaciones manuales no comprobadas", tal como pide el plan.
Con datasets tan chicos (353 llamadas) evitamos deliberadamente una red de fusion
mas compleja: mas parametros que datos de entrenamiento es la receta para
sobreajustar en un fin de semana de hackathon.

Este script NO se cae si data/latents_spectral.csv todavia no existe: corre solo
la ablacion prosodica y deja claro que falta para completar la matriz. El dia
que Persona 2 entregue su CSV (mismo contrato, ver CONTRATOS.md seccion 3), se
vuelve a correr este script sin tocar una linea de codigo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from persona3_prosodic.train_isolated_branch import compute_eer

DATA_DIR = Path(__file__).parent / "data"
PROSODIC_PATH = DATA_DIR / "latents_prosodic.csv"
SPECTRAL_PATH = DATA_DIR / "latents_spectral.csv"
MATRIX_PATH = DATA_DIR / "matriz_ablacion.csv"
SCORES_FUSION_PATH = DATA_DIR / "scores_fusion.csv"


def _feature_cols(df):
    return [c for c in df.columns if c not in ("anon_id", "label", "fold")]


def _run_branch_cv(df, feature_cols):
    """Mismo protocolo de CV que train_isolated_branch.py: out-of-fold, folds
    fijos de data/splits.csv, regresion logistica con features estandarizadas."""
    X = df[feature_cols].values
    y = (df["label"] == "synthetic").astype(int).values
    folds = df["fold"].values

    oof = np.full(len(y), np.nan)
    for fold in sorted(np.unique(folds)):
        train_mask, val_mask = folds != fold, folds == fold
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_val = scaler.transform(X[val_mask])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y[train_mask])
        oof[val_mask] = model.predict_proba(X_val)[:, 1]

    valid = ~np.isnan(oof)
    auc = roc_auc_score(y[valid], oof[valid])
    eer, _ = compute_eer(y[valid], oof[valid])
    return oof, auc, eer


def main():
    if not PROSODIC_PATH.exists():
        raise SystemExit(
            f"No existe {PROSODIC_PATH}. Corre primero: "
            "python -m persona3_prosodic.build_prosodic_dataset"
        )

    prosodic_df = pd.read_csv(PROSODIC_PATH)
    prosodic_feats = _feature_cols(prosodic_df)
    prosodic_df = prosodic_df.dropna(subset=prosodic_feats)

    results = []

    oof_p, auc_p, eer_p = _run_branch_cv(prosodic_df, prosodic_feats)
    print(f"[Rama Prosodica sola]  AUC={auc_p:.4f}  EER={eer_p*100:.2f}%  (n={len(prosodic_df)})")
    results.append({"escenario": "prosodica_sola", "auc": auc_p, "eer_pct": eer_p * 100,
                     "n_llamadas": len(prosodic_df), "n_features": len(prosodic_feats)})

    if not SPECTRAL_PATH.exists():
        print(
            f"\n[INFO] {SPECTRAL_PATH} todavia no existe -- falta el entregable de "
            "Persona 2 (vector latente espectral, ver CONTRATOS.md seccion 3) para "
            "completar la matriz de ablacion de 3 escenarios. Por ahora solo se "
            "reporta la rama prosodica aislada."
        )
        pd.DataFrame(results).to_csv(MATRIX_PATH, index=False)
        print(f"Resultado parcial guardado en {MATRIX_PATH}")
        return

    spectral_df = pd.read_csv(SPECTRAL_PATH)
    spectral_feats = _feature_cols(spectral_df)
    spectral_df = spectral_df.dropna(subset=spectral_feats)

    oof_s, auc_s, eer_s = _run_branch_cv(spectral_df, spectral_feats)
    print(f"[Rama Espectral sola]  AUC={auc_s:.4f}  EER={eer_s*100:.2f}%  (n={len(spectral_df)})")
    if auc_s > 0.97:
        print(
            f"  [ALERTA] AUC={auc_s:.4f} es sospechosamente alto -- NO tomar este numero "
            "al pie de la letra. Ver persona2_reference_spectral/train_isolated_spectral.py "
            "(seccion 'chequeo sobreajuste vs. senal real') y "
            "persona3_prosodic/data/feature_forensics_spectral.csv: se confirmo que al menos "
            "una feature (static3_mean) es un atajo de canal reexpresado via LFCC, y que la "
            "separacion casi perfecta persiste incluso con regularizacion L2 fuerte. Hipotesis "
            "abierta sin descartar: pocas voces/motores TTS distintos en la clase synthetic de "
            "este dataset, no necesariamente generalizacion real a voces nunca vistas."
        )
    results.append({"escenario": "espectral_sola", "auc": auc_s, "eer_pct": eer_s * 100,
                     "n_llamadas": len(spectral_df), "n_features": len(spectral_feats)})

    # fusion tardia: inner join por anon_id (solo llamadas con AMBOS vectores latentes)
    merged = prosodic_df.merge(
        spectral_df, on=["anon_id", "label", "fold"], how="inner", suffixes=("_pros", "_spec")
    )
    if len(merged) == 0:
        raise SystemExit(
            "latents_prosodic.csv y latents_spectral.csv no comparten ningun anon_id -- "
            "revisar que ambos se generaron sobre el mismo data/splits.csv"
        )
    fusion_feats = [c for c in prosodic_feats] + [c for c in spectral_feats]
    # los nombres de columna de cada rama no deberian chocar (prefijos shimmer_/
    # dshimmer_/ddshimmer_ vs los que use Persona 2), pero si chocan, el merge ya
    # les puso sufijo _pros/_spec -- se ajustan las listas de features en ese caso
    fusion_feats = [c for c in merged.columns if c not in ("anon_id", "label", "fold")]

    oof_f, auc_f, eer_f = _run_branch_cv(merged, fusion_feats)
    print(f"[Fusion tardia]        AUC={auc_f:.4f}  EER={eer_f*100:.2f}%  (n={len(merged)})")
    if auc_f > 0.97:
        print(
            "  [ALERTA] igual que la rama espectral sola: no reportar este AUC de fusion sin "
            "el caveat de arriba -- la fusion esta heredando la senal sospechosa de la rama "
            "espectral, no aportando una validacion independiente de que sea real."
        )
    results.append({"escenario": "fusion", "auc": auc_f, "eer_pct": eer_f * 100,
                     "n_llamadas": len(merged), "n_features": len(fusion_feats)})

    pd.DataFrame(results).to_csv(MATRIX_PATH, index=False)
    print(f"\nMatriz de ablacion completa guardada en {MATRIX_PATH}")

    fusion_scores = pd.DataFrame({
        "anon_id": merged["anon_id"].values,
        "label": merged["label"].values,
        "fold": merged["fold"].values,
        "branch": "fusion",
        "score": oof_f,
    })
    fusion_scores.to_csv(SCORES_FUSION_PATH, index=False)
    print(f"Scores de fusion (out-of-fold) guardados en {SCORES_FUSION_PATH} para Persona 1")


if __name__ == "__main__":
    main()
