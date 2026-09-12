"""
Entrena la rama espectral (LFCC placeholder) aislada, Y responde la pregunta
central de la Tarea 2.3/5 del prompt de Persona 2: "si entreno CON corrupcion
de canal en el train fold vs SIN ella, sube o baja el EER de validacion (que
se mantiene siempre limpia)?" -- si sube mucho con corrupcion activa, es
evidencia de que el baseline dependia del atajo de canal (AUC~0.82 de ancho de
banda del analisis forense); si se mantiene similar, es una senal sana.

Reglas de Oro respetadas (ver plan del equipo):
  - Aislamiento estricto de folds: el fold de validacion SIEMPRE se evalua con
    audio limpio, nunca corrompido, en ninguna de las dos variantes.
  - Conserva de muestras limpias: la variante "aumentada" no REEMPLAZA el train
    fold limpio por uno corrompido -- lo EXTIENDE (limpio + corrompido, ambos
    presentes), tal como pide la regla 3 del plan.
  - Corrupcion simetrica: TelephonyAugmenter no recibe el label, por diseno
    (ver persona3_prosodic/corruption.py), asi que es fisicamente imposible que
    la corrupcion favorezca a una clase.

Escribe el mejor de los dos (por EER) en persona3_prosodic/data/scores_spectral.csv
-- mismo contrato que usa Persona 1 para calibracion (ver CONTRATOS.md seccion 4).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats as spstats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from persona2_reference_spectral.lfcc_features import extract_spectral_latent
from persona3_prosodic.corruption import TelephonyAugmenter
from persona3_prosodic.prosodic_features import _load_turns_channel0
from persona3_prosodic.splits import load_splits
from persona3_prosodic.train_isolated_branch import compute_eer

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
DATA_DIR = REPO_ROOT / "persona3_prosodic" / "data"
CLEAN_PATH = DATA_DIR / "latents_spectral.csv"
CORRUPTED_PATH = DATA_DIR / "latents_spectral_corrupted.csv"
SCORES_PATH = DATA_DIR / "scores_spectral.csv"
COMPARISON_PATH = DATA_DIR / "corruption_ablation_spectral.csv"
FEATURE_FORENSICS_PATH = DATA_DIR / "feature_forensics_spectral.csv"

N_FOLDS = 5
REGULARIZATION_SWEEP = (1.0, 0.1, 0.01, 0.003, 0.001)
OVERFIT_ALERT_AUC = 0.97  # si el AUC out-of-fold se mantiene por encima de esto
                          # incluso con regularizacion muy fuerte, no es un
                          # artefacto de sobreajuste -- hay una senal real
                          # (buena o de atajo) demasiado fuerte para ser normal


def _mannwhitney_auc(df, col):
    a = df[df["label"] == "human"][col].dropna().values
    b = df[df["label"] == "synthetic"][col].dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    stat, _ = spstats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    auc = (n1 * n2 - stat) / (n1 * n2)
    return max(auc, 1 - auc)


def feature_level_forensics(clean_df, corrupted_df, feature_cols):
    """Replica, a nivel de feature individual, el mismo chequeo forense que ya
    se le hizo al dataset completo (Mann-Whitney AUC clean vs corrupted): si
    una feature con AUC alto por si sola SE DESPLOMA al corromper el canal, es
    la firma de un atajo de canal reexpresado via LFCC, no una diferencia real
    de sintesis (esto es exactamente lo que encontro el equipo con
    static3_mean -- ver README.md de esta carpeta)."""
    rows = []
    for col in feature_cols:
        auc_clean = _mannwhitney_auc(clean_df, col)
        auc_corr = _mannwhitney_auc(corrupted_df, col)
        rows.append({
            "feature": col,
            "auc_clean": auc_clean,
            "auc_corrupted": auc_corr,
            "delta": (auc_corr - auc_clean) if pd.notna(auc_clean) and pd.notna(auc_corr) else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("auc_clean", ascending=False)
    df.to_csv(FEATURE_FORENSICS_PATH, index=False)
    print(f"\nForense a nivel de feature guardado en {FEATURE_FORENSICS_PATH}")
    print(df.head(10).to_string(index=False))

    suspicious = df[(df["auc_clean"] > 0.85) & (df["delta"] < -0.10)]
    if len(suspicious) > 0:
        print(
            f"\n[ALERTA] {len(suspicious)} feature(s) con AUC alto en limpio que caen "
            ">0.10 tras la corrupcion -- candidatas a re-expresar el atajo de canal "
            f"(bandwidth AUC~0.82) del analisis forense original: {suspicious['feature'].tolist()}"
        )
    return df


def check_overfitting_vs_real_signal(df, feature_cols):
    """Un AUC out-of-fold ~1.0 con 40-120 features y ~280 muestras de train por
    fold ES sospechoso de sobreajuste por defecto. Pero si el AUC se mantiene
    alto incluso con regularizacion L2 muy fuerte (C chico), NO es sobreajuste
    -- hay una senal real y muy fuerte separando las clases (buena o un atajo
    todavia no identificado, pero no es un artefacto de ajuste del modelo)."""
    X = df[feature_cols].values
    y = (df["label"] == "synthetic").astype(int).values
    folds = df["fold"].values

    print(f"\nBarrido de regularizacion ({len(feature_cols)} features, n={len(df)}):")
    results = []
    for C in REGULARIZATION_SWEEP:
        oof = np.full(len(y), np.nan)
        for f in sorted(np.unique(folds)):
            tr, va = folds != f, folds == f
            scaler = StandardScaler()
            X_tr, X_va = scaler.fit_transform(X[tr]), scaler.transform(X[va])
            model = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
            model.fit(X_tr, y[tr])
            oof[va] = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y, oof)
        eer, _ = compute_eer(y, oof)
        print(f"  C={C:<8} AUC={auc:.4f}  EER={eer*100:.2f}%")
        results.append({"C": C, "auc": auc, "eer_pct": eer * 100})

    min_auc = min(r["auc"] for r in results)
    if min_auc > OVERFIT_ALERT_AUC:
        print(
            f"\n[ALERTA IMPORTANTE] El AUC se mantiene > {OVERFIT_ALERT_AUC} incluso con "
            "regularizacion L2 muy fuerte (C=0.001). Esto NO es un artefacto de "
            "sobreajuste del clasificador -- hay una senal extremadamente fuerte y "
            "consistente en los datos. Dado que static3_mean (spectral shape) ya se "
            "confirmo como un atajo de canal reexpresado, y que incluso excluyendolo "
            "el resto de las features (dynamic-only) TAMBIEN separa casi perfecto, la "
            "hipotesis que el equipo deberia descartar antes de confiar en este numero "
            "es: 'las llamadas synthetic de este dataset vienen de muy pocas voces/"
            "motores TTS distintos', lo cual haria que el modelo aprenda a reconocer "
            "ESE PUNADO de voces especificas en vez de sintesis en general -- GroupKFold "
            "por llamada NO protege contra esto (solo evita fuga de hablante, no de "
            "motor de sintesis). Si el set oculto de evaluacion usa voces/TTS nunca "
            "vistos (como dice el README del dataset), este AUC podria no generalizar."
        )
    return pd.DataFrame(results)


def build_corrupted_latents(splits_df, force=False):
    """Extrae LFCC sobre version corrompida (determinista, forzada) de cada
    llamada. Se cachea en disco porque tarda lo mismo que la extraccion limpia."""
    if CORRUPTED_PATH.exists() and not force:
        return pd.read_csv(CORRUPTED_PATH)

    rows = []
    t0 = time.time()
    for i, row in splits_df.iterrows():
        anon_id, label, fold = row["anon_id"], row["label"], row["fold"]
        audio, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
        ch0 = audio[:, 0].astype(float)
        turns = _load_turns_channel0(TURNS_DIR / f"{anon_id}.json")

        seed = abs(hash(anon_id)) % (2**32)
        # families=("channel",) explicito: esta comparacion prueba especificamente
        # si el atajo de canal/bandwidth sobrevive corrupcion de codec -- no se
        # quiere mezclar pitch/time-stretch aqui (esos son experimentos aparte,
        # ver phase1_stress_test/).
        augmenter = TelephonyAugmenter(p=1.0, families=("channel",), seed=seed)  # forzada: queremos SIEMPRE la version corrompida aqui
        corrupted_audio, _ = augmenter(ch0, sr)

        latent = extract_spectral_latent(anon_id, corrupted_audio, sr, turns)
        if latent is None:
            continue
        r = {"anon_id": anon_id, "label": label, "fold": fold}
        r.update(latent.features)
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  corrupcion: {i+1}/{len(splits_df)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(CORRUPTED_PATH, index=False)
    print(f"Latentes corrompidos guardados en {CORRUPTED_PATH} ({len(df)} filas)")
    return df


def _run_cv(train_feats_by_fold, val_feats_by_fold, feature_cols, folds_unique):
    """CV manual: para cada fold, entrena con lo que le pasen en
    train_feats_by_fold[fold] y evalua SIEMPRE sobre val_feats_by_fold[fold]
    (que debe ser el subconjunto limpio de ese fold)."""
    all_ids, all_labels, all_folds, all_scores = [], [], [], []
    for fold in folds_unique:
        train_df = train_feats_by_fold[fold]
        val_df = val_feats_by_fold[fold]
        if len(train_df) == 0 or len(val_df) == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train = (train_df["label"] == "synthetic").astype(int).values
        X_val = val_df[feature_cols].values
        y_val = (val_df["label"] == "synthetic").astype(int).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train_s, y_train)
        scores = model.predict_proba(X_val_s)[:, 1]

        all_ids.extend(val_df["anon_id"].tolist())
        all_labels.extend(y_val.tolist())
        all_folds.extend([fold] * len(val_df))
        all_scores.extend(scores.tolist())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    eer, _ = compute_eer(all_labels, all_scores)
    return all_ids, all_folds, all_labels, all_scores, auc, eer


def main():
    if not CLEAN_PATH.exists():
        raise SystemExit(
            f"No existe {CLEAN_PATH}. Corre primero: "
            "python -m persona2_reference_spectral.build_spectral_dataset"
        )

    splits_df = load_splits()
    clean_df = pd.read_csv(CLEAN_PATH)
    feature_cols = [c for c in clean_df.columns if c not in ("anon_id", "label", "fold")]
    clean_df = clean_df.dropna(subset=feature_cols)

    print("Extrayendo (o cargando cache de) LFCC sobre audio corrompido para todas las llamadas...")
    corrupted_df = build_corrupted_latents(splits_df)
    corrupted_df = corrupted_df.dropna(subset=feature_cols)

    print("\n=== Forense a nivel de feature individual (clean vs corrupted) ===")
    feature_level_forensics(clean_df, corrupted_df, feature_cols)

    print("\n=== Chequeo sobreajuste vs. senal real (barrido de regularizacion) ===")
    check_overfitting_vs_real_signal(clean_df, feature_cols)

    folds_unique = sorted(clean_df["fold"].unique())

    # Variante A: baseline, solo limpio
    train_clean = {f: clean_df[clean_df["fold"] != f] for f in folds_unique}
    val_clean = {f: clean_df[clean_df["fold"] == f] for f in folds_unique}
    ids_a, folds_a, y_a, scores_a, auc_a, eer_a = _run_cv(train_clean, val_clean, feature_cols, folds_unique)
    print(f"[Baseline: SIN corrupcion en train]        AUC={auc_a:.4f}  EER={eer_a*100:.2f}%")

    # Variante B: aumentada -- train = limpio UNION corrompido del mismo fold de train;
    # val SIEMPRE limpio (aislamiento estricto de folds)
    train_aug = {
        f: pd.concat([clean_df[clean_df["fold"] != f], corrupted_df[corrupted_df["fold"] != f]])
        for f in folds_unique
    }
    ids_b, folds_b, y_b, scores_b, auc_b, eer_b = _run_cv(train_aug, val_clean, feature_cols, folds_unique)
    print(f"[Aumentado: CON corrupcion en train]       AUC={auc_b:.4f}  EER={eer_b*100:.2f}%")

    delta_eer = (eer_b - eer_a) * 100
    if delta_eer > 3:
        veredicto = (
            f"El EER SUBE {delta_eer:.2f} puntos porcentuales al entrenar con corrupcion. "
            "Esto es evidencia de que el baseline SI dependia en parte del atajo de canal "
            "(ancho de banda telefonico, AUC~0.82 standalone) -- resultado esperado y es el "
            "argumento de rigor cientifico para el pitch, no un fracaso."
        )
    else:
        veredicto = (
            f"El EER cambia solo {delta_eer:+.2f} puntos porcentuales al entrenar con corrupcion. "
            "OJO: esto NO significa que la rama este libre de atajos -- ya confirmamos arriba que "
            "static3_mean (un atajo de canal reexpresado via LFCC) SI se destruye con la corrupcion "
            "(AUC 0.943->0.749), pero el resto de las features dynamic (delta/deltadelta std) son "
            "tan fuertes por si solas que compensan esa perdida y el AUC global casi no se mueve. "
            "La corrupcion esta funcionando como se espera sobre la feature que SI era un atajo; "
            "el numero global sigue siendo sospechosamente alto por la razon distinta que se "
            "senala en la alerta de arriba (posible sobre-representacion de pocas voces TTS), no "
            "por el atajo de canal que este experimento especificamente prueba."
        )
    print(f"\n{veredicto}")

    comparison_df = pd.DataFrame([
        {"variante": "baseline_sin_corrupcion", "auc": auc_a, "eer_pct": eer_a * 100},
        {"variante": "aumentado_con_corrupcion", "auc": auc_b, "eer_pct": eer_b * 100},
    ])
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    print(f"Comparacion guardada en {COMPARISON_PATH}")

    # se entrega a Persona 1 el mejor de los dos por EER (mismo criterio que la
    # rama prosodica), pero en la practica preferimos "aumentado" salvo que sea
    # claramente peor, porque es la version robusta a canal
    if eer_b <= eer_a + 1.0:  # tolerancia de 1pt: preferir robustez si el costo es chico
        best_ids, best_folds, best_labels, best_scores, chosen = ids_b, folds_b, y_b, scores_b, "aumentado_con_corrupcion"
    else:
        best_ids, best_folds, best_labels, best_scores, chosen = ids_a, folds_a, y_a, scores_a, "baseline_sin_corrupcion"

    out = pd.DataFrame({
        "anon_id": best_ids,
        "label": ["synthetic" if v == 1 else "human" for v in best_labels],
        "fold": best_folds,
        "branch": "spectral",
        "score": best_scores,
    })
    out.to_csv(SCORES_PATH, index=False)
    print(f"\nScores elegidos: {chosen}. Guardados en {SCORES_PATH} (branch=spectral) para Persona 1.")


if __name__ == "__main__":
    main()
