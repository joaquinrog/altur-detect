"""
Analisis forense del dataset Altur (HackMTY 2026).

Objetivo: antes de construir cualquier clasificador humano-vs-sintetico, verificar
que el dataset no se pueda resolver con un "atajo" del canal de grabacion/transmision
(ruido de fondo, ancho de banda, artefactos de codec) en lugar de una diferencia real
de sintesis de voz. Si una sola caracteristica de bajo nivel del canal separa casi
perfectamente humano vs sintetico (AUC ~1.0), es una bandera roja de dataset amanado.

Este script es EXCLUSIVAMENTE exploratorio. No entrena ningun clasificador humano-vs-
sintetico; el unico "clasificador" que aparece es un modelo de una sola variable
(threshold sobre una sola feature) usado unicamente para medir el AUC de esa feature,
como pide la tarea.

Uso:
    python forensic_analysis.py
Requiere que audio/ y turns/ ya existan en la raiz del repo (ver README.md para bajar
el zip de audio desde Releases) y que manifest.csv este presente.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
from scipy import signal as sps
from scipy import stats as spstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# np.trapz fue removido en numpy>=2.0 en favor de np.trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _band_energy(f, pxx, lo, hi):
    """Energia dentro de [lo, hi] Hz via suma rectangular (pxx * df).
    A diferencia de un trapz sobre el subconjunto enmascarado, esto no requiere que la
    banda contenga >=2 bins de frecuencia: con un solo bin (bandas angostas como el hum
    de 50/60 Hz a esta resolucion espectral) trapz devolveria 0 por definicion."""
    if len(f) < 2:
        return 0.0
    df = f[1] - f[0]
    mask = (f >= lo) & (f <= hi)
    return float(np.sum(pxx[mask]) * df)

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
MANIFEST_PATH = REPO_ROOT / "manifest.csv"
AUDIO_DIR = REPO_ROOT / "audio"
TURNS_DIR = REPO_ROOT / "turns"
OUT_DIR = REPO_ROOT / "analisis_forense"
OUT_DIR.mkdir(exist_ok=True)

SR = 8000  # sample rate esperado (8 kHz segun README)
CALLER_CHANNEL = 0  # canal 0 = quien llama (a clasificar)

N_PER_CLASS = 40          # tamano de muestra estratificada por clase (split train)
MIN_SILENCE_S = 0.3       # minimo de silencio utilizable por clip para el analisis de silencio
SILENCE_MARGIN_S = 0.05   # margen de seguridad para no comerse el borde de un turno
BANDPASS_LO, BANDPASS_HI = 300.0, 3400.0  # banda telefonica clasica
LINE_HUM_FREQS = [50.0, 60.0]  # hum de linea electrica y armonicos
LINE_HUM_HARMONICS = 3
SPEC_EXAMPLES_PER_CLASS = 5  # para la grilla de espectrogramas

RNG_SEED = 42

CLASSES = ["human", "synthetic"]
COLORS = {"human": "#1f77b4", "synthetic": "#d62728"}

LOG_LINES = []  # errores/omisiones para el log de reproducibilidad


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


# ----------------------------------------------------------------------------
# 1. Carga de datos
# ----------------------------------------------------------------------------

def load_manifest_sample():
    """Lee manifest.csv y toma una muestra estratificada por clase, usando split=train."""
    df = pd.read_csv(MANIFEST_PATH)
    train = df[df["split"] == "train"].copy()

    rng = np.random.default_rng(RNG_SEED)
    sampled = []
    for label in CLASSES:
        subset = train[train["label"] == label]
        n = min(N_PER_CLASS, len(subset))
        if n < len(subset):
            idx = rng.choice(subset.index.values, size=n, replace=False)
            sampled.append(subset.loc[idx])
        else:
            sampled.append(subset)
    sample_df = pd.concat(sampled).reset_index(drop=True)
    log(f"Muestra estratificada: {sample_df.groupby('label').size().to_dict()} "
        f"(de {train.groupby('label').size().to_dict()} disponibles en train)")
    return sample_df


def load_clip_channel0(anon_id):
    """Carga el canal 0 (llamante) de un clip estereo. Devuelve (audio, sr) o (None, None)."""
    wav_path = AUDIO_DIR / f"{anon_id}.wav"
    if not wav_path.exists():
        log(f"[WARN] audio faltante: {wav_path.name}")
        return None, None
    try:
        audio, sr = sf.read(wav_path, always_2d=True)
        ch0 = audio[:, CALLER_CHANNEL].astype(np.float64)
        return ch0, sr
    except Exception as e:
        log(f"[WARN] fallo al cargar {wav_path.name}: {e}")
        return None, None


def load_turns(anon_id):
    """Carga turns/<anon_id>.json. Devuelve lista de turns del canal 0, o None si falla."""
    turns_path = TURNS_DIR / f"{anon_id}.json"
    if not turns_path.exists():
        log(f"[WARN] turns faltante: {turns_path.name}")
        return None
    try:
        with open(turns_path) as f:
            data = json.load(f)
        ch0_turns = [t for t in data["turns"] if t["channel"] == CALLER_CHANNEL]
        ch0_turns.sort(key=lambda t: t["start"])
        return ch0_turns
    except Exception as e:
        log(f"[WARN] fallo al parsear {turns_path.name}: {e}")
        return None


# ----------------------------------------------------------------------------
# 2. Aislar silencio / no-habla del canal 0
# ----------------------------------------------------------------------------

def silence_segments(turns, duration_s):
    """Dado los turns del canal 0 (ordenados), devuelve lista de (start, end) de huecos
    SIN habla: antes del primer turno y entre turnos consecutivos."""
    gaps = []
    prev_end = 0.0
    for t in turns:
        if t["start"] > prev_end:
            gaps.append((prev_end, t["start"]))
        prev_end = max(prev_end, t["end"])
    if duration_s > prev_end:
        gaps.append((prev_end, duration_s))

    # aplica margen de seguridad para no rozar el borde de un turno de habla
    safe_gaps = []
    for s, e in gaps:
        s2, e2 = s + SILENCE_MARGIN_S, e - SILENCE_MARGIN_S
        if e2 - s2 > 0:
            safe_gaps.append((s2, e2))
    return safe_gaps


def extract_segments(audio, sr, segments):
    """Concatena las muestras de audio dentro de una lista de segmentos (start,end) en segundos."""
    chunks = []
    n = len(audio)
    for s, e in segments:
        i0, i1 = int(s * sr), int(e * sr)
        i0, i1 = max(0, i0), min(n, i1)
        if i1 > i0:
            chunks.append(audio[i0:i1])
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


# ----------------------------------------------------------------------------
# Recoleccion principal: recorre la muestra y arma estructuras por clase
# ----------------------------------------------------------------------------

def collect_data(sample_df):
    """Recorre la muestra, carga audio+turns, separa silencio/habla del canal 0 y
    devuelve un dict con listas de senales de silencio, senales de habla y metadatos,
    agrupadas por clase."""
    silence_by_class = {c: [] for c in CLASSES}       # lista de arrays de audio (solo silencio)
    speech_by_class = {c: [] for c in CLASSES}        # lista de arrays de audio (solo habla)
    ids_by_class = {c: [] for c in CLASSES}           # anon_id paralelo a silence_by_class
    speech_ids_by_class = {c: [] for c in CLASSES}    # anon_id paralelo a speech_by_class
    examples_by_class = {c: [] for c in CLASSES}      # (anon_id, audio_completo, sr) para espectrogramas

    n_ok, n_skip_silence, n_fail = 0, 0, 0

    for _, row in sample_df.iterrows():
        anon_id, label = row["anon_id"], row["label"]
        audio, sr = load_clip_channel0(anon_id)
        turns = load_turns(anon_id)
        if audio is None or turns is None:
            n_fail += 1
            continue
        if sr != SR:
            log(f"[WARN] {anon_id}: sr={sr} distinto de {SR} esperado, se usa el real")

        duration_s = len(audio) / sr

        # habla: dentro de los turns del canal 0
        speech_segs = [(t["start"], t["end"]) for t in turns]
        speech_audio = extract_segments(audio, sr, speech_segs)

        # silencio: huecos entre turns
        sil_segs = silence_segments(turns, duration_s)
        sil_audio = extract_segments(audio, sr, sil_segs)

        if len(speech_audio) > 0:
            speech_by_class[label].append(speech_audio)
            speech_ids_by_class[label].append(anon_id)

        if len(sil_audio) / sr < MIN_SILENCE_S:
            n_skip_silence += 1
        else:
            silence_by_class[label].append(sil_audio)
            ids_by_class[label].append(anon_id)

        if len(examples_by_class[label]) < SPEC_EXAMPLES_PER_CLASS:
            examples_by_class[label].append((anon_id, audio, sr))

        n_ok += 1

    log(f"Clips procesados OK: {n_ok}, fallos de carga: {n_fail}, "
        f"descartados por silencio insuficiente (<{MIN_SILENCE_S}s): {n_skip_silence}")
    return silence_by_class, ids_by_class, speech_by_class, speech_ids_by_class, examples_by_class


# ----------------------------------------------------------------------------
# 3a. Piso de ruido espectral (Welch PSD sobre silencio)
# ----------------------------------------------------------------------------

def compute_psd(audio, sr, nperseg=512):
    if len(audio) < nperseg:
        nperseg = max(64, len(audio))
    f, pxx = sps.welch(audio, fs=sr, nperseg=nperseg)
    return f, pxx


def plot_noise_floor(silence_by_class):
    fig, ax = plt.subplots(figsize=(9, 5))
    common_f = None
    for label in CLASSES:
        clips = silence_by_class[label]
        if not clips:
            continue
        psds = []
        for audio in clips:
            f, pxx = compute_psd(audio, SR)
            if common_f is None or len(f) != len(common_f):
                common_f = f
            psds.append(pxx)
        # alinear longitudes (nperseg fijo => misma longitud salvo clips muy cortos)
        min_len = min(len(p) for p in psds)
        psds = np.array([p[:min_len] for p in psds])
        f = common_f[:min_len]
        pxx_db = 10 * np.log10(psds + 1e-14)
        mean_db = pxx_db.mean(axis=0)
        std_db = pxx_db.std(axis=0)
        ax.plot(f, mean_db, label=f"{label} (n={len(clips)})", color=COLORS[label])
        ax.fill_between(f, mean_db - std_db, mean_db + std_db, color=COLORS[label], alpha=0.2)

    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad espectral de potencia (dB/Hz)")
    ax.set_title("Piso de ruido espectral (Welch PSD) sobre tramos de silencio, canal 0")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_piso_de_ruido_psd.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3b. LTAS (espectro promedio de largo plazo) sobre tramos con habla
# ----------------------------------------------------------------------------

def plot_ltas(speech_by_class):
    fig, ax = plt.subplots(figsize=(9, 5))
    common_f = None
    for label in CLASSES:
        clips = speech_by_class[label]
        if not clips:
            continue
        psds = []
        for audio in clips:
            f, pxx = compute_psd(audio, SR, nperseg=1024)
            if common_f is None or len(f) != len(common_f):
                common_f = f
            psds.append(pxx)
        min_len = min(len(p) for p in psds)
        psds = np.array([p[:min_len] for p in psds])
        f = common_f[:min_len]
        pxx_db = 10 * np.log10(psds + 1e-14)
        mean_db = pxx_db.mean(axis=0)
        std_db = pxx_db.std(axis=0)
        ax.plot(f, mean_db, label=f"{label} (n={len(clips)})", color=COLORS[label])
        ax.fill_between(f, mean_db - std_db, mean_db + std_db, color=COLORS[label], alpha=0.2)

    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("LTAS (dB/Hz)")
    ax.set_title("Espectro promedio de largo plazo (LTAS) sobre tramos con habla, canal 0")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_ltas.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3c. Ancho de banda ocupado (energia dentro/fuera de 300-3400 Hz)
# ----------------------------------------------------------------------------

def in_band_energy_fraction(audio, sr, lo=BANDPASS_LO, hi=BANDPASS_HI):
    """Fraccion de energia (welch PSD) que cae dentro de [lo, hi] Hz."""
    if len(audio) < 32:
        return np.nan
    f, pxx = compute_psd(audio, sr, nperseg=1024)
    total = _band_energy(f, pxx, f[0], f[-1])
    if total <= 0:
        return np.nan
    in_band = _band_energy(f, pxx, lo, hi)
    return in_band / total


def compute_bandwidth_feature(speech_by_class, speech_ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(speech_ids_by_class[label], speech_by_class[label]):
            frac = in_band_energy_fraction(audio, SR)
            rows.append({"anon_id": anon_id, "label": label, "in_band_energy_frac": frac})
    return pd.DataFrame(rows).dropna()


def plot_bandwidth(df_bw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in CLASSES:
        vals = df_bw[df_bw["label"] == label]["in_band_energy_frac"]
        axes[0].hist(vals, bins=20, alpha=0.55, label=f"{label} (n={len(vals)})", color=COLORS[label])
    axes[0].set_xlabel(f"Fraccion de energia dentro de {BANDPASS_LO:.0f}-{BANDPASS_HI:.0f} Hz")
    axes[0].set_ylabel("Numero de clips")
    axes[0].set_title("Histograma: energia en banda telefonica")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    data = [df_bw[df_bw["label"] == l]["in_band_energy_frac"].values for l in CLASSES]
    bp = axes[1].boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.5)
    axes[1].set_ylabel(f"Fraccion de energia en {BANDPASS_LO:.0f}-{BANDPASS_HI:.0f} Hz")
    axes[1].set_title("Boxplot por clase")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Ancho de banda ocupado: energia dentro vs fuera de la banda telefonica clasica")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_ancho_de_banda.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3d. Espectrogramas lado a lado
# ----------------------------------------------------------------------------

def plot_spectrogram_grid(examples_by_class):
    n_cols = SPEC_EXAMPLES_PER_CLASS
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5))
    for row_idx, label in enumerate(CLASSES):
        examples = examples_by_class[label][:n_cols]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(examples):
                anon_id, audio, sr = examples[col_idx]
                S = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr,
                                                     n_fft=512, hop_length=160, n_mels=64,
                                                     fmax=sr / 2)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, sr=sr, hop_length=160, x_axis="time",
                                                 y_axis="mel", ax=ax, cmap="magma")
                ax.set_title(anon_id, fontsize=8)
            else:
                ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")
    fig.suptitle("Espectrogramas mel de ejemplos (canal 0, llamante) — arriba: human, abajo: synthetic")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_espectrogramas_grid.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3e. RMS / loudness sobre tramos con habla
# ----------------------------------------------------------------------------

def compute_rms_feature(speech_by_class, speech_ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(speech_ids_by_class[label], speech_by_class[label]):
            if len(audio) == 0:
                continue
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            rows.append({"anon_id": anon_id, "label": label, "rms": rms})
    return pd.DataFrame(rows)


def plot_rms(df_rms):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in CLASSES:
        vals = df_rms[df_rms["label"] == label]["rms"]
        axes[0].hist(vals, bins=20, alpha=0.55, label=f"{label} (n={len(vals)})", color=COLORS[label])
    axes[0].set_xlabel("RMS (amplitud lineal)")
    axes[0].set_ylabel("Numero de clips")
    axes[0].set_title("Histograma: RMS de tramos con habla")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    data = [df_rms[df_rms["label"] == l]["rms"].values for l in CLASSES]
    bp = axes[1].boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.5)
    axes[1].set_ylabel("RMS")
    axes[1].set_title("Boxplot por clase")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Distribucion de RMS / loudness (tramos con habla, canal 0)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_rms_loudness.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3f. Zero-crossing rate
# ----------------------------------------------------------------------------

def compute_zcr_feature(speech_by_class, speech_ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(speech_ids_by_class[label], speech_by_class[label]):
            if len(audio) < 2:
                continue
            zcr = librosa.feature.zero_crossing_rate(audio.astype(np.float32),
                                                        frame_length=1024, hop_length=512)
            rows.append({"anon_id": anon_id, "label": label, "zcr": float(np.mean(zcr))})
    return pd.DataFrame(rows)


def plot_zcr(df_zcr):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in CLASSES:
        vals = df_zcr[df_zcr["label"] == label]["zcr"]
        axes[0].hist(vals, bins=20, alpha=0.55, label=f"{label} (n={len(vals)})", color=COLORS[label])
    axes[0].set_xlabel("Zero-crossing rate")
    axes[0].set_ylabel("Numero de clips")
    axes[0].set_title("Histograma: ZCR")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    data = [df_zcr[df_zcr["label"] == l]["zcr"].values for l in CLASSES]
    bp = axes[1].boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.5)
    axes[1].set_ylabel("ZCR")
    axes[1].set_title("Boxplot por clase")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Zero-crossing rate (tramos con habla, canal 0)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_zcr.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3g. Artefactos de cuantizacion/codec: spectral flatness + patrones tipo peine
# ----------------------------------------------------------------------------

def compute_flatness_feature(silence_by_class, ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(ids_by_class[label], silence_by_class[label]):
            if len(audio) < 1024:
                continue
            flat = librosa.feature.spectral_flatness(y=audio.astype(np.float32),
                                                        n_fft=512, hop_length=256)
            rows.append({"anon_id": anon_id, "label": label, "spectral_flatness": float(np.mean(flat))})
    return pd.DataFrame(rows)


def comb_pattern_score(audio, sr, nperseg=1024):
    """Heuristica simple para detectar patrones tipo 'peine' (picos periodicos anomalos)
    en la PSD del silencio, tipicos de codecs de banda angosta. Se mide como la energia
    de la autocorrelacion del espectro (en dB) despues de restar una tendencia suave:
    picos periodicos producen autocorrelacion elevada a lags != 0."""
    if len(audio) < nperseg:
        return np.nan
    f, pxx = sps.welch(audio, fs=sr, nperseg=nperseg)
    pxx_db = 10 * np.log10(pxx + 1e-14)
    # quitar tendencia de baja frecuencia (suavizado) para quedarnos con la "textura" periodica
    smooth = sps.medfilt(pxx_db, kernel_size=9 if len(pxx_db) >= 9 else 1)
    residual = pxx_db - smooth
    residual = residual - residual.mean()
    # autocorrelacion normalizada del residual
    ac = np.correlate(residual, residual, mode="full")
    ac = ac[len(ac) // 2:]
    if ac[0] <= 0:
        return np.nan
    ac_norm = ac / ac[0]
    # score = energia de la autocorrelacion en lags 1..N (excluye lag 0), a mayor valor
    # mas estructura periodica (candidato a patron de peine de codec)
    return float(np.mean(ac_norm[1:min(len(ac_norm), 40)] ** 2))


def compute_comb_feature(silence_by_class, ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(ids_by_class[label], silence_by_class[label]):
            score = comb_pattern_score(audio, SR)
            if not np.isnan(score):
                rows.append({"anon_id": anon_id, "label": label, "comb_score": score})
    return pd.DataFrame(rows)


def plot_flatness(df_flat):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in CLASSES:
        vals = df_flat[df_flat["label"] == label]["spectral_flatness"]
        axes[0].hist(vals, bins=20, alpha=0.55, label=f"{label} (n={len(vals)})", color=COLORS[label])
    axes[0].set_xlabel("Spectral flatness (silencio)")
    axes[0].set_ylabel("Numero de clips")
    axes[0].set_title("Histograma: spectral flatness")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    data = [df_flat[df_flat["label"] == l]["spectral_flatness"].values for l in CLASSES]
    bp = axes[1].boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.5)
    axes[1].set_ylabel("Spectral flatness")
    axes[1].set_title("Boxplot por clase")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Spectral flatness sobre tramos de silencio (artefactos de cuantizacion/codec)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_spectral_flatness.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3h. Hum de linea electrica (50/60 Hz y armonicos)
# ----------------------------------------------------------------------------

def line_hum_energy_ratio(audio, sr, base_freqs=LINE_HUM_FREQS, n_harm=LINE_HUM_HARMONICS,
                            bw=2.0, nperseg=2048):
    """Fraccion de energia total (PSD del silencio) concentrada en bandas estrechas
    alrededor de 50/60 Hz y sus armonicos."""
    if len(audio) < nperseg:
        nperseg = max(64, len(audio))
    f, pxx = sps.welch(audio, fs=sr, nperseg=nperseg)
    total = _band_energy(f, pxx, f[0], f[-1])
    if total <= 0:
        return np.nan
    hum_energy = 0.0
    for base in base_freqs:
        for h in range(1, n_harm + 1):
            target = base * h
            if target >= f[-1]:
                continue
            hum_energy += _band_energy(f, pxx, target - bw, target + bw)
    return hum_energy / total


def compute_hum_feature(silence_by_class, ids_by_class):
    rows = []
    for label in CLASSES:
        for anon_id, audio in zip(ids_by_class[label], silence_by_class[label]):
            ratio = line_hum_energy_ratio(audio, SR)
            if ratio is not None and not np.isnan(ratio):
                rows.append({"anon_id": anon_id, "label": label, "line_hum_frac": ratio})
    return pd.DataFrame(rows)


def plot_hum(df_hum):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [df_hum[df_hum["label"] == l]["line_hum_frac"].values for l in CLASSES]
    bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.5)
    ax.set_ylabel("Fraccion de energia en bandas de 50/60 Hz + armonicos")
    ax.set_title("Hum de linea electrica en el piso de ruido, por clase")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_hum_linea_electrica.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Pruebas estadisticas: Mann-Whitney U + AUC de una sola feature
# ----------------------------------------------------------------------------

def mannwhitney_auc(df, feature_col, label_col="label", pos_label="synthetic"):
    """Corre Mann-Whitney U entre human y synthetic para una feature, y calcula el AUC
    equivalente (U / (n1*n2)) que mide que tan bien esa UNA feature separa las clases,
    sin necesidad de sklearn."""
    a = df[df[label_col] == "human"][feature_col].dropna().values
    b = df[df[label_col] == "synthetic"][feature_col].dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(a), len(b)
    stat, pval = spstats.mannwhitneyu(a, b, alternative="two-sided")
    # AUC = P(synthetic > human) usando el estadistico U respecto a "b" (synthetic)
    # mannwhitneyu con orden (a, b) da U para "a"; U_b = n1*n2 - U_a
    n1, n2 = len(a), len(b)
    u_a = stat
    u_b = n1 * n2 - u_a
    auc = u_b / (n1 * n2)  # P(synthetic > human); si <0.5, invertimos para reportar separacion
    auc_sep = max(auc, 1 - auc)  # tamano de efecto de separacion, independiente de direccion
    return pval, auc_sep, n1, n2


def build_stats_table(feature_frames):
    """feature_frames: dict feature_name -> (df, col_name)"""
    rows = []
    for feat_name, (df, col) in feature_frames.items():
        pval, auc, n_h, n_s = mannwhitney_auc(df, col)
        verdict = "sospechoso" if (not np.isnan(auc) and auc > 0.85) else "normal"
        rows.append({
            "feature": feat_name,
            "n_human": n_h,
            "n_synthetic": n_s,
            "p_value": pval,
            "auc_separacion": auc,
            "veredicto": verdict,
        })
    return pd.DataFrame(rows).sort_values("auc_separacion", ascending=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No se encontro {MANIFEST_PATH}. Clona el repo alturio/hackmty26 primero.")
    if not AUDIO_DIR.exists():
        raise SystemExit(
            f"No se encontro {AUDIO_DIR}. Descarga altur-challenge-audio.zip desde Releases "
            f"del repo y descomprimelo en la raiz (README.md tiene el link)."
        )

    log("=== Analisis forense del dataset Altur (HackMTY 2026) ===")
    sample_df = load_manifest_sample()

    (silence_by_class, ids_by_class,
     speech_by_class, speech_ids_by_class,
     examples_by_class) = collect_data(sample_df)

    log("\n--- Generando graficas ---")
    plot_noise_floor(silence_by_class)
    log("  [ok] 01_piso_de_ruido_psd.png")

    plot_ltas(speech_by_class)
    log("  [ok] 02_ltas.png")

    df_bw = compute_bandwidth_feature(speech_by_class, speech_ids_by_class)
    plot_bandwidth(df_bw)
    log("  [ok] 03_ancho_de_banda.png")

    plot_spectrogram_grid(examples_by_class)
    log("  [ok] 04_espectrogramas_grid.png")

    df_rms = compute_rms_feature(speech_by_class, speech_ids_by_class)
    plot_rms(df_rms)
    log("  [ok] 05_rms_loudness.png")

    df_zcr = compute_zcr_feature(speech_by_class, speech_ids_by_class)
    plot_zcr(df_zcr)
    log("  [ok] 06_zcr.png")

    df_flat = compute_flatness_feature(silence_by_class, ids_by_class)
    plot_flatness(df_flat)
    log("  [ok] 07_spectral_flatness.png")

    df_comb = compute_comb_feature(silence_by_class, ids_by_class)
    log(f"  [info] comb_score calculado para {len(df_comb)} clips (ver tabla resumen)")

    df_hum = compute_hum_feature(silence_by_class, ids_by_class)
    plot_hum(df_hum)
    log("  [ok] 08_hum_linea_electrica.png")

    log("\n--- Pruebas estadisticas (Mann-Whitney U + AUC de una sola feature) ---")
    feature_frames = {
        "ancho_de_banda_telefonico (in_band_energy_frac)": (df_bw, "in_band_energy_frac"),
        "rms_loudness": (df_rms, "rms"),
        "zero_crossing_rate": (df_zcr, "zcr"),
        "spectral_flatness_silencio": (df_flat, "spectral_flatness"),
        "comb_pattern_score_silencio": (df_comb, "comb_score"),
        "line_hum_fraction": (df_hum, "line_hum_frac"),
    }
    stats_df = build_stats_table(feature_frames)
    stats_df.to_csv(OUT_DIR / "resumen_estadistico.csv", index=False)

    with open(OUT_DIR / "resumen_estadistico.md", "w") as f:
        f.write("# Resumen estadistico — analisis forense dataset Altur\n\n")
        f.write(stats_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    log("\n" + stats_df.to_string(index=False))

    n_suspicious = (stats_df["veredicto"] == "sospechoso").sum()
    if n_suspicious > 0:
        interp = (
            "**Alerta:** se encontraron " + str(n_suspicious) + " caracteristica(s) de bajo nivel "
            "del canal de grabacion — no relacionadas con el contenido linguistico ni con la calidad "
            "de sintesis en si — que separan casi perfectamente las clases human vs synthetic "
            "(AUC de una sola variable > 0.85). Esto es una senal de que el dataset podria contener "
            "un atajo de canal: por ejemplo, si el audio humano se grabo por una via distinta al "
            "audio sintetico (linea telefonica real con su ruido caracteristico, vs. inyeccion "
            "directa del pipeline TTS sin pasar por el mismo canal), cualquier clasificador podria "
            "estar aprendiendo a distinguir el canal de grabacion en vez de la sintesis de voz. "
            "Un modelo asi funcionaria muy bien en train/val pero se caeria en el set de evaluacion "
            "oculto, que usa voces y grabaciones nunca vistas. Se recomienda revisar a fondo la(s) "
            "caracteristica(s) marcadas como 'sospechoso' en la tabla antes de construir cualquier "
            "clasificador, y considerar normalizar o descartar esa senal (por ejemplo, con "
            "data augmentation que iguale el ruido de fondo entre clases, o filtrando el dataset)."
        )
    else:
        borderline = stats_df[(stats_df["auc_separacion"] > 0.75) & (stats_df["auc_separacion"] <= 0.85)]
        if len(borderline) > 0:
            nombres_valores = "; ".join(
                f"{r.feature} (AUC={r.auc_separacion:.2f})" for r in borderline.itertuples()
            )
            aviso_borderline = (
                " Sin embargo, no todo es perfecto: " + nombres_valores + " quedaron en zona "
                "'borderline' (AUC entre 0.75 y 0.85) — no cruzan el umbral de alarma, pero son las "
                "candidatas mas probables a un sesgo de canal si el problema existe de forma mas sutil "
                "de lo que este umbral captura. Vale la pena inspeccionar sus graficas correspondientes "
                "con mas detalle antes de dar luz verde."
            )
        else:
            aviso_borderline = ""
        interp = (
            "**Buenas noticias:** ninguna caracteristica de bajo nivel del canal de grabacion "
            "(piso de ruido, ancho de banda, RMS, zero-crossing rate, spectral flatness, patrones "
            "tipo peine de codec, hum de linea electrica) separa las clases human vs synthetic de "
            "forma casi perfecta por si sola (todos los AUC de una sola variable quedaron por debajo "
            "de 0.85). Esto sugiere que el dataset no depende de un atajo obvio de canal de grabacion "
            "para distinguir humano de sintetico, y que un futuro clasificador tendria que aprender "
            "senales relacionadas con la sintesis de voz en si (prosody, naturalidad, micro-variaciones) "
            "en vez de artefactos de captura." + aviso_borderline
        )

    summary_text = (
        "# Resumen interpretativo (no tecnico)\n\n" + interp + "\n\n"
        "## Detalle de caracteristicas evaluadas\n\n"
        "Se evaluaron seis caracteristicas de bajo nivel del canal, todas independientes del "
        "contenido linguistico: (1) fraccion de energia dentro de la banda telefonica clasica "
        "300-3400 Hz, (2) RMS/loudness de los tramos con habla, (3) zero-crossing rate, (4) "
        "spectral flatness sobre silencio (indicador de ruido tipo codec/cuantizacion), (5) un "
        "score de patrones periodicos tipo 'peine' en el silencio (tipico de codecs de banda "
        "angosta), y (6) la fraccion de energia concentrada en 50/60 Hz y sus armonicos (hum de "
        "linea electrica). Ver `resumen_estadistico.csv` / `.md` para los valores exactos de "
        "p-value y AUC por caracteristica, y las graficas PNG para inspeccion visual.\n"
    )
    with open(OUT_DIR / "resumen_interpretativo.md", "w") as f:
        f.write(summary_text)

    with open(OUT_DIR / "log_ejecucion.txt", "w") as f:
        f.write("\n".join(LOG_LINES))

    log(f"\nTodo listo. Resultados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
