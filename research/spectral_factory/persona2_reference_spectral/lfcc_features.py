"""
LFCC 60-dim (20 estaticos + 20 Delta + 20 DeltaDelta) -- implementacion de
REFERENCIA para la rama espectral. Ver README.md de esta carpeta: esto es un
placeholder documentado, no el entregable real de Persona 2.

Especificacion (igual a la Tarea 2.2 del prompt de Persona 2): ventana de 20ms,
salto de 10ms, a 8kHz.

LFCC vs MFCC (una frase, como pide el propio prompt de Persona 2): LFCC usa un
banco de filtros LINEALMENTE espaciado en frecuencia en vez del banco de
filtros MEL (que comprime las frecuencias altas); en deteccion de voz sintetica
esto importa porque muchos artefactos de vocoder/sintesis se concentran en la
banda alta (>2kHz), justo la zona que un banco Mel resuelve peor por diseno
(Sahidullah & Saha 2012; Todisco et al. 2017, ASVspoof baseline usa CQCC/LFCC
en vez de MFCC por esta razon).
"""

from dataclasses import dataclass

import numpy as np
import librosa
from scipy.fft import dct

SR_EXPECTED = 8000
FRAME_LENGTH_MS = 20
HOP_LENGTH_MS = 10
N_FFT = 256          # >= frame_length en muestras (160 a 8kHz), potencia de 2 mas cercana
N_LINEAR_FILTERS = 20  # -> 20 coeficientes estaticos tras la DCT
N_STATIC = 20
DELTA_WIDTH = 9      # ventana de regresion para delta/delta-delta (default librosa)

# Normalizacion de nivel (ver hallazgo de atajo de volumen en README.md): -20 dBFS
# es una referencia estandar de audio de voz (mismo orden de magnitud que usan
# broadcast/telefonia para AGC). target_rms se deriva de ahi para que quede
# documentado en dB, no como un numero magico en escala lineal.
TARGET_RMS_DBFS = -20.0
TARGET_RMS = 10 ** (TARGET_RMS_DBFS / 20.0)  # = 0.1 en amplitud lineal (full-scale = 1.0)


def _frame_params(sr):
    frame_length = int(sr * FRAME_LENGTH_MS / 1000)
    hop_length = int(sr * HOP_LENGTH_MS / 1000)
    return frame_length, hop_length


def linear_filterbank(sr, n_fft, n_filters=N_LINEAR_FILTERS):
    """Banco de filtros triangulares espaciados LINEALMENTE en Hz (a diferencia
    de librosa.filters.mel, que los espacia en escala mel). No hay equivalente
    directo en librosa para banco lineal, asi que se construye a mano siguiendo
    el mismo esquema de filtros triangulares superpuestos que usa MFCC."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    edges = np.linspace(0, sr / 2, n_filters + 2)  # n_filters triangulos -> n_filters+2 bordes

    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        lo, center, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs >= lo) & (freqs <= center)
        falling = (freqs > center) & (freqs <= hi)
        if center > lo:
            fb[i, rising] = (freqs[rising] - lo) / (center - lo)
        if hi > center:
            fb[i, falling] = (hi - freqs[falling]) / (hi - center)
    return fb


def extract_lfcc(audio, sr):
    """LFCC estatico (n_frames, N_STATIC) sobre el audio completo (ya se espera
    que solo se le pasen los tramos de habla, ver build_spectral_dataset.py)."""
    frame_length, hop_length = _frame_params(sr)
    n_fft = max(N_FFT, frame_length)

    stft = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length,
                         win_length=frame_length, window="hamming")
    power_spec = np.abs(stft) ** 2  # (n_bins, n_frames)

    fb = linear_filterbank(sr, n_fft, N_LINEAR_FILTERS)
    filtered = fb @ power_spec  # (n_filters, n_frames)
    log_filtered = np.log(filtered + 1e-10)

    lfcc = dct(log_filtered, type=2, axis=0, norm="ortho")[:N_STATIC, :]  # (N_STATIC, n_frames)
    return lfcc.T  # (n_frames, N_STATIC), tiempo en filas para que sea facil de concatenar entre turnos


def extract_lfcc_60dim(audio, sr):
    """LFCC + Delta + DeltaDelta -> (n_frames, 60). Si hay muy pocos frames
    para calcular delta con la ventana default, se reduce la ventana en vez de
    fallar (llamadas cortas / turnos cortos)."""
    static = extract_lfcc(audio, sr)  # (n_frames, 20)
    n_frames = static.shape[0]
    if n_frames < 3:
        return None  # insuficiente para delta/delta-delta de forma significativa

    width = DELTA_WIDTH
    if width >= n_frames:
        width = n_frames - 1 if (n_frames - 1) % 2 == 1 else n_frames - 2
        width = max(3, width)

    static_t = static.T  # librosa.feature.delta espera (n_features, n_frames)
    delta = librosa.feature.delta(static_t, width=width, order=1)
    delta2 = librosa.feature.delta(static_t, width=width, order=2)

    full = np.concatenate([static_t, delta, delta2], axis=0)  # (60, n_frames)
    return full.T  # (n_frames, 60)


@dataclass
class SpectralLatent:
    anon_id: str
    n_frames: int
    features: dict


def pool_lfcc(lfcc_60, prefix_names=None):
    """Pooling estadistico (media + std) por dimension -> 120 features fijas
    independientemente de la duracion de la llamada. Mismo criterio de diseno
    que la rama prosodica: dataset chico (353 llamadas), pooling simple
    generaliza mejor que un backbone/pooling aprendido -- ver README.md de esta
    carpeta para por que esto es aceptable en un PLACEHOLDER pero no sustituye
    a un backbone LCNN/ECAPA real."""
    if lfcc_60 is None or lfcc_60.shape[0] == 0:
        return None
    mean = lfcc_60.mean(axis=0)
    std = lfcc_60.std(axis=0)
    names = prefix_names or [f"lfcc_{i}" for i in range(lfcc_60.shape[1])]
    feats = {}
    for i, name in enumerate(names):
        feats[f"{name}_mean"] = float(mean[i])
        feats[f"{name}_std"] = float(std[i])
    return feats


def _lfcc_dim_names():
    names = [f"static{i}" for i in range(N_STATIC)]
    names += [f"delta{i}" for i in range(N_STATIC)]
    names += [f"deltadelta{i}" for i in range(N_STATIC)]
    return names


def extract_spectral_latent(anon_id, audio, sr, turns_ch0):
    """Pipeline completo: audio+turns (VAD oracle) -> LFCC 60-dim por frame ->
    pooling -> vector latente de 120 features.

    IMPORTANTE (hallazgo del propio equipo, ver README.md de esta carpeta):
    normalizamos el RMS del audio de habla del CANAL 0 (el llamante, el unico
    que se clasifica) a un nivel objetivo fijo de -20 dBFS, ANTES de calcular
    LFCC. Sin esto, el coeficiente estatico 0 (log-energia de la banda de
    frecuencia mas baja) separaba human/synthetic con AUC~0.88 -- casi con
    seguridad un artefacto de nivel/normalizacion de grabacion (~9dB de
    diferencia promedio entre clases: humano ~-24 dBFS via atenuacion de la
    red PSTN/movil, sintetico ~-15 dBFS por venir exportado digitalmente sin
    esa perdida), NO una diferencia de sintesis real.

    Dos decisiones de diseno deliberadas sobre COMO normalizar:
    1. El calculo de RMS usa SOLO los tramos de habla del propio Canal 0
       (turns_ch0, VAD oracle), nunca el clip completo -- si se incluyera
       silencio/ruido de fondo, la estimacion de nivel se contaminaria con
       exactamente el mismo piso de ruido de canal que ya se identifico como
       sospechoso en el analisis forense original.
    2. La normalizacion es respecto a un valor ABSOLUTO fijo (-20 dBFS), NO
       respecto al Canal 1 (el agente). Usar la razon Canal0/Canal1 como
       referencia reintroduciria la misma variable de ganancia por otra
       puerta: el Canal 1 viene del servidor del banco a volumen practicamente
       constante, asi que dividir por el no cancela nada nuevo, solo agrega
       una fuente de varianza distinta sin resolver el problema real (el nivel
       absoluto del Canal 0 en si mismo)."""
    frame_length, hop_length = _frame_params(sr)
    chunks = []
    for turn in turns_ch0:
        i0, i1 = int(turn["start"] * sr), int(turn["end"] * sr)
        i0, i1 = max(0, i0), min(len(audio), i1)
        if i1 - i0 >= frame_length:
            chunks.append(audio[i0:i1])
    if not chunks:
        return None

    concat_speech = np.concatenate(chunks)
    speech_rms = np.sqrt(np.mean(concat_speech.astype(np.float64) ** 2))
    gain = (TARGET_RMS / speech_rms) if speech_rms > 1e-8 else 1.0
    chunks = [chunk * gain for chunk in chunks]

    all_frames = []
    for chunk in chunks:
        feats = extract_lfcc_60dim(chunk, sr)
        if feats is not None:
            all_frames.append(feats)
    if not all_frames:
        return None

    full = np.concatenate(all_frames, axis=0)
    pooled = pool_lfcc(full, prefix_names=_lfcc_dim_names())
    if pooled is None:
        return None
    return SpectralLatent(anon_id=anon_id, n_frames=full.shape[0], features=pooled)
