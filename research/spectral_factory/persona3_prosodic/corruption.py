"""
TelephonyAugmenter — implementacion de referencia. Ver CONTRATOS.md seccion 2.

Esto NO es responsabilidad final de Persona 3 (es la Tarea 2.1/2.2 de Persona 2),
pero Persona 3 necesita ALGO con esta firma exacta hoy mismo para poder correr el
experimento de robustez (Tarea 4: "shimmer sigue discriminando despues de
corrupcion?"). Si Persona 2 entrega su clase con la misma firma, se reemplaza el
import en robustness_check.py y esto se vuelve codigo muerto — no rompe nada mas
en el proyecto porque nadie mas importa TelephonyAugmenter salvo ese script.

Por que estos parametros especificamente (para poder defenderlo con un juez):
- G.711 mu-law / A-law: son los codecs de companding estandar de la red
  telefonica publica (PSTN) y de la mayoria de troncales VoIP que la emulan.
  Comprimen 16-bit PCM a 8-bit con una curva logaritmica, perdiendo resolucion
  de amplitud sobre todo en las muestras de mayor energia. mu-law se usa en
  Norteamerica/Japon, A-law en el resto del mundo (incluida Mexico via ITU-T);
  aplicamos ambos con probabilidad igual para no favorecer una region.
- Filtro paso-banda 300-3400 Hz: es el ancho de banda de voz telefonica clasico
  (ITU-T G.711/POTS), definido por los filtros analogicos de la central
  telefonica, no por el codec en si. Es literalmente la caracteristica que nuestro
  analisis forense encontro con AUC~0.82 separando humano/sintetico por si sola
  — este filtro busca igualar esa banda entre las dos clases para destruir el atajo.

AMPLIADO (Fase 1 -> Fase 2, ver HALLAZGOS_LOG.md adenda 3): el stress test sobre
el ZIP real de Persona 2 encontro que ni codec ni filtro paso-banda mueven de
forma significativa NINGUNA rama (evidencia de que ese no es el eje que hay que
reforzar en entrenamiento), pero pitch-shift y time-stretch SI destruyen la rama
prosodica (AUC 0.83 -> ~0.62), algo que no se habia probado antes. Por eso se
agregan aqui como dos familias de corrupcion mas, con la MISMA regla de oro:
simetricas (nunca dependen del label) y aplicadas solo al train fold.

IMPORTANTE -- time_stretch cambia la DURACION del audio. Los turns.json de
referencia (con timestamps de la version original) quedan desalineados tras
un time-stretch. Por eso el meta dict devuelve "time_scale": el factor por el
que hay que multiplicar los timestamps de turns para que sigan alineados
(1.0 para cualquier corrupcion que no altere la duracion). Mismo criterio que
uso Ricardo en su propio pipeline (ver phase1_stress_test/README.md).
"""

from dataclasses import dataclass

import numpy as np
import librosa
from scipy import signal as sps


def mu_law_roundtrip(audio: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Codifica y decodifica mu-law (G.711), dejando el artefacto de cuantizacion
    de 8 bits que introduce el codec, sin cambiar el sample rate."""
    x = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # cuantiza a 8 bits (256 niveles), como hace el codec real
    q = np.round((encoded + 1) / 2 * 255) / 255 * 2 - 1
    decoded = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu))) / mu
    return decoded.astype(np.float64)


def a_law_roundtrip(audio: np.ndarray, A: float = 87.6) -> np.ndarray:
    """Codifica y decodifica A-law (G.711), variante europea/ITU-T."""
    x = np.clip(audio, -1.0, 1.0)
    ax = np.abs(x)
    # np.where evalua ambas ramas en todo el array antes de elegir, asi que
    # log(A*ax) se evalua incluso donde ax=0 (silencio) y dispara un warning de
    # log(0) -- el valor nunca se usa (esa rama no se selecciona ahi), pero se
    # protege igual con un piso para no ensuciar la salida con RuntimeWarning.
    ax_safe = np.maximum(ax, 1e-12)
    encoded = np.where(
        ax < 1 / A,
        A * ax / (1 + np.log(A)),
        (1 + np.log(A * ax_safe)) / (1 + np.log(A)),
    ) * np.sign(x)
    q = np.round((encoded + 1) / 2 * 255) / 255 * 2 - 1
    aq = np.abs(q)
    decoded = np.where(
        aq < (1 / (1 + np.log(A))),
        aq * (1 + np.log(A)) / A,
        np.exp(aq * (1 + np.log(A)) - 1) / A,
    ) * np.sign(q)
    return decoded.astype(np.float64)


def telephone_bandpass(audio: np.ndarray, sr: int, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    """Filtro paso-banda IIR (Butterworth orden 4) que emula el ancho de banda
    telefonico clasico. hi se recorta a 0.49*sr si el audio ya viene a 8kHz para
    no pedirle a scipy una frecuencia de corte por encima de Nyquist."""
    nyq = sr / 2.0
    hi_eff = min(hi, nyq * 0.98)
    sos = sps.butter(4, [lo / nyq, hi_eff / nyq], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, audio)


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Ruido gaussiano aditivo al SNR objetivo (bajo nivel, 15-25 dB por diseno)."""
    sig_power = np.mean(audio ** 2)
    if sig_power <= 0:
        return audio
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
    return audio + noise


def apply_pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Pitch shift +-1 semitono (n_steps en semitonos, puede ser fraccional)."""
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=n_steps).astype(np.float64)


def apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch: rate<1 alarga el audio (mas lento), rate>1 lo acorta.
    OJO: cambia la duracion -- ver time_scale en el meta dict del __call__."""
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate).astype(np.float64)


FAMILIES = ("channel", "pitch_shift", "time_stretch")


@dataclass
class TelephonyAugmenter:
    """Ver CONTRATOS.md seccion 2 para la firma exacta que debe respetar
    cualquier reemplazo de esta clase."""

    p: float = 0.5
    snr_db_range: tuple = (15.0, 25.0)
    pitch_semitone_range: tuple = (-1.0, 1.0)
    time_stretch_rate_range: tuple = (0.9, 1.1)
    families: tuple = FAMILIES  # que familias de corrupcion puede elegir -- se
                                # puede acotar (ej. families=("pitch_shift","time_stretch"))
                                # para un experimento que solo quiera esas dos
    seed: int | None = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        # Nota de diseno critica: esta decision NO puede depender del label
        # (human/synthetic) — la funcion ni siquiera lo recibe como argumento,
        # asi que es fisicamente imposible introducir una firma artificial aqui.
        if self._rng.random() >= self.p:
            return audio, {"corrupted": False, "family": None, "time_scale": 1.0}

        family = self._rng.choice(self.families)

        if family == "channel":
            codec = "mu-law" if self._rng.random() < 0.5 else "a-law"
            out = mu_law_roundtrip(audio) if codec == "mu-law" else a_law_roundtrip(audio)
            out = telephone_bandpass(out, sr)
            snr_db = self._rng.uniform(*self.snr_db_range)
            out = add_gaussian_noise(out, snr_db)
            return out, {"corrupted": True, "family": "channel", "codec": codec,
                         "snr_db": float(snr_db), "time_scale": 1.0}

        if family == "pitch_shift":
            n_steps = self._rng.uniform(*self.pitch_semitone_range)
            out = apply_pitch_shift(audio, sr, n_steps)
            return out, {"corrupted": True, "family": "pitch_shift",
                         "n_steps": float(n_steps), "time_scale": 1.0}

        if family == "time_stretch":
            rate = self._rng.uniform(*self.time_stretch_rate_range)
            out = apply_time_stretch(audio, rate)
            # los turns originales quedan desalineados: multiplicar sus timestamps
            # por 1/rate los realinea a la nueva duracion (mismo criterio que
            # Ricardo uso en su propio pipeline, ver phase1_stress_test/README.md)
            return out, {"corrupted": True, "family": "time_stretch",
                         "rate": float(rate), "time_scale": 1.0 / rate}

        raise ValueError(f"familia de corrupcion desconocida: {family}")


def rescale_turns(turns, time_scale):
    """Aplica el time_scale devuelto por __call__ a una lista de turns (list
    de dicts con channel/start/end) -- necesario SOLO tras time_stretch."""
    if time_scale == 1.0:
        return turns
    return [
        {**t, "start": t["start"] * time_scale, "end": t["end"] * time_scale}
        for t in turns
    ]


if __name__ == "__main__":
    # smoke test rapido con una senoidal de juguete
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    for family in FAMILIES:
        aug = TelephonyAugmenter(p=1.0, families=(family,), seed=0)
        corrupted, meta = aug(tone, sr)
        print(f"[{family}] meta={meta} len_in={len(tone)} len_out={len(corrupted)}")
