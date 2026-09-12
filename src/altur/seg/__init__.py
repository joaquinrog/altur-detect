"""Segmentadores deterministas registrados para el arnés."""

from .vad import energy, oracle, webrtc

__all__ = ["energy", "oracle", "webrtc"]
