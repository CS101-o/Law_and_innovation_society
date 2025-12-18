"""Utility for generating a lightweight waiting tune."""

from __future__ import annotations

import functools
import io
import math
import struct
import wave

SAMPLE_RATE = 22050
TONE_FREQUENCIES = (392.0, 440.0, 523.25)  # G4, A4, C5 for a calm triad
TONE_DURATION = 0.8  # seconds per note
VOLUME = 0.18
FADE_PORTION = 0.15  # apply fade on both ends for gentler tone


def _generate_wave_bytes() -> bytes:
    """Build a short looping-friendly arpeggio as raw WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        frames = []
        for freq in TONE_FREQUENCIES:
            total_samples = int(SAMPLE_RATE * TONE_DURATION)
            fade_samples = int(total_samples * FADE_PORTION)
            for i in range(total_samples):
                envelope = 1.0
                if i < fade_samples:
                    envelope = i / fade_samples
                elif i > total_samples - fade_samples:
                    envelope = (total_samples - i) / fade_samples
                sample = int(
                    VOLUME
                    * envelope
                    * 32767
                    * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))
                )
                frames.append(struct.pack("<h", sample))

        # Longer pause to keep loop gentle
        silence_samples = int(SAMPLE_RATE * 0.8)
        frames.append(b"\x00\x00" * silence_samples)

        wav_file.writeframes(b"".join(frames))
    return buffer.getvalue()


@functools.lru_cache(maxsize=1)
def get_waiting_tune_bytes() -> bytes:
    """Return cached audio bytes for the waiting tune."""
    return _generate_wave_bytes()
