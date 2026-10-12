"""Generate three short test .wav files into the ./sounds folder.

Uses only the standard library (wave, math, struct) so it can run
anywhere before pygame is installed.

Outputs:
    sounds/beep.wav   - 880 Hz sine tone, 0.35 s
    sounds/chime.wav  - C6 -> G6 two-note chime with decay, ~0.85 s
    sounds/boop.wav   - descending 520 Hz -> 90 Hz sweep, 0.30 s
"""

import math
import os
import struct
import wave

SAMPLE_RATE = 44100
AMPLITUDE = 0.6 * 32767
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def _envelope(length: int, attack: float = 0.005, release: float = 0.05):
    """Linear attack/release envelope to avoid clicks at start/stop."""
    a = max(1, int(attack * SAMPLE_RATE))
    r = max(1, int(release * SAMPLE_RATE))
    env = []
    for i in range(length):
        if i < a:
            e = i / a
        elif i > length - 1 - r:
            e = max(0.0, (length - 1 - i) / r)
        else:
            e = 1.0
        env.append(e)
    return env


def _write_wav(path: str, samples) -> None:
    """Write mono 16-bit PCM samples to a .wav file."""
    frames = bytearray()
    for s in samples:
        value = int(max(-32767, min(32767, s)))
        frames += struct.pack("<h", value)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(frames))


def make_beep():
    """880 Hz sine tone, 0.35 s."""
    duration = 0.35
    n = int(duration * SAMPLE_RATE)
    env = _envelope(n)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        v = math.sin(2 * math.pi * 880.0 * t)
        samples.append(AMPLITUDE * v * env[i])
    return samples


def make_chime():
    """Two-note chime: C6 (1046.5 Hz) then G6 (1567.98 Hz), exponential decay."""
    notes = [(1046.5, 0.35), (1567.98, 0.50)]
    samples = []
    for freq, duration in notes:
        n = int(duration * SAMPLE_RATE)
        for i in range(n):
            t = i / SAMPLE_RATE
            v = math.sin(2 * math.pi * freq * t)
            v *= math.exp(-3.0 * t)  # pluck-like decay per note
            samples.append(AMPLITUDE * v)
    return samples


def make_boop():
    """Descending pitch sweep 520 Hz -> 90 Hz, 0.30 s, linear fade out."""
    duration = 0.30
    n = int(duration * SAMPLE_RATE)
    f_start, f_end = 520.0, 90.0
    samples = []
    phase = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = f_start + (f_end - f_start) * (t / duration)
        phase += 2 * math.pi * freq / SAMPLE_RATE
        v = math.sin(phase)
        fade = 1.0 - (t / duration)
        samples.append(AMPLITUDE * v * fade)
    return samples


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, maker in (
        ("beep.wav", make_beep),
        ("chime.wav", make_chime),
        ("boop.wav", make_boop),
    ):
        path = os.path.join(OUT_DIR, name)
        _write_wav(path, maker())
        size = os.path.getsize(path)
        print(f"Wrote {path} ({size} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
