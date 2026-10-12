"""Verify mic audio passes through the engine to the virtual cable unprocessed."""

import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import sounddevice as sd

import soundboard as sb


def main():
    devs = sd.query_devices()
    wasapi = next(i for i, h in enumerate(sd.query_hostapis())
                  if "wasapi" in h["name"].lower())
    mic = next(i for i, d in enumerate(devs)
               if d["hostapi"] == wasapi and d["max_input_channels"] > 0
               and "SSL" in d["name"].upper())
    cin = next(i for i, d in enumerate(devs)
               if d["hostapi"] == wasapi and d["max_input_channels"] > 0
               and "CABLE OUTPUT" in d["name"].upper())
    cout = next(i for i, d in enumerate(devs)
                if d["hostapi"] == wasapi and d["max_output_channels"] > 0
                and "CABLE INPUT" in d["name"].upper())

    engine = sb.AudioEngine()
    engine.mic_device = mic
    engine.headphone_device = None  # cable only, for this test
    engine.cable_device = cout

    rec = {"buf": []}

    def cb(indata, frames, t, status):
        rec["buf"].append(indata.copy())

    rs = sd.InputStream(device=cin, channels=1, samplerate=48000,
                        dtype="float32", blocksize=512, callback=cb)
    engine.start()
    rs.start()
    time.sleep(2.0)
    rs.stop()
    rs.close()
    engine.stop()

    data = np.concatenate(rec["buf"])
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.abs(data).max())
    print(f"frames: {data.shape[0]}  RMS: {rms:.6f}  peak: {peak:.5f}")
    # Room noise on a live SSL 2+ input should be well above digital silence.
    assert rms > 1e-4, "no mic audio reached the cable — passthrough broken!"
    print("MIC PASSTHROUGH OK: live mic audio reached the virtual cable")


if __name__ == "__main__":
    main()