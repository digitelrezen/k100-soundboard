"""Verify the push-to-talk mic gate: muted when released, live when held."""
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
    engine.headphone_device = None
    engine.cable_device = cout

    rec = {"buf": []}

    def cb(indata, frames, t, status):
        rec["buf"].append(indata.copy())

    def capture(seconds):
        rec["buf"] = []
        rs = sd.InputStream(device=cin, channels=1, samplerate=48000,
                            dtype="float32", blocksize=512, callback=cb)
        rs.start()
        time.sleep(seconds)
        rs.stop()
        rs.close()
        data = np.concatenate(rec["buf"])
        return float(np.sqrt(np.mean(np.square(data)))), float(np.abs(data).max())

    engine.start()
    try:
        # 1) PTT enabled, key not held -> mic muted on the cable
        engine.ptt_enabled = True
        engine.ptt_active = False
        time.sleep(0.3)  # let the gate ramp fully closed
        rms_muted, _ = capture(1.5)

        # 2) PTT enabled, key held -> mic live on the cable
        engine.ptt_active = True
        time.sleep(0.3)  # let the gate ramp fully open
        rms_live, _ = capture(1.5)

        # 3) PTT disabled -> mic always live
        engine.ptt_enabled = False
        time.sleep(0.3)
        rms_always, _ = capture(1.5)

        print(f"rms muted={rms_muted:.6f}  live={rms_live:.6f}  always={rms_always:.6f}")
        assert rms_live > 1e-4, "mic not live while PTT held!"
        assert rms_always > 1e-4, "mic not live with PTT disabled!"
        assert rms_muted < max(1e-4, rms_live * 0.1), \
            f"mic not muted while PTT released (muted={rms_muted}, live={rms_live})"
        print("PTT GATE TEST PASSED: mic muted when released, live when held")
    finally:
        engine.stop()

    # 4) settings round-trip
    sb.save_settings({"ptt_enabled": True, "ptt_key": "Space"})
    assert sb.load_settings() == {"ptt_enabled": True, "ptt_key": "Space"}
    print("SETTINGS ROUND-TRIP OK")


if __name__ == "__main__":
    main()