"""Headless smoke test for the software-mixing engine (no GUI).

Starts the AudioEngine with a real mic, headphone, and virtual-cable device,
records the virtual cable's capture side while triggering G1, and verifies
that the triggered sound actually reaches the cable output (i.e. what
Discord would receive).
"""

import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import sounddevice as sd

import soundboard as sb


def find_device(pred, kind, hostapi):
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != hostapi:
            continue
        ch = d["max_input_channels"] if kind == "in" else d["max_output_channels"]
        if ch > 0 and pred(d["name"]):
            return i
    return None


def main():
    wasapi = next(i for i, h in enumerate(sd.query_hostapis())
                  if "wasapi" in h["name"].lower())
    mic = find_device(lambda n: "SSL" in n.upper(), "in", wasapi)
    hp = find_device(lambda n: "Headset Earphone" in n, "out", wasapi)
    cable_out = find_device(lambda n: "CABLE INPUT" in n.upper(), "out", wasapi)
    cable_in = find_device(lambda n: "CABLE OUTPUT" in n.upper(), "in", wasapi)
    print(f"mic={mic} headphone={hp} cable_out={cable_out} cable_in={cable_in}")
    assert mic is not None and hp is not None
    assert cable_out is not None and cable_in is not None

    engine = sb.AudioEngine(status_cb=lambda m: print("  [engine]", m))
    engine.mic_device = mic
    engine.headphone_device = hp
    engine.cable_device = cable_out

    # The engine will run at the cable's native rate; decode the beep at it too.
    rate = sb.native_rate(cable_out, "output") or sb.ENGINE_RATE

    # Record what the cable's capture side receives (= what Discord gets).
    rec_rate = sb.native_rate(cable_in, "input") or 48000
    rec = {"buf": None, "n": 0}

    def rec_cb(indata, frames, t, status):
        if rec["buf"] is None:
            rec["buf"] = np.zeros((rec_rate * 3, indata.shape[1]), dtype=np.float32)
        end = min(rec["n"] + frames, len(rec["buf"]))
        rec["buf"][rec["n"]:end] = indata[: end - rec["n"]]
        rec["n"] = end

    rec_stream = sd.InputStream(
        device=cable_in, channels=1, samplerate=rec_rate,
        dtype="float32", blocksize=1024, callback=rec_cb,
    )

    beep = os.path.join(sb.SOUNDS_DIR, "beep.wav")
    arr = engine.load_sound("G1", beep, rate)
    print(f"loaded G1 @ {rate} Hz: {arr.shape} peak={np.abs(arr).max():.3f}")

    engine.start()
    rec_stream.start()
    time.sleep(0.4)
    ok = engine.trigger("G1")
    print("trigger G1 ->", ok)
    time.sleep(1.5)
    rec_stream.stop()
    rec_stream.close()
    engine.stop()

    data = rec["buf"][:rec["n"]]
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.abs(data).max())
    print(f"recorded {data.shape[0]} frames ({data.shape[0]/rec_rate:.2f}s) "
          f"RMS={rms:.5f} peak={peak:.4f}")
    # beep.wav peaks at ~0.6; expect a clearly audible signal on the cable.
    assert ok, "trigger failed"
    assert peak > 0.2, "beep did NOT reach the virtual cable output!"
    print("SMOKE TEST PASSED: G1 sound mixed into the virtual cable output")


if __name__ == "__main__":
    main()