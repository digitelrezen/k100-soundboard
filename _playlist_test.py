"""End-to-end test: playlist persistence + monitor device dropdown."""
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import tkinter as tk

import soundboard as sb

# Seed a saved playlist with two real files.
sb.save_assignments({
    "G1": "beep.wav", "G2": "", "G3": "chime.wav",
    "G4": "", "G5": "", "G6": "",
})

root = tk.Tk()
app = sb.SoundboardApp(root)


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.05)


pump(6)  # let restore + decode threads finish

# 1) Saved playlist restored into combos + engine
assert app.assign_vars["G1"].get() == "beep.wav", f"G1 combo: {app.assign_vars['G1'].get()!r}"
assert app.assign_vars["G3"].get() == "chime.wav", f"G3 combo: {app.assign_vars['G3'].get()!r}"
assert app.assignments["G1"].endswith("beep.wav"), app.assignments["G1"]
assert app.assignments["G3"].endswith("chime.wav"), app.assignments["G3"]
assert app.engine.slot_arrays["G1"] is not None, "G1 not decoded after restore"
assert app.engine.slot_arrays["G3"] is not None, "G3 not decoded after restore"
assert app.engine.slot_arrays["G2"] is None, "G2 should be empty"
print("PASS: saved playlist restored (G1=beep.wav, G3=chime.wav) and decoded")

# 2) Monitor dropdown offers SSL 2+ outputs (not just headphones)
mon_values = list(app.dev_combos["headphone"]["values"])
print("monitor options:")
for v in mon_values:
    print("   ", v)
assert any("Output 1/2 (SSL 2+" in v for v in mon_values), "SSL Output 1/2 missing from monitor list"
assert any("Headset Earphone" in v for v in mon_values), "headset missing from monitor list"
print("PASS: monitor dropdown includes SSL 2+ Output 1/2 and headset")

# 3) User change is persisted
app.assign_vars["G2"].set("chime.wav")
app._on_select("G2")
pump(3)
saved = sb.load_assignments()
assert saved["G2"] == "chime.wav", f"saved G2: {saved['G2']!r}"
assert app.engine.slot_arrays["G2"] is not None, "G2 not decoded after selection"
print("PASS: selection change persisted to assignments.json (G2=chime.wav)")

# 4) Missing file in saved playlist is cleared cleanly
sb.save_assignments({
    "G1": "beep.wav", "G2": "chime.wav", "G3": "does_not_exist.mp3",
    "G4": "", "G5": "", "G6": "",
})
app.saved_names = sb.load_assignments()
app.refresh_files()
pump(3)
assert app.assign_vars["G3"].get() == "", f"G3 combo: {app.assign_vars['G3'].get()!r}"
saved = sb.load_assignments()
assert saved["G3"] == "", f"saved G3: {saved['G3']!r}"
assert saved["G1"] == "beep.wav" and saved["G2"] == "chime.wav"
print("PASS: missing saved file cleared from combo and assignments.json")

root.destroy()
print("PLAYLIST TEST PASSED")