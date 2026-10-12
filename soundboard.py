"""K100 Soundboard — all-in-one software mixing soundboard.

Maps the six G-Keys on a Corsair K100 (bound to F13-F18 in iCUE) to audio
files kept in the ./sounds folder, and mixes everything in software:

    microphone  ─┐
                 ├─► [software mixer @48 kHz] ─► headphone monitor (you hear it)
    G1..G6 audio ─┘                              └► virtual cable (Discord picks it up)

How it works:
  - sounddevice (PortAudio) streams with non-blocking callbacks. The mic input
    callback feeds a float32 ring buffer; a mixer thread sums mic + triggered
    sounds into a shared mix ring buffer; two output streams (headphones and
    virtual cable) both read that mix, so Discord and your ears get the exact
    same mix.
  - The engine runs at the virtual cable device's native sample rate (queried
    via sounddevice.query_devices); sounds are resampled to that rate with
    scipy.signal on load, so PortAudio never resamples the Discord path.
  - numpy does all buffer math in float32. The mic passes through unprocessed
    at unity gain (adjustable 0-200%) and a linear safety limiter + hard clip
    keep the mix inside [-1, 1], so the voice never gets quiet or muffled.
  - pygame (pygame.sndarray) decodes .wav/.mp3/.ogg to float32 stereo.
  - pynput global hotkey listener (F13-F18) in a daemon thread.
  - Push-to-talk: hold the PTT key (configurable in the GUI) to open the mic
    path; soundboard triggers are always on. The mic gate ramps over ~10 ms
    so opening/closing never clicks. Set Discord's input mode to Voice
    Activity (not Push to Talk) so the cable stream is always transmitted.
  - The engine auto-starts ~0.3 s after the GUI opens; the Start/Stop button
    still toggles it at any time.

Usage:
    python soundboard.py
"""

import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

# Silence pygame's startup banner before importing it.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import sounddevice as sd
from pynput import keyboard
from scipy.signal import resample_poly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
ASSIGNMENTS_FILE = os.path.join(BASE_DIR, "assignments.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
AUDIO_EXTS = {".wav", ".mp3", ".ogg"}
SLOTS = [f"G{i}" for i in range(1, 7)]
HOTKEYS = {
    "G1": keyboard.Key.f13,
    "G2": keyboard.Key.f14,
    "G3": keyboard.Key.f15,
    "G4": keyboard.Key.f16,
    "G5": keyboard.Key.f17,
    "G6": keyboard.Key.f18,
}
KEY_TO_SLOT = {key: slot for slot, key in HOTKEYS.items()}

# Push-to-talk key options: (label shown in the GUI, predicate for a pynput
# key event). Letters arrive as KeyCode with .char; special keys as Key.
# F13-F18 are taken by the G-Keys, so F1-F12 are free for PTT.
def _ptt_label(spec: str) -> str:
    if spec.isalpha() and len(spec) == 1:
        return spec.upper()
    if spec == "space":
        return "Space"
    if "_" in spec:  # e.g. "shift_l" -> "Left Shift"
        mod, side = spec.split("_")
        return f"{side.title()} {mod.title()}"
    return spec.upper()  # "f1" -> "F1"


def _ptt_match(spec: str):
    if spec.isalpha() and len(spec) == 1:
        ch = spec.lower()
        return lambda key: str(getattr(key, "char", "")).lower() == ch
    target = getattr(keyboard.Key, spec)
    return lambda key: key == target


PTT_KEY_SPECS = (
    ["v", "b", "n", "m", "space", "shift_l", "shift_r", "ctrl_l", "ctrl_r"]
    + [f"f{i}" for i in range(1, 13)]
)
PTT_KEY_OPTIONS = [(_ptt_label(s), _ptt_match(s)) for s in PTT_KEY_SPECS]

DECODE_RATE = 48000     # rate pygame decodes files at (before resampling)
ENGINE_RATE = 48000     # fallback engine rate if the cable device's native rate is unknown
BLOCKSIZE = 1024        # frames per audio block (~23 ms at 44.1 kHz) — large enough to avoid underruns
RING_SECONDS = 5        # mic ring buffer capacity
MIX_RING_SECONDS = 2    # shared mix ring buffer capacity


def native_rate(dev, kind):
    """Native (default) sample rate of a device via sounddevice.query_devices()."""
    if dev is None:
        return None
    try:
        info = sd.query_devices(dev, kind)
        return int(info["default_samplerate"])
    except Exception:
        return None


def wasapi_hostapi():
    """Index of the WASAPI host API (cleanest, lowest-latency path on Windows)."""
    try:
        for i, h in enumerate(sd.query_hostapis()):
            if "wasapi" in h["name"].lower():
                return i
    except Exception:
        pass
    return None


def list_devices(kind: str):
    """List (label, index) pairs for 'input'/'output' devices.

    WASAPI entries only — each physical device otherwise appears 3-4 times
    (MME / DirectSound / WASAPI / WDM-KS), and MME is the lowest-quality path.
    Falls back to all host APIs if the system has no WASAPI devices.
    Labels include channel count and native rate so entries are easy to tell apart.
    """
    ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devs = sd.query_devices()
    wasapi = wasapi_hostapi()

    def label(i, d):
        rate = int(d["default_samplerate"])
        return f"{d['name']}  [{int(d[ch_key])}ch @ {rate // 1000} kHz]"

    entries = [
        (label(i, d), i)
        for i, d in enumerate(devs)
        if int(d[ch_key]) > 0 and (wasapi is None or d["hostapi"] == wasapi)
    ]
    if not entries:  # no WASAPI devices at all — fall back to everything
        entries = [(label(i, d), i) for i, d in enumerate(devs) if int(d[ch_key]) > 0]
    return entries


def find_cable_capture(out_idx):
    """Find the capture-side device matching a virtual-cable output device.

    Matches on the parenthesized product name, channel count, and rate, e.g.
    'CABLE Input (VB-Audio Virtual Cable)' -> 'CABLE Output (VB-Audio Virtual Cable)'.
    Returns the device index, or None.
    """
    if out_idx is None:
        return None
    try:
        out = sd.query_devices(out_idx)
    except Exception:
        return None
    if "cable" not in out["name"].lower():
        return None
    m = re.search(r"\(([^)]*)\)\s*$", out["name"])
    suffix = m.group(1).upper() if m else ""
    rate = int(out["default_samplerate"])
    ch = int(out["max_output_channels"])
    for i, d in enumerate(sd.query_devices()):
        if int(d["max_input_channels"]) <= 0:
            continue
        name = d["name"].upper()
        if "CABLE" not in name or "OUT" not in name:
            continue
        if suffix and suffix not in name:
            continue
        if int(d["default_samplerate"]) != rate or int(d["max_input_channels"]) != ch:
            continue
        return i
    return None


def load_assignments() -> dict[str, str]:
    """Load the saved slot -> sound file name mapping ({} if none or corrupt)."""
    try:
        with open(ASSIGNMENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {slot: data[slot] for slot in SLOTS if isinstance(data.get(slot), str)}


def save_assignments(names: dict[str, str]):
    """Persist the slot -> sound file name mapping (best effort)."""
    try:
        with open(ASSIGNMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump({slot: names.get(slot, "") for slot in SLOTS}, f, indent=2)
    except OSError:
        pass


def load_settings() -> dict:
    """Load app settings (PTT enabled/key) — {} if none or corrupt."""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: dict):
    """Persist app settings (best effort)."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def decode_sound(path: str, target_rate: int) -> np.ndarray:
    """Decode an audio file to float32 stereo at target_rate.

    The file is decoded by pygame at DECODE_RATE, resampled to target_rate
    with scipy.signal (polyphase FIR, properly anti-aliased), normalized to
    stereo, and clipped to [-1.0, 1.0]. Returns shape (samples, 2).
    """
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=DECODE_RATE, size=-16, channels=2, buffer=1024)
    snd = pygame.mixer.Sound(path)
    arr = pygame.sndarray.array(snd)  # int16, shape (N, C) at DECODE_RATE
    arr = arr.astype(np.float32) / 32768.0
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)  # mono -> stereo
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    if int(target_rate) != DECODE_RATE:
        arr = resample_poly(arr, up=int(target_rate), down=DECODE_RATE)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


class RingBuffer:
    """Float32 ring buffer of (frames, channels). Caller must hold the lock."""

    def __init__(self, capacity: int, channels: int):
        self.capacity = capacity
        self.channels = channels
        self.buf = np.zeros((capacity, channels), dtype=np.float32)
        self.pos = 0

    def write(self, data: np.ndarray):
        n = len(data)
        if n == 0:
            return
        if n >= self.capacity:
            self.buf[:] = data[-self.capacity:]
            self.pos = 0
            return
        end = self.pos + n
        if end <= self.capacity:
            self.buf[self.pos:end] = data
            self.pos = end
        else:
            first = self.capacity - self.pos
            self.buf[self.pos:] = data[:first]
            self.buf[: end - self.capacity] = data[first:]
            self.pos = end - self.capacity

    def read(self, n: int) -> np.ndarray:
        """Return the most recent n frames (zero-padded if nothing was written)."""
        if self.pos == 0:
            return np.zeros((n, self.channels), dtype=np.float32)
        if n >= self.capacity:
            n = self.capacity
        if self.pos >= n:
            return self.buf[self.pos - n: self.pos].copy()
        part1 = self.buf[self.capacity - (n - self.pos):]
        part2 = self.buf[: self.pos]
        return np.concatenate((part1, part2))


class AudioEngine:
    """Software mixer: mic + G-key sounds -> headphone output + virtual cable.

    Threads:
      * PortAudio input callback  -> writes mic frames into mic_ring
      * mixer thread              -> reads mic_ring + players, writes mix_ring
      * PortAudio output callbacks-> read mix_ring into each output stream
    """

    def __init__(self, status_cb=None):
        self.status_cb = status_cb
        self.lock = threading.Lock()       # mic ring + players + slot arrays
        self.mix_lock = threading.Lock()   # mix ring
        self.mic_ring = RingBuffer(ENGINE_RATE * RING_SECONDS, 2)
        self.mix_ring = RingBuffer(ENGINE_RATE * MIX_RING_SECONDS, 2)
        self.players: list[list] = []      # [buffer (N,2) float32, pos (float)]
        self.slot_arrays: dict[str, np.ndarray | None] = {slot: None for slot in SLOTS}
        self.rate = ENGINE_RATE        # active engine rate (set in start())
        self.decode_rate = ENGINE_RATE  # rate the cached slot arrays are at
        self.mic_gain = 1.0
        self.sb_gain = 1.0
        self.ptt_enabled = False   # gate the mic behind the PTT key (set by GUI)
        self.ptt_active = False    # PTT key currently held (set by listener thread)
        self._mic_gate = 1.0       # smoothed 0..1 mic gate (mixer thread only)
        self.mic_device = None
        self.headphone_device = None
        self.cable_device = None
        self.mic_stream = None
        self.headphone_stream = None
        self.cable_stream = None
        self._mixer_thread: threading.Thread | None = None
        self._mixing = False

    # ------------------------------------------------------------- lifecycle

    @property
    def running(self) -> bool:
        return self._mixer_thread is not None and self._mixer_thread.is_alive()

    def start(self):
        if self.running:
            return
        # Run the whole engine at the cable (Discord) device's native rate so
        # PortAudio never has to resample the outgoing stream.
        self.rate = native_rate(self.cable_device, "output") or ENGINE_RATE
        with self.mix_lock:
            self.mix_ring = RingBuffer(self.rate * MIX_RING_SECONDS, 2)
        self._mixing = True
        self._mixer_thread = threading.Thread(
            target=self._mixer_loop, daemon=True, name="mixer"
        )
        self._mixer_thread.start()
        self.restart_mic()
        self.restart_output("headphone")
        self.restart_output("cable")
        self._log(f"Engine running @ {self.rate} Hz — press Start/Stop to toggle")

    def stop(self):
        self._mixing = False
        if self._mixer_thread is not None:
            self._mixer_thread.join(timeout=1.0)
            self._mixer_thread = None
        self._close_stream("mic")
        self._close_stream("headphone")
        self._close_stream("cable")
        with self.lock:
            self.players = []
            self.mic_ring = RingBuffer(ENGINE_RATE * RING_SECONDS, 2)
        with self.mix_lock:
            self.mix_ring = RingBuffer(ENGINE_RATE * MIX_RING_SECONDS, 2)
        self._log("Engine stopped")

    def _close_stream(self, which: str):
        stream = getattr(self, f"{which}_stream")
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        setattr(self, f"{which}_stream", None)

    # --------------------------------------------------------------- streams

    def restart_mic(self):
        self._close_stream("mic")
        dev = self.mic_device
        if dev is None:
            self._log("Microphone: no device selected")
            return
        try:
            info = sd.query_devices(dev, "input")
            channels = 2 if info["max_input_channels"] >= 2 else 1
            with self.lock:
                self.mic_ring = RingBuffer(self.rate * RING_SECONDS, 2)
            stream = sd.InputStream(
                device=dev,
                channels=channels,
                samplerate=self.rate,
                dtype="float32",
                blocksize=BLOCKSIZE,
                callback=self._mic_cb,
            )
            stream.start()
            self.mic_stream = stream
            self._log(f"Microphone: {info['name']}")
        except Exception as exc:
            self._log(f"Microphone failed: {exc}")

    def restart_output(self, which: str):
        """which: 'headphone' or 'cable'."""
        self._close_stream(which)
        dev = self.headphone_device if which == "headphone" else self.cable_device
        if dev is None:
            self._log(f"{which}: no device selected")
            return
        try:
            info = sd.query_devices(dev, "output")
            stream = sd.OutputStream(
                device=dev,
                channels=2,
                samplerate=self.rate,
                dtype="float32",
                blocksize=BLOCKSIZE,
                callback=self._out_cb,
            )
            stream.start()
            setattr(self, f"{which}_stream", stream)
            self._log(f"{which.capitalize()}: {info['name']}")
        except Exception as exc:
            self._log(f"{which} output failed: {exc}")

    # ----------------------------------------------------------- audio thread

    def _mixer_loop(self):
        """Sum mic + active sounds into mix_ring once per block, in real time."""
        period = BLOCKSIZE / self.rate
        next_t = time.perf_counter()
        while self._mixing:
            n = BLOCKSIZE
            with self.lock:
                # Push-to-talk gate: ramp toward 0/1 over ~10 ms (click-free).
                # While PTT is on and the key is not held, the mic is muted but
                # still captured into the ring, so pressing the key is instant.
                target = 1.0 if (not self.ptt_enabled or self.ptt_active) else 0.0
                step = period / 0.010
                if self._mic_gate < target:
                    self._mic_gate = min(target, self._mic_gate + step)
                elif self._mic_gate > target:
                    self._mic_gate = max(target, self._mic_gate - step)
                mix = self.mic_ring.read(n) * (self.mic_gain * self._mic_gate)
                alive = []
                for player in self.players:
                    buf, pos = player
                    start = int(pos)
                    if start >= len(buf):
                        continue
                    end = min(start + n, len(buf))
                    mix[: end - start] += buf[start:end] * self.sb_gain
                    player[1] = pos + n
                    if end < len(buf):
                        alive.append(player)
                self.players = alive
                # Linear safety limiter: only touches the mix when it would clip.
                peak = float(np.abs(mix).max())
                if peak > 1.0:
                    mix = mix / peak
                # Hard safety clip before writing to the float32 mix buffer.
                mix = np.clip(mix, -1.0, 1.0)
            with self.mix_lock:
                self.mix_ring.write(mix)
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            # else: we fell behind — produce the next block immediately to
            # catch up (the mix ring absorbs short bursts; never skip audio).

    def _mic_cb(self, indata, frames, time_info, status):
        if status:
            self._log(f"mic: {status}")
        data = indata if indata.shape[1] >= 2 else indata[:, :1]
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        with self.lock:
            self.mic_ring.write(data)

    def _out_cb(self, outdata, frames, time_info, status):
        if status:
            self._log(f"output: {status}")
        with self.mix_lock:
            data = self.mix_ring.read(frames)
        outdata[:] = data

    # ----------------------------------------------------------------- sounds

    def load_sound(self, slot: str, path: str, rate: int) -> np.ndarray:
        """Decode + resample a file and cache it for the slot (thread-safe)."""
        arr = decode_sound(path, rate)
        with self.lock:
            self.slot_arrays[slot] = arr
            self.decode_rate = int(rate)
        return arr

    def trigger(self, slot: str) -> bool:
        """Start (or overlap) the slot's sound in the outgoing mix."""
        with self.lock:
            arr = self.slot_arrays.get(slot)
            if arr is None or arr.size == 0:
                return False
            self.players.append([arr, 0.0])
        return True

    # ------------------------------------------------------------------ misc

    def _log(self, msg: str):
        if self.status_cb:
            self.status_cb(msg)


class SoundboardApp:
    """Owns the GUI, the hotkey listener, and the audio engine."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("K100 Soundboard — Software Mixer")
        root.resizable(False, False)

        self.sounds_dir = SOUNDS_DIR
        os.makedirs(self.sounds_dir, exist_ok=True)
        self.files: list[str] = []  # full paths of valid audio files
        self.assignments: dict[str, str] = {slot: "" for slot in SLOTS}
        self.saved_names: dict[str, str] = load_assignments()  # persisted playlist
        self.assign_vars: dict[str, tk.StringVar] = {}
        self.combos: dict[str, ttk.Combobox] = {}
        self.dev_vars: dict[str, tk.StringVar] = {}
        self.dev_combos: dict[str, ttk.Combobox] = {}
        self.dev_indices: dict[str, dict[str, int]] = {}
        self.gain_vars: dict[str, tuple[ttk.Scale, tk.StringVar]] = {}

        self._lock = threading.Lock()  # guards self.assignments
        self._ui_q: queue.Queue = queue.Queue()  # callables for the UI thread
        self._listener: keyboard.Listener | None = None

        # Push-to-talk settings (persisted in settings.json).
        settings = load_settings()
        self.ptt_var = tk.BooleanVar(value=bool(settings.get("ptt_enabled", True)))
        ptt_label = settings.get("ptt_key", PTT_KEY_OPTIONS[0][0])
        self.ptt_idx = next(
            (i for i, (lab, _k) in enumerate(PTT_KEY_OPTIONS) if lab == ptt_label), 0
        )
        self._ptt_match = PTT_KEY_OPTIONS[self.ptt_idx][1]

        self.engine = AudioEngine(status_cb=self._status_q_put)
        self.engine.ptt_enabled = self.ptt_var.get()
        self._init_mixer()
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)  # 1 ms timer for the mixer thread
            except Exception:
                pass
        self._build_ui()
        self._populate_devices()
        self._start_listener()
        self.refresh_files()
        self._poll_ui()
        # Auto-start the engine a fraction of a second after the GUI is up.
        self.root.after(300, self._start_engine)

    # ------------------------------------------------------------------ audio

    def _init_mixer(self):
        try:
            pygame.mixer.init(frequency=DECODE_RATE, size=-16, channels=2, buffer=1024)
        except pygame.error as exc:
            self._status_q_put(f"Audio mixer unavailable: {exc}")

    # --------------------------------------------------------------------- UI

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        dev_frame = ttk.LabelFrame(self.root, text="Audio Devices")
        dev_frame.pack(fill="x", **pad)
        self._device_row(dev_frame, "Microphone Input", "mic")
        self._device_row(dev_frame, "Monitor Output (you hear it)", "headphone")
        self._device_row(dev_frame, "Virtual Cable Output (Discord)", "cable")
        self.cable_hint_var = tk.StringVar(value="")
        ttk.Label(
            dev_frame, textvariable=self.cable_hint_var, foreground="#0a6", anchor="w"
        ).pack(fill="x", padx=8, pady=(0, 4))

        ptt_frame = ttk.LabelFrame(self.root, text="Push-to-Talk (mic)")
        ptt_frame.pack(fill="x", **pad)
        ptt_row = ttk.Frame(ptt_frame)
        ptt_row.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(
            ptt_row,
            text="Mic only while key held",
            variable=self.ptt_var,
            command=self._on_ptt_toggle,
        ).pack(side="left")
        ttk.Label(ptt_row, text="  Key:").pack(side="left")
        self.ptt_combo = ttk.Combobox(
            ptt_row,
            values=[lab for lab, _k in PTT_KEY_OPTIONS],
            state="readonly",
            width=12,
        )
        self.ptt_combo.set(PTT_KEY_OPTIONS[self.ptt_idx][0])
        self.ptt_combo.pack(side="left", padx=4)
        self.ptt_combo.bind("<<ComboboxSelected>>", self._on_ptt_key)
        self.ptt_state_var = tk.StringVar(value="")
        self.ptt_state_label = tk.Label(ptt_row, textvariable=self.ptt_state_var, fg="#888")
        self.ptt_state_label.pack(side="left", padx=12)
        self._update_ptt_indicator(False)

        gain_frame = ttk.LabelFrame(self.root, text="Gain")
        gain_frame.pack(fill="x", **pad)
        self._gain_row(gain_frame, "Microphone Gain", "mic")
        self._gain_row(gain_frame, "Soundboard Gain", "sb")

        slot_frame = ttk.LabelFrame(self.root, text="G-Keys  (G1-G6  →  F13-F18)")
        slot_frame.pack(fill="x", **pad)
        for slot in SLOTS:
            key_name = HOTKEYS[slot].name.upper()  # 'f13' -> 'F13'
            row = ttk.Frame(slot_frame)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=f"{slot}  ({key_name})", width=10, anchor="w").pack(
                side="left"
            )
            var = tk.StringVar()
            combo = ttk.Combobox(row, textvariable=var, width=40, state="readonly")
            combo.pack(side="left", padx=(8, 8))
            combo.bind("<<ComboboxSelected>>", lambda e, s=slot: self._on_select(s))
            ttk.Button(row, text="Test", width=6, command=lambda s=slot: self._test(s)).pack(
                side="left"
            )
            self.assign_vars[slot] = var
            self.combos[slot] = combo

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.engine_btn = ttk.Button(btn_frame, text="Start Engine", command=self._toggle_engine)
        self.engine_btn.pack(side="left")
        ttk.Button(btn_frame, text="Refresh Files", command=self.refresh_files).pack(
            side="left", padx=8
        )
        ttk.Button(btn_frame, text="Refresh Devices", command=self._populate_devices).pack(
            side="left"
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken").pack(
            fill="x", side="bottom"
        )

    def _device_row(self, parent, label: str, key: str):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(frame, text=label, width=28, anchor="w").pack(side="left")
        var = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=var, width=46, state="readonly")
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e, k=key: self._apply_device(k))
        self.dev_vars[key] = var
        self.dev_combos[key] = combo

    def _gain_row(self, parent, label: str, key: str):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(frame, text=label, width=28, anchor="w").pack(side="left")
        scale = ttk.Scale(frame, from_=0.0, to=2.0, orient="horizontal", length=220)
        scale.set(1.0)
        scale.pack(side="left", padx=8)
        val_var = tk.StringVar(value="100%")
        ttk.Label(frame, textvariable=val_var, width=6).pack(side="left")
        scale.configure(
            command=lambda v, k=key, vv=val_var: self._on_gain(k, float(v), vv)
        )
        self.gain_vars[key] = (scale, val_var)

    def _populate_devices(self):
        try:
            inputs = list_devices("input")
            outputs = list_devices("output")
        except Exception as exc:
            self._post_status(f"Cannot query audio devices: {exc}")
            return
        try:
            d_in, d_out = sd.default.device
        except Exception:
            d_in = d_out = None

        # Filter to the devices this app cares about (labels already carry ch/rate).
        mic_cands = [
            (it, i) for it, i in inputs
            if "ssl" in it.lower() or "input" in it.lower() or "microphone" in it.lower()
        ]
        hp_cands = outputs  # any output device: SSL 2+, headset, monitor, ...
        cable_cands = [(it, i) for it, i in outputs if "cable" in it.lower()]

        # Enrich cable entries: mark the recommended one (has a matching
        # capture side) so the user knows which virtual cable to pick.
        cable_labeled = []
        for it, i in cable_cands:
            cable_labeled.append((it, i, find_cable_capture(i) is not None))
        rec_idx = None
        for it, i, has_cap in cable_labeled:
            if has_cap and "cable input" in it.lower():
                rec_idx = i
                break
        if rec_idx is None:
            for _it, i, has_cap in cable_labeled:
                if has_cap:
                    rec_idx = i
                    break
        cable_items = [
            (f"{it}  ★ recommended" if i == rec_idx else it, i)
            for it, i, _cap in cable_labeled
        ]

        def pick(cands, default, prefer=None):
            if prefer is not None:
                for it, i in cands:
                    if prefer in it.lower():
                        return i
            if default is not None and any(i == default for _it, i in cands):
                return default
            return cands[0][1] if cands else None

        defaults = {
            "mic": pick(mic_cands, d_in, prefer="ssl"),
            "headphone": pick(hp_cands, d_out, prefer="headset"),
            "cable": rec_idx if rec_idx is not None else pick(cable_items, None),
        }
        for key, items in (("mic", mic_cands), ("headphone", hp_cands), ("cable", cable_items)):
            combo = self.dev_combos[key]
            combo["values"] = [it for it, _ in items]
            self.dev_indices[key] = {it: i for it, i in items}
            combo.set(next((it for it, i in items if i == defaults[key]), ""))
            self._apply_device(key)
        self._update_cable_hint()
        self._post_status(
            f"Devices (WASAPI): {len(mic_cands)} mic(s), {len(hp_cands)} monitor(s), "
            f"{len(cable_items)} cable(s)"
        )

    def _update_cable_hint(self):
        """Tell the user which device to select as Discord's input."""
        item = self.dev_vars["cable"].get()
        idx = self.dev_indices.get("cable", {}).get(item)
        cap = find_cable_capture(idx)
        if cap is not None:
            d = sd.query_devices(cap)
            self.cable_hint_var.set(
                f"→ In Discord set Voice Input to: “{d['name']}” "
                f"({int(d['default_samplerate']) // 1000} kHz)"
            )
        else:
            self.cable_hint_var.set(
                "→ No matching capture device found — check the VB-Audio cable install"
            )

    def _target_rate(self) -> int:
        """Sample rate sounds should be resampled to (cable device's native rate).

        Reads the engine's cable device (plain attribute) rather than a Tk
        variable, so it is safe to call from background decode threads.
        """
        return native_rate(self.engine.cable_device, "output") or ENGINE_RATE

    def _apply_device(self, key: str):
        item = self.dev_vars[key].get()
        idx = self.dev_indices.get(key, {}).get(item)
        if key == "mic":
            self.engine.mic_device = idx
        elif key == "headphone":
            self.engine.headphone_device = idx
        else:
            self.engine.cable_device = idx
        if idx is None:
            return
        if key == "cable":
            self._update_cable_hint()
            new_rate = self._target_rate()
            if self.engine.running:
                if new_rate != self.engine.rate:
                    self._post_status(
                        f"Cable rate {new_rate} Hz — re-decoding sounds and restarting engine…"
                    )
                    self.engine.stop()
                    self.engine_btn.configure(text="Start Engine")
                    threading.Thread(
                        target=self._redecode_then_start, args=(new_rate,), daemon=True
                    ).start()
                    return
                self.engine.restart_output("cable")
            elif new_rate != self.engine.decode_rate:
                threading.Thread(target=self._redecode_all, args=(new_rate,), daemon=True).start()
            return
        if self.engine.running:
            if key == "mic":
                self.engine.restart_mic()
            else:
                self.engine.restart_output("headphone")

    def _redecode_all(self, rate: int):
        """Re-decode all assigned sounds at a new sample rate (background)."""
        for slot in SLOTS:
            with self._lock:
                path = self.assignments.get(slot, "")
            if path and os.path.isfile(path):
                try:
                    self.engine.load_sound(slot, path, rate)
                except Exception as exc:
                    self._status_q_put(f"{slot}: re-decode failed: {exc}")
        self._status_q_put(f"Sounds decoded at {rate} Hz")

    def _redecode_then_start(self, rate: int):
        self._redecode_all(rate)
        self._ui_q.put(
            lambda: (self.engine.start(), self.engine_btn.configure(text="Stop Engine"))
        )

    def _on_gain(self, key: str, value: float, val_var: tk.StringVar):
        val_var.set(f"{value * 100:.0f}%")
        if key == "mic":
            self.engine.mic_gain = value
        else:
            self.engine.sb_gain = value

    def _start_engine(self):
        """Start the engine with the currently selected devices (idempotent)."""
        if self.engine.running:
            return
        for key in ("mic", "headphone", "cable"):
            self._apply_device(key)
        self.engine.start()
        self.engine_btn.configure(text="Stop Engine")

    def _toggle_engine(self):
        if self.engine.running:
            self.engine.stop()
            self.engine_btn.configure(text="Start Engine")
        else:
            self._start_engine()

    # ------------------------------------------------------------------ files

    def refresh_files(self):
        if not os.path.isdir(self.sounds_dir):
            os.makedirs(self.sounds_dir, exist_ok=True)
        found = sorted(
            os.path.join(self.sounds_dir, f)
            for f in os.listdir(self.sounds_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS
            and os.path.isfile(os.path.join(self.sounds_dir, f))
        )
        self.files = found
        restored: list[str] = []
        saved_changed = False
        for slot in SLOTS:
            combo = self.combos[slot]
            saved_name = self.saved_names.get(slot, "")
            match = next((p for p in found if os.path.basename(p) == saved_name), "")
            combo["values"] = [os.path.basename(p) for p in found]
            if match:
                combo.set(os.path.basename(match))
                with self._lock:
                    self.assignments[slot] = match
                restored.append(slot)
            else:
                combo.set("")
                with self._lock:
                    self.assignments[slot] = ""
                if saved_name:
                    self.saved_names[slot] = ""
                    saved_changed = True
        if saved_changed:
            save_assignments(self.saved_names)
        self._post_status(f"Found {len(found)} sound file(s) in sounds/")
        if restored:
            threading.Thread(
                target=self._decode_restored, args=(restored,), daemon=True
            ).start()
        if found:
            threading.Thread(
                target=self._validate_files, args=(list(found),), daemon=True
            ).start()

    def _decode_restored(self, slots):
        """Decode sounds restored from the saved playlist (skip already-loaded ones)."""
        rate = self._target_rate()
        n = 0
        for slot in slots:
            with self._lock:
                path = self.assignments.get(slot, "")
            if not path or not os.path.isfile(path):
                continue
            with self.engine.lock:
                loaded = self.engine.slot_arrays.get(slot) is not None
            if loaded:
                continue
            try:
                self.engine.load_sound(slot, path, rate)
                n += 1
            except Exception as exc:
                self._status_q_put(f"{slot}: restore failed: {exc}")
        if n:
            self._status_q_put(f"Restored {n} saved sound(s)")

    def _validate_files(self, paths):
        """Background check: drop files the mixer cannot decode."""
        invalid = []
        for path in paths:
            try:
                pygame.mixer.Sound(path)
            except Exception:
                invalid.append(path)
        if invalid:
            self._status_q_put(
                "Removed invalid file(s): " + ", ".join(os.path.basename(p) for p in invalid)
            )
            self._ui_q.put(lambda: self._drop_invalid(invalid))

    def _drop_invalid(self, invalid):
        bad = set(invalid)
        self.files = [p for p in self.files if p not in bad]
        saved_changed = False
        for slot in SLOTS:
            combo = self.combos[slot]
            sel = self.assign_vars[slot].get()
            combo["values"] = [os.path.basename(p) for p in self.files]
            if sel and not any(os.path.basename(p) == sel for p in self.files):
                combo.set("")
                with self._lock:
                    self.assignments[slot] = ""
                self.saved_names[slot] = ""
                saved_changed = True
        if saved_changed:
            save_assignments(self.saved_names)

    # ------------------------------------------------------------------- slots

    def _on_select(self, slot):
        name = self.assign_vars[slot].get()
        path = next((p for p in self.files if os.path.basename(p) == name), "")
        with self._lock:
            self.assignments[slot] = path
        self.saved_names[slot] = os.path.basename(path) if path else ""
        save_assignments(self.saved_names)
        if not path:
            with self.engine.lock:
                self.engine.slot_arrays[slot] = None
            return
        self._post_status(f"{slot} assigned: {name} — decoding…")
        threading.Thread(target=self._decode_slot, args=(slot, path), daemon=True).start()

    def _decode_slot(self, slot, path):
        try:
            self.engine.load_sound(slot, path, self._target_rate())
            self._status_q_put(f"{slot} ready: {os.path.basename(path)}")
        except Exception as exc:
            clear = False
            with self._lock:
                if self.assignments.get(slot) == path:
                    self.assignments[slot] = ""
                    self.saved_names[slot] = ""
                    clear = True
            if clear:
                save_assignments(self.saved_names)
            self._status_q_put(f"{slot}: cannot decode {os.path.basename(path)}: {exc}")
            self._ui_q.put(lambda: self.combos[slot].set(""))

    def _test(self, slot):
        if not self.engine.running:
            self._post_status("Start the engine first")
            return
        with self._lock:
            path = self.assignments.get(slot, "")
        if not path:
            self._post_status(f"{slot}: no sound assigned")
            return
        if not os.path.isfile(path):
            self._post_status(f"{slot}: file missing — click Refresh Files")
            return
        if self.engine.trigger(slot):
            self._post_status(f"{slot} playing: {os.path.basename(path)}")
        else:
            self._post_status(f"{slot}: sound not decoded yet — re-select it")

    # ----------------------------------------------------------------- hotkeys

    def _start_listener(self):
        def on_press(key):
            slot = KEY_TO_SLOT.get(key)
            if slot is not None:
                self.engine.trigger(slot)
            if self._ptt_match(key):
                self._set_ptt(True)

        def on_release(key):
            if self._ptt_match(key):
                self._set_ptt(False)

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:
            self._post_status(f"Hotkey listener failed: {exc}")

    # ------------------------------------------------------------- push-to-talk

    def _set_ptt(self, active: bool):
        """Called from the listener thread when the PTT key is pressed/released."""
        if self.engine.ptt_active == active:
            return
        self.engine.ptt_active = active
        self._ui_q.put(lambda a=active: self._update_ptt_indicator(a))

    def _update_ptt_indicator(self, active: bool):
        if not self.ptt_var.get():
            self.ptt_state_var.set("LIVE (always on)")
            self.ptt_state_label.configure(fg="#0a0")
            return
        if active:
            self.ptt_state_var.set("LIVE")
            self.ptt_state_label.configure(fg="#0a0")
        else:
            self.ptt_state_var.set("muted (hold key)")
            self.ptt_state_label.configure(fg="#888")

    def _on_ptt_toggle(self):
        self.engine.ptt_enabled = bool(self.ptt_var.get())
        save_settings(self._settings_dict())
        self._update_ptt_indicator(self.engine.ptt_active)

    def _on_ptt_key(self, _event=None):
        label = self.ptt_combo.get()
        self.ptt_idx = next(
            (i for i, (lab, _k) in enumerate(PTT_KEY_OPTIONS) if lab == label), 0
        )
        self._ptt_match = PTT_KEY_OPTIONS[self.ptt_idx][1]
        self.engine.ptt_active = False  # the old key may still be physically held
        save_settings(self._settings_dict())
        self._post_status(f"Push-to-talk key: {label}")
        self._update_ptt_indicator(False)

    def _settings_dict(self):
        return {
            "ptt_enabled": bool(self.ptt_var.get()),
            "ptt_key": PTT_KEY_OPTIONS[self.ptt_idx][0],
        }

    # ------------------------------------------------------------------ status

    def _status_q_put(self, msg):
        self._ui_q.put(lambda: self.status_var.set(msg))

    def _post_status(self, msg):
        self._status_q_put(msg)

    def _poll_ui(self):
        """Drain thread-safe UI updates on the tkinter thread."""
        try:
            while True:
                fn = self._ui_q.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ui)

    # -------------------------------------------------------------------- exit

    def on_close(self):
        if self._listener is not None:
            self._listener.stop()
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
        self.engine.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SoundboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()