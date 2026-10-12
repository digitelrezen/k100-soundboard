"""Quick device inventory: index, channels, native rate, full name."""
import sounddevice as sd

print("HOST APIS:")
for i, h in enumerate(sd.query_hostapis()):
    print(f"  {i}: {h['name']}")
print()
for i, d in enumerate(sd.query_devices()):
    print(
        f"{i:2d}  in={int(d['max_input_channels']):2d}  "
        f"out={int(d['max_output_channels']):2d}  "
        f"rate={int(d['default_samplerate']):5d}  "
        f"host={d['hostapi']}  |  {d['name']}"
    )