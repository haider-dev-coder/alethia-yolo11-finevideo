from src.video_decoder import VideoDecoder
from src.detector import ObjectDetector

frames = []
for p in ("/content/v0.mp4", "/content/v1.mp4", "/content/v2.mp4"):
    for i, (t, fr) in enumerate(VideoDecoder(open(p, "rb").read()).decode()):
        if i % 20 == 0:
            frames.append(fr)
        if len(frames) % 100 == 99: pass
    frames = frames[:len(frames)]
frames = frames[:600]
print("frames", len(frames))

def run(half):
    d = ObjectDetector(half=half); out = []
    for i in range(0, len(frames), 32):
        out += d.detect_batch(frames[i:i+32])
    return out

a, b = run(False), run(True)
nonempty = sum(1 for x in a if x["objects"])
objdiff = sum(x["objects"] != y["objects"] for x, y in zip(a, b))
pdiff = sum(x["people"] != y["people"] for x, y in zip(a, b))
print("frames with detections (fp32):", nonempty)
print("objects differ:", objdiff, "| people differ:", pdiff, "of", len(a))