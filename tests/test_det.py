import time, torch
from src.video_decoder import VideoDecoder
from src.detector import ObjectDetector

b = open("/content/v0.mp4", "rb").read()
frames = []
for t, fr in VideoDecoder(b).decode():
    frames.append(fr)
    if len(frames) == 64: break

for half in (False, True):
    d = ObjectDetector(half=half)
    d.detect_batch(frames[:8])                      # warmup
    torch.cuda.synchronize(); t0 = time.time()
    res = []
    for i in range(0, 64, 32):
        res += d.detect_batch(frames[i:i+32])
    torch.cuda.synchronize()
    print("half", half, "fps", round(64 / (time.time() - t0), 1))
    print(res[0], res[-1])