(# WRITEUP

## 1. Timings
- File 0 (Colab, 1x T4, FP16): 9144 s of video in 42m12s, about 3.6x realtime, 0 errors.
- Benchmark on a T4: FP32 about 52 fps, FP16 about 102 fps. Decoding alone ran at about 1000 fps.
- Full run of 50 files at this speed: about 35 GPU-hours on one T4, so about 17-18 hours on two.
- What I actually completed: <N> of 50 files (files <list>). Total wall time: <fill in>.
  Files 0 and <list> ran on <Colab/Kaggle>.

## 2. Pipeline
- `run.py` walks the sorted Parquet files. A decode thread reads each video (ffmpeg-based
  decoder, rotation tag applied, BGR frames) and feeds a queue. The main thread batches 32
  frames to YOLO11l (imgsz 640, conf 0.4, NMS IoU 0.7, FP16, every frame).
- The detector filters the 5 target classes itself and counts persons per frame.
- The aggregator groups frames into seconds: an object is present if it appears in at least
  50% of that second's frames, and people is the per-frame median, halves rounded down.
  Empty seconds are still written.
- Output goes to `<parquet>.jsonl.tmp` and is renamed atomically when the file is complete,
  then a `.done` marker is written. Reruns skip files with `.done`, which made resuming after
  session resets possible. Per-video errors become one error line and do not stop the run.

## 3. Where the time goes
Detection on the GPU is the bottleneck (about 100 fps) while decoding is about 10x faster, so
the GPU is always the limit. FP16 roughly doubled throughput over FP32.

## 4. Correctness checks
- Decoded frame counts matched ffprobe exactly on 3 videos (4994, 18086, 6702); timestamps monotonic.
- Rotation: a synthetic rotated video matched ffmpeg autorotate with zero pixel difference.
- Aggregator unit test for the 50% rule, median rounding and gap seconds.
- FP16 vs FP32 on 600 frames (262 with detections): identical objects and people counts.
  This is a limited sample.
- Not tested: real damaged videos (only the code path).
- Final validation of the merged file: <JSON valid? videos counted? error lines?>

## 5. Where AI suggestions were wrong
- A `pip freeze` was proposed as the base for requirements.txt; it contained `+cu130` builds and
  three OpenCV packages that conflict. I cleaned it.
- The first push cell printed the full GitHub token in its output. I revoked the token.
- The code recovered from the notebook had an older `run.py` without `--start/--end`; the
  suggested commands assumed those flags existed until I checked.
- Several commands were given with `<placeholders>` that I ran literally and they failed.
- The Hugging Face "save as you go" idea needed a write token I did not have, and cost time.
- Advice to rely on an interactive Kaggle session was wrong in practice: it reset about five
  times, each time wiping `/kaggle/working`.

## 6. Next steps
- Run as a Kaggle commit or on a machine with a stable session so work is not lost.
- Write results to durable storage (Drive or a bucket) as each file finishes.
- Try batch 64 and `cudnn.benchmark`, then a TensorRT export, checking outputs stay identical.
- Test on real damaged videos.
- Finish the remaining files: <list>.)
