(# Hardware and run record

## Where it ran
- File 0 (`train-00000`): Google Colab, 1x Tesla T4, ran in 42m12s
  (9144 s of video, about 3.6x realtime, 0 errors).
- Files <1-50>: Kaggle notebook, 2x Tesla T4 (16 GB each), about 31 GB RAM,
  Ubuntu-based image, Python 3.13. One process per GPU:
  GPU 0 files 1-24, GPU 1 files 25-49.
- Disk: <size> used for the input Parquet files (~24 GB).

## Software
ultralytics 8.4.216, opencv-python-headless 5.0.0.93, numpy 2.4.6,
pyarrow 23.0.1, torch 2.14.1, FP16 inference.

## Timing
- Measured on a T4: FP32 about 52 fps, FP16 about 102 fps.
- Total wall time for the full run: <fill in>
- Interruptions: <write honestly, e.g. Kaggle session restarts, resumed from `.done` files>
- Files completed: <N of 50>. <If not all, say which are missing and why.>

## Notes
- FP16 was checked against FP32 on 600 sampled frames (262 with detections):
  identical objects and people counts. This is a limited check.
- Error handling for truly damaged videos was tested only by code logic, not on
  real damaged files.)
