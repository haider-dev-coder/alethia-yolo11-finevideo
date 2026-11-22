(# ALETHIA: YOLO11l per-second object detection on FineVideo

Runs the official `yolo11l.pt` over FineVideo Parquet files and writes, for every
second of every video, which of 5 objects are present (person, car, dog, laptop,
cell phone) and how many people are visible.

## Install

```bash
pip install ultralytics==8.4.216 opencv-python-headless==5.0.0.93 pyarrow
```

`requirements.txt` lists the pinned versions. If PyTorch with CUDA is already
installed (for example on Kaggle), do not reinstall torch from it. Install only
the three packages above. Use only the headless OpenCV package, because several
OpenCV packages together conflict.

Prepare the model (downloaded automatically on first run, or manually):

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

```bash
python run.py --input <folder with .parquet files> --output results.jsonl.gz --work <checkpoint dir>
```

Options:

- `--input`: folder searched recursively for `*.parquet`
- `--output`: final file (`.gz` extension produces gzip)
- `--work`: checkpoint folder (default `<output>.parts`)
- `--start`, `--end`: file index range over the sorted file list (end exclusive)
- `--device`: GPU index (default 0)
- `--fp32`: disable FP16 (FP16 is the default)
- `--merge-only`: only merge finished parts into `--output`

Resume: each finished Parquet file gets a `.done` marker and is skipped on rerun.
Output is written to `.tmp` and renamed atomically when the file is complete.

Two GPUs: run two processes with `CUDA_VISIBLE_DEVICES=0` and `=1` and different
`--start/--end` ranges, sharing one `--work` folder, then merge once:

```bash
python run.py --input data --work parts --output results.jsonl.gz --merge-only
```

## Output format

One JSON line per video per second:

```json
{"video":"train-00000-of-01357.parquet:0","second":0,"objects":["person"],"people":2}
```

- `video`: `<parquet filename>:<row index>` (row index from 0)
- `objects`: classes present in at least 50% of that second's frames
- `people`: median of per-frame person counts, halves rounded down
- Seconds with nothing detected are still written (`objects: []`, `people: 0`)
- A damaged video produces `{"video": "...", "error": "..."}`; if decoding fails
  midway, the lines already produced are kept followed by one error line

## Fixed settings

imgsz 640, conf 0.4, NMS IoU 0.7, every decoded frame, BGR input, rotation tag respected.

## Layout

- `run.py`: entry point (decode thread, batch of 32, resume, per-video error handling)
- `src/video_decoder.py`: decoding with rotation handling
- `src/detector.py`: YOLO11l inference, class filtering
- `src/aggregator.py`: per-second aggregation
- `src/output_writer.py`: per-file output, `.done` markers, merge
- `tests/`: the checks used during development)
