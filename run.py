import argparse, glob, os, queue, threading, sys
import pyarrow.parquet as pq
from src.video_decoder import VideoDecoder
from src.detector import ObjectDetector
from src.aggregator import SecondAggregator
from src.output_writer import FileWriter, is_done, merge

BATCH = 32


def decode_worker(video_bytes, q):
    try:
        for ts, fr in VideoDecoder(video_bytes).decode():
            q.put((ts, fr))
        q.put(None)
    except Exception as e:
        q.put(e)


def process_video(video_bytes, vid, det, writer):
    agg = SecondAggregator(vid)
    q = queue.Queue(maxsize=BATCH * 4)
    threading.Thread(target=decode_worker, args=(video_bytes, q), daemon=True).start()
    batch_ts, batch_fr = [], []

    def flush():
        for ts, r in zip(batch_ts, det.detect_batch(batch_fr)):
            for row in agg.add_frame(ts, r["objects"], r["people"]):
                writer.write(row)
        batch_ts.clear(); batch_fr.clear()

    err = None
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            err = item
            break
        batch_ts.append(item[0]); batch_fr.append(item[1])
        if len(batch_fr) == BATCH:
            flush()
    try:
        if batch_fr:
            flush()
        for row in agg.finish():
            writer.write(row)
    except Exception as e:
        err = err or e
    if err is not None:
        writer.write({"video": vid, "error": str(err)[:200] or "decode error"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--work", default=None, help="checkpoint dir (default: <output>.parts)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--start", type=int, default=0, help="first file index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last file index (exclusive)")
    ap.add_argument("--merge-only", action="store_true", help="only merge parts into --output")
    a = ap.parse_args()
    work = a.work or a.output + ".parts"

    if a.merge_only:
        merge(work, a.output)
        print("merged ->", a.output)
        return
    dev = int(a.device) if a.device.isdigit() else a.device
    det = ObjectDetector(device=dev, half=not a.fp32)
    files = sorted(glob.glob(os.path.join(a.input, "**", "*.parquet"), recursive=True))
    files = files[a.start:a.end]
    print(f"processing {len(files)} files (index {a.start}..{a.end})", flush=True)
    for fp in files:
        name = os.path.basename(fp)
        if is_done(work, name):
            print("skip", name, flush=True); continue
        writer = FileWriter(work, name)
        try:
            pf = pq.ParquetFile(fp)
            row = 0
            for g in range(pf.num_row_groups):
                col = pf.read_row_group(g, columns=["mp4"]).column("mp4")
                for i in range(len(col)):
                    vid = f"{name}:{row}"
                    try:
                        process_video(col[i].as_py(), vid, det, writer)
                    except Exception as e:
                        writer.write({"video": vid, "error": str(e)[:200] or "failed"})
                    row += 1
                del col
        except Exception as e:
            writer.write({"video": f"{name}:*", "error": str(e)[:200]})
        writer.close()
        print("done", name, flush=True)
    print("all assigned files done", flush=True)


if __name__ == "__main__":
    main()