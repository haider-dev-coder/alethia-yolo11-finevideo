import glob, time
import pyarrow.parquet as pq
from src.video_decoder import VideoDecoder

f = sorted(glob.glob("/content/finevideo/data/*.parquet"))[0]
pf = pq.ParquetFile(f)
print(f, pf.metadata.num_rows, "rows,", pf.num_row_groups, "groups")
col = pf.read_row_group(0, columns=["mp4"]).column("mp4")
for i in range(3):
    b = col[i].as_py()
    open(f"/content/v{i}.mp4", "wb").write(b)
    ts = []; t0 = time.time(); fr = None
    for t, fr in VideoDecoder(b).decode():
        ts.append(t)
    dt = time.time() - t0
    print(i, "frames", len(ts), "last_ts", round(ts[-1], 3), "shape", fr.shape, fr.dtype,
          "decode_fps", round(len(ts) / dt))
    print("  first ts", [round(x, 3) for x in ts[:5]],
          "monotonic", all(b2 >= a for a, b2 in zip(ts, ts[1:])))