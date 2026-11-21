import glob, subprocess, json
import pyarrow.parquet as pq

f = sorted(glob.glob("/content/finevideo/data/*.parquet"))[0]
col = pq.ParquetFile(f).read_row_group(0, columns=["mp4"]).column("mp4")
for i in range(len(col)):
    open("/tmp/x.mp4", "wb").write(col[i].as_py())
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
        "-of", "json", "/tmp/x.mp4"], capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    rot = [d.get("rotation") for d in s.get("side_data_list", [])] or s.get("tags", {}).get("rotate")
    if rot:
        print("ROTATED row", i, s["width"], s["height"], rot)
print("scan done")