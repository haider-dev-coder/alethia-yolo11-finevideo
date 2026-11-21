import json, gzip, glob, os


class FileWriter:
    """Writes one parquet file's rows to <out_dir>/<name>.jsonl, marks .done when complete."""

    def __init__(self, out_dir, parquet_name):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, parquet_name + ".jsonl")
        self.done = self.path + ".done"
        self.f = open(self.path + ".tmp", "w")

    def write(self, row):
        self.f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        self.f.close()
        os.replace(self.path + ".tmp", self.path)
        open(self.done, "w").close()


def is_done(out_dir, parquet_name):
    return os.path.exists(os.path.join(out_dir, parquet_name + ".jsonl.done"))


def merge(out_dir, output_path):
    opener = gzip.open if output_path.endswith(".gz") else open
    with opener(output_path, "wt") as out:
        for p in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
            with open(p) as f:
                for line in f:
                    out.write(line)