
from pathlib import Path
import pyarrow.parquet as pq


def iter_parquet_videos(parquet_path):
    """
    Read videos from a Parquet file one row at a time.

    Yields:
        row_index: Original row number in the Parquet file.
        video_bytes: Raw MP4 bytes.
        error: None or an error message.
    """

    parquet_path = Path(parquet_path)

    parquet_file = pq.ParquetFile(parquet_path)

    required_columns = {"mp4"}

    available_columns = set(parquet_file.schema_arrow.names)

    if not required_columns.issubset(available_columns):
        raise ValueError(
            f"Missing mp4 column in {parquet_path.name}"
        )

    row_index = 0

    for batch in parquet_file.iter_batches(
        columns=["mp4"],
        batch_size=1
    ):

        video_bytes = batch.column(0)[0].as_py()

        if not video_bytes:
            yield row_index, None, "Empty or missing video data"
        else:
            yield row_index, video_bytes, None

        row_index += 1


def get_parquet_files(input_dir):
    """
    Return Parquet files in sorted order.
    """

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    return sorted(input_dir.rglob("*.parquet"))
