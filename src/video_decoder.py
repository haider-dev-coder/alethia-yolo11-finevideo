
import cv2
import numpy as np
import tempfile
import os


class VideoDecoder:
    """
    Decode MP4 video bytes into timestamped BGR frames.
    """

    def __init__(self, video_bytes):
        self.video_bytes = video_bytes

    def decode(self):

        temp_path = None
        cap = None

        try:
            # Write MP4 bytes to a temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as temp_file:

                temp_file.write(self.video_bytes)
                temp_path = temp_file.name

            # Open video using OpenCV
            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                raise ValueError("Could not decode video")

            # Disable automatic rotation so we apply it ourselves
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

            fps = cap.get(cv2.CAP_PROP_FPS)

            if not fps or fps <= 0:
                raise ValueError("Invalid video FPS")

            # Read rotation metadata
            rotation = 0

            if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
                rotation = cap.get(
                    cv2.CAP_PROP_ORIENTATION_META
                )

            rotation = int(round(rotation)) % 360

            if rotation not in (0, 90, 180, 270):
                raise ValueError(
                    f"Unsupported rotation metadata: {rotation}"
                )

            first_timestamp = None
            frame_index = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                if frame is None:
                    raise ValueError("Invalid decoded frame")

                # Apply rotation before inference
                if rotation == 90:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_CLOCKWISE
                    )

                elif rotation == 180:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_180
                    )

                elif rotation == 270:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_COUNTERCLOCKWISE
                    )

                # Read frame timestamp
                timestamp_ms = cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                if timestamp_ms is not None and timestamp_ms >= 0:
                    timestamp = timestamp_ms / 1000.0
                else:
                    timestamp = frame_index / fps

                # Timestamp relative to first decoded frame
                if first_timestamp is None:
                    first_timestamp = timestamp

                relative_timestamp = max(
                    0.0,
                    timestamp - first_timestamp
                )

                # OpenCV returns BGR frames by default
                yield relative_timestamp, frame

                frame_index += 1

            if frame_index == 0:
                raise ValueError("Video contains no decodable frames")

        finally:

            if cap is not None:
                cap.release()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
