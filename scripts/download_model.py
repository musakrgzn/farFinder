"""Build-time only download of a versioned, checksum-verified public model."""

import hashlib
from pathlib import Path
import urllib.request

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main() -> None:
    destination = Path("/app/models/face_landmarker.task")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"Model checksum mismatch: {actual}")
    destination.write_bytes(content)
    print(f"Downloaded verified MediaPipe Face Landmarker v1 ({len(content)} bytes).")


if __name__ == "__main__":
    main()
