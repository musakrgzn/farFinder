"""Build-time only download of a versioned, checksum-verified public model."""

import hashlib
from pathlib import Path
import urllib.request

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
SEGMENTATION_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
SEGMENTATION_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"


def main() -> None:
    download(URL, SHA256, Path("/app/models/face_landmarker.task"))
    download(SEGMENTATION_URL, SEGMENTATION_SHA256, Path("/app/models/selfie_multiclass.tflite"))


def download(url, checksum, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != checksum:
        raise RuntimeError(f"Model checksum mismatch: {actual}")
    destination.write_bytes(content)
    print(f"Downloaded verified {destination.name} ({len(content)} bytes).")


if __name__ == "__main__":
    main()
