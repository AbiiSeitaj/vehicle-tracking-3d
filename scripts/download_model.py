"""Download the exact detector artifact and record its checksum."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def main():
    path = ROOT / "models" / "yolov8n.pt"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = path.with_suffix(".manifest.json")
    if manifest.exists():
        expected = json.loads(manifest.read_text())["sha256"]
        if digest != expected:
            raise RuntimeError("Model checksum differs from recorded artifact.")
    manifest.write_text(json.dumps({"url": URL, "sha256": digest, "bytes": path.stat().st_size,
        "license": "Ultralytics AGPL-3.0 / enterprise; see README.md"}, indent=2) + "\n", encoding="utf-8")
    print(f"Model ready: {path.name}; SHA256={digest}")


if __name__ == "__main__":
    main()
