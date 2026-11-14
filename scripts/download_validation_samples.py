"""Download real bus/highway sequences from the public Xiph test-media mirror.

Run: python scripts/download_validation_samples.py
Default outputs stay under scripts/validation_samples/; --output-dir overrides.
Original Y4M bytes are retained, verified, and converted to MP4 without temporal
subsampling. The provenance JSON records source and derived checksums, decoder
version, dimensions, frame count, FPS and retrieval time. No annotations or
accuracy scores are provided. MP4 bytes can vary by OpenCV/codec version.

Publisher provenance and rights statement:
https://media.xiph.org/video/derf/
The publisher states these sequences were publicly available or explicitly
provided for hosting and are believed freely redistributable. Bus/highway have
no individual license link on this page; no CC or public-domain status is
asserted here. Preserve this statement and consult the publisher for uses
requiring a more specific rights grant.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


SOURCE_PAGE = "https://media.xiph.org/video/derf/"
SAMPLES = {
    "bus": {"url": "https://media.xiph.org/video/derf/y4m/bus_cif.y4m",
            "bytes": 22810538, "frames": 150,
            "original_source": "ftp://ftp.tnt.uni-hannover.de/pub/svc/testsequences/",
            "sha256": "08f557aade483465fca3ad9f5ce9a5b6aad7e833c871ec969b5b8fbb0a791c87",
            "purpose": "Real moving bus footage; independently annotate before scoring."},
    "highway": {"url": "https://media.xiph.org/video/derf/y4m/highway_qcif.y4m",
                "bytes": 76044044, "frames": 2000,
                "original_source": "http://trace.eas.asu.edu/yuv/",
                "sha256": "19929592227c8062c39fcf6025d01f8f3cd20134b91a71b23f2c6dd34ae29d52",
                "purpose": "Real highway traffic; low resolution is an explicit stress case."},
}
RIGHTS = {
    "publisher": "Xiph.Org test-media mirror",
    "evidence_url": SOURCE_PAGE,
    "statement": "The publisher says these clips were previously public or supplied with "
                 "permission to host, and believes they may be freely redistributed.",
    "specific_license": None,
    "qualification": "No individual license link is listed for bus or highway. The publisher "
                     "notes some sequences may have additional restrictions. No CC or "
                     "public-domain designation is inferred.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, expected_bytes: int | None = None,
             expected_sha256: str | None = None, max_bytes: int = 100_000_000,
             opener=urlopen) -> dict:
    """Stream into a temporary file; publish only after size and digest checks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_bytes is not None and expected_sha256 is not None:
        if destination.stat().st_size == expected_bytes and sha256(destination) == expected_sha256:
            return {"url": url, "resolved_url": url, "bytes": expected_bytes,
                    "sha256": expected_sha256, "reused_verified_file": True}
    partial = None
    try:
        request = Request(url, headers={"User-Agent": "cv-validation-sample-downloader/1.0"})
        with opener(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f"{url}: expected HTTP 200, got {response.status}")
            digest, size = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as target:
                partial = Path(target.name)
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"{url}: exceeds {max_bytes} byte download limit")
                    target.write(chunk)
                    digest.update(chunk)
            checksum = digest.hexdigest()
            if expected_bytes is not None and size != expected_bytes:
                raise ValueError(f"{url}: expected {expected_bytes} bytes, got {size}")
            if expected_sha256 is not None and checksum != expected_sha256:
                raise ValueError(f"{url}: SHA-256 mismatch; existing file preserved")
            partial.replace(destination)
            return {"url": url, "resolved_url": response.geturl(), "bytes": size,
                    "sha256": checksum, "reused_verified_file": False}
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def convert_y4m(source: Path, destination: Path, expected_frames: int) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(source))
    writer = None
    partial = destination.with_name(destination.stem + ".partial.mp4")
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not 0 < fps < 1000:
            raise ValueError(f"Invalid source FPS: {fps}")
        count = 0
        width = height = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"Cannot encode {destination}")
            if frame.shape[:2] != (height, width):
                raise ValueError("Source frame dimensions changed")
            writer.write(frame)
            count += 1
        if count != expected_frames:
            raise ValueError(f"Expected {expected_frames} frames, decoded {count}")
        writer.release()
        writer = None
        verify = cv2.VideoCapture(str(partial))
        try:
            decoded = 0
            while True:
                ok, frame = verify.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("Converted frame dimensions changed")
                decoded += 1
            if decoded != count:
                raise ValueError(f"Converted MP4 decoded {decoded} of {count} frames")
        finally:
            verify.release()
        partial.replace(destination)
        return {"path": str(destination.resolve()), "sha256": sha256(destination),
                "bytes": destination.stat().st_size, "frames": count, "fps": fps,
                "width": width, "height": height, "codec": "mp4v",
                "opencv_version": cv2.__version__, "temporal_subsampling": False,
                "conversion": "OpenCV Y4M decode to BGR, then lossy MP4 encoding"}
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "validation_samples")
    parser.add_argument("--samples", nargs="+", choices=sorted(SAMPLES), default=sorted(SAMPLES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        evidence = args.output_dir / "xiph_source_page.html"
        page = download(SOURCE_PAGE, evidence, max_bytes=2_000_000)
        for name in dict.fromkeys(args.samples):
            spec = SAMPLES[name]
            source = args.output_dir / Path(spec["url"]).name
            print(f"Downloading and verifying {name}...", flush=True)
            original = download(spec["url"], source, expected_bytes=spec["bytes"],
                                expected_sha256=spec["sha256"])
            with source.open("rb") as stream:
                if not stream.readline(4096).startswith(b"YUV4MPEG2 "):
                    raise ValueError(f"{source}: not a Y4M video")
            converted = convert_y4m(source, args.output_dir / f"{name}.mp4", spec["frames"])
            provenance = {"sample": name, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                          "original": {**original, "path": str(source.resolve()),
                                       "upstream_source_listed_by_publisher": spec["original_source"]},
                          "converted": converted, "rights": RIGHTS,
                          "source_page_snapshot": {**page, "path": str(evidence.resolve())},
                          "intended_use": spec["purpose"], "annotations": None,
                          "limitations": ["Compression-research sample, not an annotated tracking benchmark.",
                                          "Source FPS may be inferred, as noted by the publisher.",
                                          "No camera calibration or 3D object ground truth supplied."]}
            path = args.output_dir / f"{name}.provenance.json"
            path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
            print(f"Obtained {converted['frames']} frames: {converted['path']}", flush=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
