#!/usr/bin/env python3
"""Utility to call the v3 embeddings API and persist face crops + embeddings."""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
from PIL import Image


def encode_image(image_path: Path) -> str:
    """Read an image from disk and return a base64-encoded string."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 payload (optionally prefixed with data URI) into a PIL image."""
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_face_assets(
    *,
    image_id: str,
    faces: Iterable[dict[str, object]],
    original_image: Image.Image,
    output_dir: Path,
    use_aligned: bool,
) -> None:
    """Persist cropped face images and embeddings to disk."""
    for idx, face in enumerate(faces, start=1):
        face_dir = output_dir / f"{image_id}_face{idx:02d}"
        ensure_output_dir(face_dir)

        embedding = np.asarray(face["embedding"], dtype=np.float32)
        np.save(face_dir / "embedding.npy", embedding)

        crop_path = face_dir / ("aligned.png" if use_aligned else "crop.png")
        image_to_save: Image.Image

        if use_aligned and face.get("aligned_face"):
            image_to_save = decode_base64_image(face["aligned_face"])  # type: ignore[arg-type]
        else:
            xmin, ymin, xmax, ymax = (int(round(v)) for v in face["bbox"])  # type: ignore[index]
            image_to_save = original_image.crop((xmin, ymin, xmax, ymax))

        image_to_save.save(crop_path)


def collect_images(input_path: Path, patterns: list[str], recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise SystemExit(f"Input {input_path} is neither a file nor a directory")

    images: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matcher = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
        for candidate in matcher:
            if candidate.is_file() and candidate not in seen:
                images.append(candidate)
                seen.add(candidate)

    images.sort()
    return images


def process_image(
    image_path: Path,
    *,
    api_url: str,
    output_root: Path,
    return_aligned: bool,
    skip_normalization: bool,
    session: requests.Session,
) -> dict[str, object] | None:
    payload = {
        "images": [
            {
                "id": image_path.stem,
                "data": encode_image(image_path),
            }
        ],
        "return_aligned_faces": return_aligned,
        "skip_embedding_normalization": skip_normalization,
    }

    response = session.post(api_url, json=payload, timeout=90)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(
            f"[ERROR] {image_path}: {response.status_code} {response.text[:200]}"
        )
        return None

    data = response.json()
    results = data.get("results") or []
    if not results:
        print(f"[WARN] {image_path}: no faces detected")
        return {
            "image": str(image_path),
            "output_dir": None,
            "num_faces": 0,
        }

    image_output_dir = output_root / image_path.stem
    ensure_output_dir(image_output_dir)
    with Image.open(image_path) as pil_image:
        original_image = pil_image.convert("RGB")

    total_faces = 0
    for result in results:
        save_face_assets(
            image_id=result["image_id"],
            faces=result["faces"],
            original_image=original_image,
            output_dir=image_output_dir,
            use_aligned=return_aligned,
        )
        total_faces += int(result.get("num_faces", 0))

    summary = {
        "image": str(image_path),
        "output_dir": str(image_output_dir),
        "faces": [
            {
                "image_id": result["image_id"],
                "num_faces": result["num_faces"],
            }
            for result in results
        ],
        "total_faces": total_faces,
        "normalized": not skip_normalization,
        "aligned": return_aligned,
    }

    summary_path = image_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] {image_path}: exported {total_faces} faces -> {image_output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory of images")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/embeddings",
        help="Embeddings endpoint URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("face_exports"),
        help="Directory where crops and embeddings will be stored",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Request raw (non-normalized) embeddings from the API",
    )
    parser.add_argument(
        "--no-aligned",
        action="store_true",
        help="Do not request aligned face crops; use raw bounding boxes instead.",
    )
    parser.add_argument(
        "--glob",
        default="*.jpg,*.jpeg,*.png,*.bmp",
        help="Comma-separated glob when input is a directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when input is a directory",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input path {args.input} does not exist")

    patterns = [pattern.strip() for pattern in args.glob.split(",") if pattern.strip()]
    images = collect_images(args.input, patterns, args.recursive)
    if not images:
        raise SystemExit("No images found to process")

    ensure_output_dir(args.output)
    return_aligned = not args.no_aligned

    session = requests.Session()
    summaries: list[dict[str, object]] = []
    for image_path in images:
        summary = process_image(
            image_path,
            api_url=args.api_url,
            output_root=args.output,
            return_aligned=return_aligned,
            skip_normalization=args.skip_normalization,
            session=session,
        )
        if summary is not None:
            summaries.append(summary)

    total_faces = sum(int(item.get("total_faces", 0)) for item in summaries)
    overall = {
        "input": str(args.input),
        "count": len(images),
        "processed": len(summaries),
        "total_faces": total_faces,
        "aligned": return_aligned,
        "normalized": not args.skip_normalization,
        "images": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(
        f"Processed {len(summaries)}/{len(images)} images, exported {total_faces} faces -> {args.output}"
    )


if __name__ == "__main__":
    main()
