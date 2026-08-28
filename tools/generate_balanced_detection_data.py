from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageEnhance, ImageOps

from chatdemo.vision.augment import transform_square_boxes


PHOTOMETRIC_VARIANTS = (
    (1.00, 1.00),
    (0.92, 1.05),
    (1.08, 0.95),
)


def _load_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"manifest contains no rows: {manifest}")
    return rows


def _resolve_image(row: dict[str, Any], manifest: Path) -> Path:
    image = Path(row["image"])
    return image if image.is_absolute() else (manifest.parent / image).resolve()


def _pil_geometry(image: Image.Image, quarter_turns: int, mirror_horizontal: bool) -> Image.Image:
    transformed = image
    for _ in range(quarter_turns % 4):
        transformed = transformed.transpose(Image.Transpose.ROTATE_90)
    if mirror_horizontal:
        transformed = ImageOps.mirror(transformed)
    return transformed


def generate_balanced_dataset(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    samples_per_class: int = 24,
) -> Path:
    manifest = Path(source_manifest).resolve()
    output = Path(output_dir).resolve()
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_rows(manifest):
        metadata = row.get("metadata", {})
        source_group = metadata.get("source_filename_class")
        if not source_group:
            labels = {int(label) for label in row.get("labels", [])}
            if len(labels) == 1:
                source_group = f"class_{next(iter(labels))}"
            else:
                stem = Path(row["image"]).stem
                source_group = stem
        grouped[str(source_group)].append(row)

    generated_rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for group_index, group_name in enumerate(sorted(grouped)):
        sources = grouped[group_name]
        class_counts[group_name] = samples_per_class
        for output_index in range(samples_per_class):
            source = sources[output_index % len(sources)]
            variant_index = output_index // len(sources)
            geometry_index = variant_index % 8
            photo_index = (variant_index // 8) % len(PHOTOMETRIC_VARIANTS)
            quarter_turns = geometry_index % 4
            mirror_horizontal = geometry_index >= 4
            brightness, contrast = PHOTOMETRIC_VARIANTS[photo_index]

            source_image = _resolve_image(source, manifest)
            with Image.open(source_image) as loaded:
                image = loaded.convert("RGB")
            if image.width != image.height:
                raise ValueError(f"D4 balancing requires square images: {source_image}")
            boxes = torch.tensor(source["boxes"], dtype=torch.float32).reshape(-1, 4)
            transformed_boxes = transform_square_boxes(
                boxes,
                size=float(image.width),
                quarter_turns=quarter_turns,
                mirror_horizontal=mirror_horizontal,
            )
            transformed_image = _pil_geometry(image, quarter_turns, mirror_horizontal)
            transformed_image = ImageEnhance.Brightness(transformed_image).enhance(brightness)
            transformed_image = ImageEnhance.Contrast(transformed_image).enhance(contrast)

            stem = source_image.stem
            filename = f"group_{group_index:02d}_{output_index:03d}_{stem}.jpg"
            transformed_image.save(image_output / filename, quality=95)
            metadata = dict(source.get("metadata", {}))
            metadata.update(
                {
                    "generated_from": str(source_image),
                    "generation": {
                        "quarter_turns_ccw": quarter_turns,
                        "mirror_horizontal": mirror_horizontal,
                        "brightness_factor": brightness,
                        "contrast_factor": contrast,
                        "source_group": group_name,
                    },
                }
            )
            generated_rows.append(
                {
                    "image": f"images/{filename}",
                    "boxes": [[round(float(value), 4) for value in box] for box in transformed_boxes.tolist()],
                    "labels": list(source["labels"]),
                    "metadata": metadata,
                }
            )

    manifest_output = output / "manifest.jsonl"
    manifest_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in generated_rows) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "source_manifest": str(manifest),
                "samples_per_class": samples_per_class,
                "class_counts": class_counts,
                "total_samples": len(generated_rows),
                "policy": "D4 square geometry plus bounded brightness/contrast; no perspective, blur, mosaic, or pasted boxes",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a balanced, bbox-preserving detection dataset")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(generate_balanced_dataset(args.manifest, args.output_dir, samples_per_class=args.samples_per_class))


if __name__ == "__main__":
    main()
