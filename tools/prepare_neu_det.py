from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from urllib.error import HTTPError
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
BASE_URL = "https://raw.githubusercontent.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/master"


def _download(urls: tuple[str, ...], target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
            return
        except HTTPError as exc:
            if exc.code != 404:
                raise
    raise FileNotFoundError(f"source file not found at any URL: {urls}")


def _parse_xml(path: Path, class_id: int) -> tuple[list[list[float]], list[int]]:
    root = ET.parse(path).getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name == "defect":
            name = CLASSES[class_id]
        if name not in CLASSES:
            raise ValueError(f"annotation class mismatch in {path}: {name}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing bndbox in {path}")
        values = [float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if not all(value == value for value in values) or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"invalid box in {path}: {values}")
        boxes.append(values)
        labels.append(CLASSES.index(name))
    if not boxes:
        raise ValueError(f"annotation contains no objects: {path}")
    return boxes, labels


def prepare(output_dir: str | Path, *, seed: int = 42) -> dict[str, object]:
    output = Path(output_dir).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"prepared dataset already exists: {output}")
    image_dir, annotation_dir = output / "images", output / "annotations"
    records: list[dict[str, object]] = []
    seen_hashes: dict[str, Path] = {}
    jobs = []
    for class_name in CLASSES:
        for index in range(1, 301):
            stem = f"{class_name}_{index}"
            jobs.extend((((f"{BASE_URL}/IMAGES/{stem}.jpg", f"{BASE_URL}/Validation_Images/{stem}.jpg"), image_dir / f"{stem}.jpg"), ((f"{BASE_URL}/ANNOTATIONS/{stem}.xml", f"{BASE_URL}/Validation_Annotations/{stem}.xml"), annotation_dir / f"{stem}.xml")))
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda job: _download(*job), jobs))
    for class_id, class_name in enumerate(CLASSES):
        class_records = []
        for index in range(1, 301):
            stem = f"{class_name}_{index}"
            image = image_dir / f"{stem}.jpg"
            xml = annotation_dir / f"{stem}.xml"
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"duplicate image hash detected: {seen_hashes[digest]} and {image}; resolve source anomaly before splitting")
            seen_hashes[digest] = image
            with Image.open(image) as loaded:
                width, height = loaded.size
            boxes, labels = _parse_xml(xml, class_id)
            row = {"image": str(image.relative_to(output)).replace("\\", "/"), "boxes": boxes, "labels": labels, "metadata": {"source": BASE_URL, "source_image": f"IMAGES/{stem}.jpg", "source_annotation": f"ANNOTATIONS/{stem}.xml", "source_index": index, "source_class": class_name}}
            row["metadata"]["width"], row["metadata"]["height"] = width, height
            class_records.append(row)
        random.Random(seed + class_id).shuffle(class_records)
        for split, subset in (("train", class_records[:240]), ("val", class_records[240:270]), ("test", class_records[270:])):
            for row in subset:
                records.append({**row, "metadata": {**row["metadata"], "split": split}})
    output.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in ("train", "val", "test"):
        selected = [row for row in records if row["metadata"]["split"] == split]
        split_counts[split] = len(selected)
        (output / f"{split}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
        image_list = []
        label_dir = output / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            image_path = output / str(row["image"])
            image_list.append(str(image_path))
            width, height = float(row["metadata"]["width"]), float(row["metadata"]["height"])
            label_path = label_dir / f"{image_path.stem}.txt"
            lines = []
            for box, label in zip(row["boxes"], row["labels"]):
                x1, y1, x2, y2 = [float(value) for value in box]
                lines.append(f"{int(label)} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output / f"{split}_images.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
    (output / "dataset.yaml").write_text("path: .\ntrain: train_images.txt\nval: val_images.txt\ntest: test_images.txt\nnc: 6\nnames: " + json.dumps(list(CLASSES)) + "\n", encoding="utf-8")
    manifest = {"source": BASE_URL, "seed": seed, "classes": list(CLASSES), "images": len(records), "split_counts": split_counts, "sha256_images": len(seen_hashes), "license_note": "source repository does not declare a license; keep downloaded data local and do not redistribute"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and split the complete NEU-DET dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
