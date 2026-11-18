import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image


SEED = 2026
TRAIN_RATIO = 0.8
RAW_ROOT = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / 'datasets/NWPU_VHR10_YOLO'
DATA_YAML = PROJECT_ROOT / 'configs/nwpu.yaml'

CLASS_NAMES = [
    "airplane",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "harbor",
    "bridge",
    "vehicle",
]


def parse_gt_line(line):
    pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)"
    m = re.match(pattern, line.strip())
    if not m:
        raise ValueError(f"Invalid NWPU label line: {line!r}")
    x1, y1, x2, y2, cls = map(int, m.groups())
    return x1, y1, x2, y2, cls - 1


def yolo_line(box, width, height):
    x1, y1, x2, y2, cls = box
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def reset_output_dirs():
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = OUT_ROOT / sub / split
            if d.exists() and any(d.iterdir()):
                raise FileExistsError(f'Output is not empty: {d}. Choose a new --output directory.')
            d.mkdir(parents=True, exist_ok=True)


def collect_items():
    pos_dir = RAW_ROOT / "positive image set"
    neg_dir = RAW_ROOT / "negative image set"
    gt_dir = RAW_ROOT / "ground truth"

    items = []
    for image_path in sorted(pos_dir.glob("*.jpg")):
        gt_path = gt_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file for {image_path.name}: {gt_path}")
        items.append(("pos", image_path, gt_path))

    for image_path in sorted(neg_dir.glob("*.jpg")):
        items.append(("neg", image_path, None))

    random.Random(SEED).shuffle(items)
    return items


def convert_item(item, split):
    kind, image_path, gt_path = item
    out_name = f"{kind}_{image_path.name}"
    out_image = OUT_ROOT / "images" / split / out_name
    out_label = OUT_ROOT / "labels" / split / f"{Path(out_name).stem}.txt"

    shutil.copy2(image_path, out_image)

    if gt_path is None:
        out_label.write_text("", encoding="utf-8")
        return 0

    with Image.open(image_path) as im:
        width, height = im.size

    lines = []
    for raw in gt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.strip():
            converted = yolo_line(parse_gt_line(raw), width, height)
            if converted is not None:
                lines.append(converted)

    out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_yaml():
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(
        f"path: {OUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def main():
    import argparse
    global RAW_ROOT, OUT_ROOT, DATA_YAML
    parser = argparse.ArgumentParser(description='Prepare NWPU VHR-10 with seed 2026 and train ratio 0.8.')
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT_ROOT)
    parser.add_argument('--data-yaml', type=Path, default=DATA_YAML)
    args = parser.parse_args()
    RAW_ROOT, OUT_ROOT, DATA_YAML = args.raw_root.resolve(), args.output.resolve(), args.data_yaml.resolve()
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"NWPU raw dataset not found: {RAW_ROOT}")

    reset_output_dirs()
    items = collect_items()
    split_idx = int(len(items) * TRAIN_RATIO)
    train_items = items[:split_idx]
    val_items = items[split_idx:]

    train_boxes = sum(convert_item(item, "train") for item in train_items)
    val_boxes = sum(convert_item(item, "val") for item in val_items)
    write_yaml()

    print("=" * 80)
    print("NWPU VHR-10 converted to YOLO format")
    print(f"RAW_ROOT={RAW_ROOT}")
    print(f"OUT_ROOT={OUT_ROOT}")
    print(f"DATA_YAML={DATA_YAML}")
    print(f"SEED={SEED}")
    print(f"TRAIN_IMAGES={len(train_items)}")
    print(f"VAL_IMAGES={len(val_items)}")
    print(f"TRAIN_BOXES={train_boxes}")
    print(f"VAL_BOXES={val_boxes}")
    print("=" * 80)


if __name__ == "__main__":
    main()
