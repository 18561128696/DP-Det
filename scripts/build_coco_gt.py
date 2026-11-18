"""Build COCO bbox GT from prepared YOLO labels (not official VisDrone ignored-region evaluation)."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new file.')
    if not a.images.is_dir() or not a.labels.is_dir():
        p.error('Images and labels directories must exist.')
    from PIL import Image
    import yaml
    names = yaml.safe_load(a.data.read_text(encoding='utf-8-sig'))['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    result = {'info': {'description': 'COCO bbox GT converted from YOLO labels'}, 'licenses': [],
              'images': [], 'annotations': [],
              'categories': [{'id': k + 1, 'name': v} for k, v in sorted(names.items())]}
    files = sorted(x for x in a.images.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not files:
        p.error('No images found.')
    if len({x.stem for x in files}) != len(files):
        p.error('Image stems must be unique.')
    for image_id, image in enumerate(files):
        with Image.open(image) as im:
            w, h = im.size
        result['images'].append({'id': image_id, 'file_name': image.name, 'width': w, 'height': h})
        label = a.labels / (image.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing label: {label}; use an empty label file for a negative image.')
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'Expected five detection-label fields in {label}: {line}')
            cls = int(fields[0])
            xc, yc, bw, bh = map(float, fields[1:])
            if cls not in names or bw <= 0 or bh <= 0 or not all(0 <= z <= 1 for z in (xc, yc, bw, bh)):
                raise ValueError(f'Invalid label: {label}: {line}')
            box = [(xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h]
            result['annotations'].append({'id': len(result['annotations']) + 1, 'image_id': image_id,
                'category_id': cls + 1, 'bbox': box, 'area': box[2] * box[3], 'iscrowd': 0})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result), encoding='utf-8')
    print(f"Saved {len(files)} images and {len(result['annotations'])} annotations to {a.output}")

if __name__ == '__main__':
    main()
