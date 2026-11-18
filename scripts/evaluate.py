"""COCO bbox evaluation with explicit category mapping and valid image ID zero handling."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import numpy as np

def evaluate(gt_path, prediction_path, category_offset=0):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = json.loads(Path(gt_path).read_text(encoding='utf-8-sig'))
    raw = json.loads(Path(prediction_path).read_text(encoding='utf-8-sig'))
    by_name = {im['file_name']: im['id'] for im in gt['images']}
    by_stem = {Path(name).stem: idx for name, idx in by_name.items()}
    if len(by_stem) != len(by_name):
        raise ValueError('Image stems are not unique; use a GT with unique names.')
    ids = set(by_name.values())
    categories = {c['id'] for c in gt['categories']}
    fixed = []
    for pred in raw:
        name = pred.get('file_name')
        raw_id = pred['image_id']
        if name in by_name:
            image_id = by_name[name]
        elif str(raw_id) in by_stem:
            image_id = by_stem[str(raw_id)]
        elif str(raw_id) in by_name:
            image_id = by_name[str(raw_id)]
        elif isinstance(raw_id, int) and raw_id in ids:
            image_id = raw_id
        else:
            raise ValueError(f'Unmatched image ID: {raw_id!r}')
        category = int(pred['category_id']) + category_offset
        if category not in categories:
            raise ValueError('Category ID outside GT classes; explicitly set --category-offset if needed.')
        fixed.append({'image_id': image_id, 'category_id': category,
                      'bbox': [float(v) for v in pred['bbox']], 'score': float(pred['score'])})
    if not fixed:
        raise ValueError('Prediction file is empty.')
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(fixed)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.maxDets = [1, 10, 100, 300]
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision']
    def ap(iou=None, area=0):
        values = precision[:, :, :, area, 3] if iou is None else precision[iou:iou+1, :, :, area, 3]
        values = values[values > -1]
        return float(values.mean() * 100) if values.size else None
    return {'AP50_95': ap(), 'AP50': ap(0), 'AP75': ap(5), 'APS': ap(area=1),
            'APM': ap(area=2), 'APL': ap(area=3), 'max_dets': 300,
            'prediction_count': len(fixed), 'category_offset': category_offset,
            'gt_sha256': hashlib.sha256(Path(gt_path).read_bytes()).hexdigest(),
            'predictions_sha256': hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', type=Path, required=True, help='COCO-format GT; this script does not change annotations.')
    p.add_argument('--predictions', type=Path, help='Existing COCO-format predictions; no model is loaded in this mode.')
    p.add_argument('--weights', type=Path, help='Alternatively run inference to produce predictions.')
    p.add_argument('--data', type=Path)
    p.add_argument('--mode', choices=['D', 'B'], default='D', help='D: learned prior; B: uniform prior.')
    p.add_argument('--category-offset', type=int, default=0)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--device', default='0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Output exists; choose a new output file.')
    prediction_path = a.predictions
    if prediction_path is None:
        if a.weights is None or a.data is None:
            p.error('Provide --predictions, or both --weights and --data.')
        import os
        os.environ['YOLO_EXP_MODE'] = a.mode
        from ultralytics import YOLO
        import yaml
        data_cfg = yaml.safe_load(a.data.read_text(encoding='utf-8-sig'))
        data_root = Path(data_cfg.get('path', '.'))
        if not data_root.is_absolute():
            data_root = Path(__file__).resolve().parents[1] / data_root
        data_cfg['path'] = str(data_root.resolve())
        a.output.parent.mkdir(parents=True, exist_ok=True)
        resolved_data = a.output.parent / (a.output.stem + '_data.yaml')
        if resolved_data.exists():
            raise SystemExit('Resolved data config exists; choose a new output name.')
        resolved_data.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding='utf-8')
        model = YOLO(str(a.weights.resolve()))
        metrics = model.val(data=str(resolved_data), split='val', imgsz=640, batch=a.batch,
                            conf=0.001, iou=0.7, max_det=300, device=a.device, save_json=True,
                            plots=False, project=str(a.output.parent / 'validation'), exist_ok=False)
        prediction_path = Path(metrics.save_dir) / 'predictions.json'
    result = evaluate(a.gt, prediction_path, a.category_offset)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
