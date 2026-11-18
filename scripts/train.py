"""Portable training entry; explicit model variant and loss profile avoid inherited shell settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--variant', choices=['ours', 'uniform', 'fap2-only', 'baseline'], required=True)
    p.add_argument('--profile', required=True, help='Key from configs/loss-profiles.json, or ciou for baseline/fap2-only.')
    p.add_argument('--name', required=True)
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument('--pretrained', type=Path)
    init.add_argument('--scratch', action='store_true')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--warmup-epochs', type=float, default=5.0, help='Use 3 for the supplied NWPU protocol.')
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    a.data = a.data.resolve()
    if not a.data.is_file():
        raise SystemExit(f'Missing data config: {a.data}')
    if a.pretrained and not a.pretrained.is_file():
        raise SystemExit(f'Missing pretrained weights: {a.pretrained}')
    import yaml
    data_cfg = yaml.safe_load(a.data.read_text(encoding='utf-8-sig'))
    data_root = Path(data_cfg.get('path', ROOT))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    data_cfg['path'] = str(data_root.resolve())
    profiles = json.loads((ROOT / 'configs/loss-profiles.json').read_text())
    env = {'USE_DLKA': str(a.variant not in ['fap2-only', 'baseline']),
           'USE_BS_WIOU': str(a.profile != 'ciou'),
           'YOLO_EXP_MODE': 'B' if a.variant in ['uniform', 'fap2-only'] else 'D'}
    if a.variant in ['baseline', 'fap2-only'] and a.profile != 'ciou':
        raise SystemExit('Use --profile ciou for baseline/fap2-only.')
    if a.profile != 'ciou':
        if a.profile not in profiles:
            raise SystemExit(f'Unknown loss profile: {a.profile}')
        env.update({k: str(v) for k, v in profiles[a.profile]['environment'].items()})
    for key in ['USE_DLKA', 'USE_BS_WIOU', 'YOLO_EXP_MODE', 'BSWIOU_ALPHA', 'BSWIOU_DELTA', 'BSWIOU_SCALE_BIAS', 'BSWIOU_SCALE_NUM']:
        os.environ.pop(key, None)
    os.environ.update(env)
    from ultralytics import YOLO, __version__
    import ultralytics
    import torch
    from ultralytics.nn.modules.block import C2f_FAP2  # fail early if patches were not installed
    project = ROOT / 'runs'
    run_dir = project / a.name
    if run_dir.exists():
        raise SystemExit('Run name already exists; choose a new --name to preserve existing experiments.')
    record_dir = ROOT / 'run-settings'
    record_dir.mkdir(exist_ok=True)
    record = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'environment': env, 'ultralytics_version': __version__, 'ultralytics_path': ultralytics.__file__,
              'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'pretrained_sha256': hashlib.sha256(a.pretrained.read_bytes()).hexdigest() if a.pretrained else None}
    settings_path = record_dir / (a.name + '.json')
    if settings_path.exists():
        raise SystemExit('Run settings already exist; choose a new --name.')
    settings_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    model_yaml = 'yolov8s.yaml' if a.variant == 'baseline' else str(ROOT / 'patches/ultralytics/cfg/models/v8/yolov8s_fap2.yaml')
    model = YOLO(model_yaml)
    if a.pretrained:
        model.load(str(a.pretrained.resolve()))
    resolved_data = record_dir / (a.name + '_data.yaml')
    resolved_data.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding='utf-8')
    model.train(data=str(resolved_data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
                device=a.device, project=str(project), name=a.name, exist_ok=False, optimizer='SGD',
                lr0=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=a.warmup_epochs,
                patience=50, deterministic=True, seed=a.seed, max_det=300, pretrained=bool(a.pretrained))

if __name__ == '__main__':
    main()
