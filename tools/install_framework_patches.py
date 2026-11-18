"""Apply the supplied source modifications to a clean Ultralytics 8.4.7 installation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Write patches after all checks pass; otherwise check only.')
    parser.add_argument('--target', type=Path, help='Explicit ultralytics package directory, mainly for source-tree validation.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'patch-manifest.json').read_text())
    if args.target:
        target = args.target.resolve()
        version_text = (target / '__init__.py').read_text(encoding='utf-8-sig')
        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', version_text)
        version = match.group(1) if match else None
    else:
        if sys.prefix == sys.base_prefix and not os.environ.get('CONDA_PREFIX'):
            raise SystemExit('Use a dedicated venv/conda environment, or explicitly supply --target.')
        dist = importlib.metadata.distribution('ultralytics')
        version = dist.version
        target = Path(dist.locate_file('ultralytics')).resolve()
    if version != '8.4.7':
        raise SystemExit(f'Expected ultralytics 8.4.7, found {version!r}.')
    jobs = []
    for item in manifest['files']:
        relative = Path(item['path']).relative_to('ultralytics')
        source = ROOT / 'patches' / item['path']
        destination = target / relative
        if digest(source) != item['local_sha256_text']:
            raise SystemExit(f'Patch was changed: {source}')
        old = digest(destination) if destination.is_file() else None
        if old not in (item['upstream_sha256_text'], item['local_sha256_text']):
            raise SystemExit(f'Unexpected local changes; no files written: {destination}')
        jobs.append((source, destination))
    if args.apply:
        for source, destination in jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        print(f'Applied {len(jobs)} files to {target}')
    else:
        print(f'Validated {len(jobs)} files for {target}; add --apply to write them.')

if __name__ == '__main__':
    main()
