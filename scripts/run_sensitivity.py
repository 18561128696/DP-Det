"""Print or run the nine configurations transcribed from the manuscript sensitivity table."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--run', action='store_true', help='Without this flag, only print commands.')
    p.add_argument('--device', default='0')
    a = p.parse_args()
    grid = json.loads((ROOT / 'configs/sensitivity-grid.json').read_text())
    seen = set()
    for row in grid['runs']:
        if row['profile'] in seen:
            continue
        seen.add(row['profile'])
        command = [sys.executable, str(ROOT / 'scripts/train.py'), '--data', a.data,
                   '--pretrained', a.pretrained, '--variant', 'ours', '--profile', row['profile'],
                   '--name', 'sensitivity_' + row['id'], '--device', a.device]
        print(subprocess.list2cmdline(command), flush=True)
        if a.run:
            subprocess.run(command, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
