"""Check a pinned source tree; mutate only with --apply. Never build or run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'manifest.json').read_text(encoding='utf-8'))
    source = args.source.resolve(strict=True)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != manifest['commit']:
        raise SystemExit('Wrong source commit; no changes made')
    for name, digests in manifest['files'].items():
        data = (source / name).read_text(encoding='utf-8').replace('\r\n', '\n').encode()
        if hashlib.sha256(data).hexdigest() != digests['before_sha256']:
            raise SystemExit('Source differs from pinned input: ' + name)
    patch = str(here / 'greedy-mtp.patch')
    subprocess.run(['git', '-C', str(source), 'apply', '--check', patch], check=True)
    if args.apply:
        subprocess.run(['git', '-C', str(source), 'apply', patch], check=True)
    print('Applied source patch' if args.apply else 'Patch applicable; no changes made')


if __name__ == '__main__':
    main()
