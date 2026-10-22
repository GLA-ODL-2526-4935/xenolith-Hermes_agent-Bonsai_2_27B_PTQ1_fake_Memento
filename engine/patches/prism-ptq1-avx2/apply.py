"""Check pinned source by default; --apply installs the PTQ1 AVX2 source patch only."""
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
        path = source / name
        if digests['before_sha256'] is None:
            if path.exists():
                raise SystemExit('Would overwrite an existing file: ' + name)
        elif hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() != digests['before_sha256']:
            raise SystemExit('Source differs from pinned input: ' + name)
    patch_file = here / 'ptq1-avx2.patch'
    if hashlib.sha256(patch_file.read_bytes()).hexdigest() != manifest['patch_sha256']:
        raise SystemExit('Patch digest mismatch; no changes made')
    patch = str(patch_file)
    subprocess.run(['git', '-C', str(source), 'apply', '--check', patch], check=True)
    if args.apply:
        subprocess.run(['git', '-C', str(source), 'apply', patch], check=True)
        for name, digests in manifest['files'].items():
            actual = hashlib.sha256((source / name).read_text(encoding='utf-8').encode()).hexdigest()
            if actual != digests['after_sha256']:
                raise SystemExit('Post-apply source digest mismatch: ' + name)
    print('Applied PTQ1 AVX2 source patch' if args.apply else 'Applicable; no source changes made')


if __name__ == '__main__':
    main()
