"""Check an exact composed Prism source snapshot; opt-in source apply/reverse only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def digest(path):
    return hashlib.sha256(path.read_text(encoding='utf8').encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--reverse',action='store_true')
    args=p.parse_args();here=Path(__file__).resolve().parent
    manifest=json.loads((here/'manifest.json').read_text(encoding='utf8'))
    source=args.source.resolve(strict=True)
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=manifest['commit']:raise SystemExit('Wrong source commit; no changes made')
    expected=dict(manifest['dependencies'])
    for name,hashes in manifest['files'].items():
        expected[name]=hashes['after_sha256'] if args.reverse else hashes['before_sha256']
    for name,sha in expected.items():
        path=source/name
        if (path.exists() if sha is None else not path.is_file() or digest(path)!=sha):
            raise SystemExit('Source differs from required composed snapshot: '+name)
    patch=here/'suffix-workspace.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=manifest['patch_sha256']:
        raise SystemExit('Patch digest mismatch; no changes made')
    command=['git','-C',str(source),'apply']+(['--reverse'] if args.reverse else [])
    subprocess.run(command+['--check',str(patch)],check=True)
    if args.apply or args.reverse:
        subprocess.run(command+[str(patch)],check=True)
        for name,hashes in manifest['files'].items():
            sha=hashes['before_sha256'] if args.reverse else hashes['after_sha256']
            path=source/name
            if (path.exists() if sha is None else not path.is_file() or digest(path)!=sha):
                raise SystemExit('Post-change source digest mismatch: '+name)
    print('Reversed source delta' if args.reverse else 'Applied source delta' if args.apply else 'Applicable; no source changes made')

if __name__=='__main__':main()
