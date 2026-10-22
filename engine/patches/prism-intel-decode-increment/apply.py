"""Check an exact v3 source tree; --apply or --reverse changes source only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def digest(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--reverse',action='store_true')
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    m=json.loads((here/'manifest.json').read_text())
    base_bytes=(here/'base-manifest.json').read_bytes()
    if hashlib.sha256(base_bytes).hexdigest()!=m['base_manifest_sha256']:
        raise SystemExit('Prerequisite manifest digest mismatch; no changes made')
    base=json.loads(base_bytes)
    if base['commit']!=m['commit'] or base['patch_id']!=m['requires_patch_id'] or base['patch_sha256']!=m['requires_patch_sha256']:
        raise SystemExit('Wrong prerequisite package; no changes made')
    source=a.source.resolve(strict=True)
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=m['commit']: raise SystemExit('Wrong source commit; no changes made')
    # Check every v3 source, not merely this delta's two overlapping files.
    expected={name:data['after_sha256'] for name,data in base['files'].items()}
    for name,data in m['files'].items():
        expected[name]=data['after_sha256'] if a.reverse else data['before_sha256']
    for name,sha in expected.items():
        file=source/name
        if (file.exists() if sha is None else not file.is_file() or digest(file)!=sha):
            raise SystemExit('Source differs from required input: '+name)
    patch=here/'decode-increment.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=m['patch_sha256']:
        raise SystemExit('Patch digest mismatch; no changes made')
    command=['git','-C',str(source),'apply']+(['--reverse'] if a.reverse else [])
    subprocess.run(command+['--check',str(patch)],check=True)
    if a.apply or a.reverse:
        subprocess.run(command+[str(patch)],check=True)
        for name,data in m['files'].items():
            sha=data['before_sha256'] if a.reverse else data['after_sha256']
            file=source/name
            if (file.exists() if sha is None else not file.is_file() or digest(file)!=sha):
                raise SystemExit('Post-change source mismatch: '+name)
    print('Reversed source delta' if a.reverse else 'Applied source delta' if a.apply else 'Applicable; no source changes made')

if __name__=='__main__': main()
