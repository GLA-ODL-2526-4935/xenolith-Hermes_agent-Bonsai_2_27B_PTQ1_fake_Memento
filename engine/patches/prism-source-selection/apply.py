"""Check only by default; apply/reverse this exact source delta, never build/run."""
import argparse, hashlib, json, subprocess
from pathlib import Path
def digest(p): return hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--apply',action='store_true');group.add_argument('--reverse',action='store_true')
    args=parser.parse_args();here=Path(__file__).resolve().parent
    m=json.loads((here/'manifest.json').read_text())
    source=args.source.resolve(strict=True)
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=m['commit']: raise SystemExit('Wrong source commit; no changes made')
    expected=dict(m['input_files'])
    for name,data in m['files'].items():
        expected[name]=data['after_sha256'] if args.reverse else data['before_sha256']
    def check(name,sha):
        p=source/name
        if (p.exists() if sha is None else not p.is_file() or digest(p)!=sha):
            raise SystemExit('Unexpected source: '+name)
    for name,sha in expected.items(): check(name,sha)
    patch=here/'source-selection.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=m['patch_sha256']:
        raise SystemExit('Patch digest mismatch; no changes made')
    cmd=['git','-C',str(source),'apply']+(['--reverse'] if args.reverse else [])
    subprocess.run(cmd+['--check','--whitespace=error',str(patch)],check=True)
    if args.apply or args.reverse:
        subprocess.run(cmd+[str(patch)],check=True)
        for name,data in m['files'].items(): check(name,data['before_sha256'] if args.reverse else data['after_sha256'])
    print('Source reversed' if args.reverse else 'Source applied' if args.apply else 'Applicable; no source changed')
if __name__=='__main__': main()
