"""Apply or reverse only the two verified header changes; default is check-only."""
from pathlib import Path
import argparse,hashlib,json,subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
mode=p.add_mutually_exclusive_group()
mode.add_argument('--apply',action='store_true')
mode.add_argument('--reverse',action='store_true')
a=p.parse_args();here=Path(__file__).resolve().parent;source=a.source.resolve(strict=True)
m=json.loads((here/'manifest.json').read_text())
digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert head==m['commit'],'Wrong source commit'
assert digest((here/'hotpath.patch').read_text())==m['patch_sha256'],'Changed patch'
updates=[]
for name,expected in m['dependencies'].items():
    assert digest((source/name).read_text(encoding='utf8'))==expected,'Dependency drift: '+name
for name,hashes in m['files'].items():
    target=(source/name).resolve(strict=True)
    assert target.is_relative_to(source),'Source path escapes checkout'
    expected=hashes['after_sha256' if a.reverse else 'before_sha256']
    assert digest(target.read_text(encoding='utf8'))==expected,'Source drift: '+name
    folder='before' if a.reverse else 'after'
    payload=(here/folder/name).read_text(encoding='utf8')
    assert digest(payload)==hashes['before_sha256' if a.reverse else 'after_sha256'],'Payload drift: '+name
    updates.append((target,payload))
if a.apply or a.reverse:
    for target,payload in updates:
        target.write_text(payload,encoding='utf8',newline='\n')
print('Reversed' if a.reverse else 'Applied' if a.apply else 'Applicable; no source changes')
