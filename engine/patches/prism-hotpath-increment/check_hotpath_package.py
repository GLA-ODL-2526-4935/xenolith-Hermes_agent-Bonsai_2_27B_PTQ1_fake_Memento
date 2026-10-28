from pathlib import Path
from unittest.mock import patch
import tempfile,json,runpy,sys,hashlib

work=Path(__file__).resolve().parent
here=work if (work/'manifest.json').exists() else work/'hotpath-increment'
source=Path(r'C:\Users\Bion\My_github_things\gemma-4-E2b-Memento\engines\llama.cpp\prism-cost-aware-source')
m=json.loads((here/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest()
with tempfile.TemporaryDirectory(dir=here) as temp:
    root=Path(temp)
    for name in m['dependencies']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/name).read_bytes())
    for name in m['files']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((here/'before'/name).read_bytes())
    def invoke(*args):
        with patch.object(sys,'argv',[str(here/'apply.py'),str(root),*args]),patch('subprocess.check_output',return_value=m['commit']+'\n'):
            runpy.run_path(str(here/'apply.py'),run_name='__main__')
    invoke()
    invoke('--apply')
    for name,h in m['files'].items():assert sha(root/name)==h['after_sha256']
    invoke('--reverse')
    for name,h in m['files'].items():assert sha(root/name)==h['before_sha256']
    bad=root/list(m['files'])[-1];bad.write_text('drift')
    first=root/list(m['files'])[0];saved=first.read_bytes()
    try:invoke('--apply')
    except AssertionError:pass
    else:raise AssertionError('Source drift was accepted')
    assert first.read_bytes()==saved,'Partial mutation on preflight failure'
print('apply_reverse_and_drift_guards,PASS (commit query mocked in isolated fixture)')
