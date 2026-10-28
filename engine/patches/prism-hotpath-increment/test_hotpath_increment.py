from pathlib import Path
import subprocess,json,ctypes,os
work=Path(__file__).resolve().parent
root=work if (work/'manifest.json').exists() else work/'hotpath-increment'
cc=Path(r'C:\Strawberry\c\bin\g++.exe')
env=dict(os.environ);env['PATH']=str(cc.parent)+os.pathsep+env['PATH']
results=[]
def run(name,args):
    r=subprocess.run(list(map(str,args)),cwd=root,env=env,capture_output=True,text=True)
    (root/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf8')
    results.append({'name':name,'returncode':r.returncode})
    (root/'verification.json').write_text(json.dumps(results,indent=2))
    print(name,r.returncode,flush=True)
    if r.returncode:raise RuntimeError((r.stdout+r.stderr)[-3000:])
old=(root/'before/common/suffix-workspace.h').read_text()
a=old.index('template<class Token>\nstruct common_suffix_workspace')
b=old.index('// Borrow',a)
old=old[:a]+old[b:]
old=old.replace('common_suffix_chain_reuse','common_suffix_chain_previous')
(root/'suffix-previous.h').write_text(old)
test=(root/'test-suffix.cpp').read_text()
test=test.replace('#include "suffix-workspace.h"','#include "suffix-workspace.h"\n#include "suffix-previous.h"')
test=test.replace('else result=common_suffix_chain(p,last,4);','else common_suffix_chain_previous(p,last,4,w,result);')
(root/'test-suffix-current.cpp').write_text(test)
run('suffix-compile',[cc,'-O2','-std=c++17','-pthread','test-suffix-current.cpp','-o','suffix.exe'])
run('suffix-parity-benchmark',[root/'suffix.exe'])
if not ctypes.windll.kernel32.IsProcessorFeaturePresent(40):raise RuntimeError('Host OS does not report AVX2; no AVX execution')
for compact in [0,1]:
    flags=['-DGGML_PTQ1_PAIR_LUT_COMPACT'] if compact else []
    run('vector-compile-'+str(compact),[cc,'-O3','-std=c++17','-mavx2','-ffp-contract=off',*flags,'test-vector.cpp','-o',f'vector{compact}.exe'])
    run('vector-parity-benchmark-'+str(compact),[root/f'vector{compact}.exe'])
