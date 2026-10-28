from pathlib import Path
import subprocess,json
work=Path(__file__).resolve().parent
out=work if (work/'manifest.json').exists() else work/'hotpath-increment'
source=Path(r'C:\Users\Bion\My_github_things\gemma-4-E2b-Memento\engines\llama.cpp\prism-cost-aware-source')
cc=Path(r'C:\Strawberry\c\bin')
checks=json.loads((out/'verification.json').read_text())
commands=[('full-speculative-compile',[cc/'g++.exe','-std=c++17','-Icommon','-Iggml/include','-Iinclude','-Ivendor','-Isrc','-c','common/speculative.cpp','-o',out/'speculative.o'])]
for variant,flags in [('default',[]),('vector-compact',['-DGGML_PTQ1_PAIR_LUT_COMPACT','-DGGML_PTQ1_PAIR_LUT_VECTOR_SCALE'])]:
    commands.append(('full-quants-'+variant,[cc/'gcc.exe','-std=c11','-O2','-mavx2','-mfma','-mf16c','-ffp-contract=off',*flags,'-Iggml/include','-Iggml/src','-Iggml/src/ggml-cpu','-c','ggml/src/ggml-cpu/arch/x86/quants.c','-o',out/('quants-'+variant+'.o')]))
for name,args in commands:
    r=subprocess.run(list(map(str,args)),cwd=source,capture_output=True,text=True)
    (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf8')
    checks.append({'name':name,'returncode':r.returncode,'command':list(map(str,args))})
    print(name,r.returncode,flush=True)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
(out/'verification.json').write_text(json.dumps(checks,indent=2))
