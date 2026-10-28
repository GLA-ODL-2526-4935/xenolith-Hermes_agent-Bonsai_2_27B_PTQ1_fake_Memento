"""Model-free suffix parity, allocation, lifecycle and full-source compile checks."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--build-dir',type=Path,required=True)
args=p.parse_args();source=args.source.resolve();out=args.build_dir.resolve();out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_text())
for name,sha in {**manifest['dependencies'],**{k:v['after_sha256'] for k,v in manifest['files'].items()}}.items():
    assert hashlib.sha256((source/name).read_text(encoding='utf8').encode()).hexdigest()==sha,name
results=[]
def run(name,cmd):
    start=time.perf_counter();r=subprocess.run(list(map(str,cmd)),capture_output=True,text=True)
    (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf8')
    results.append({'name':name,'returncode':r.returncode,'seconds':time.perf_counter()-start})
    (out/'verification.json').write_text(json.dumps(results,indent=2))
    print(name,'PASS' if r.returncode==0 else 'FAIL',flush=True)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
run('suffix-compile',['g++','-std=c++17','-O2','-pthread','-I'+str(source/'common'),here/'test-suffix.cpp','-o',out/'suffix.exe'])
run('suffix-parity-allocation-benchmark',[out/'suffix.exe'])
text=(source/'common/speculative.cpp').read_text(encoding='utf8')
def function(name):
    a=text.index('void '+name+'(');b=text.index('{',a);depth=1;i=b+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[a:i]
stub='''
#include "suffix-workspace.h"
#include <cassert>
#include <memory>
using llama_token=int32_t; using llama_seq_id=int; using llama_tokens=std::vector<llama_token>;
#define SPC_WRN(...) ((void)0)
struct fake_profiles { int saved=0;bool fail=false;void save() {++saved;if(fail)throw 1;} };
struct fake_impl { int t_begin_us=0,n_call_begin=0;bool gen_perf=false;void begin(int,const llama_tokens&) {} };
struct common_time_meas { common_time_meas(int&,bool) {} };
struct common_speculative {
 bool source_policy=true;common_suffix_workspace<llama_token> suffix_workspace;
 common_source_policy policy;std::unique_ptr<fake_profiles> profiles;
 std::vector<fake_impl*> impls;std::vector<fake_impl*> impl_last{nullptr};
};
'''
# The actual end hook catches std::exception, as the real profile writer does.
stub=stub.replace('throw 1','throw std::runtime_error("fixture")').replace('#include <memory>','#include <memory>\n#include <stdexcept>')
test='''
int main() {
 common_speculative s;fake_impl impl;s.impls.push_back(&impl);
 s.profiles=std::make_unique<fake_profiles>();
 std::vector<int32_t> prompt{1,2,1,2,1,2},result;
 auto allocate=[&] {common_suffix_chain_reuse(prompt,1,4,s.suffix_workspace,result);assert(s.suffix_workspace.votes.capacity()>0);s.policy.active=true;};
 allocate();common_speculative_cycle_end(&s,1);assert(s.suffix_workspace.votes.capacity()>0);
 common_speculative_cycle_end(&s,0);assert(!s.policy.active && s.suffix_workspace.votes.capacity()==0 && s.profiles->saved==1);
 allocate();s.profiles->fail=true;common_speculative_cycle_end(&s,0);assert(s.suffix_workspace.matches.capacity()==0);
 allocate();common_speculative_begin(&s,0,prompt);assert(s.suffix_workspace.votes.capacity()==0 && impl.n_call_begin==1);
}
'''
(out/'hooks.cpp').write_text(stub+function('common_speculative_cycle_end')+function('common_speculative_begin')+test,encoding='utf8')
run('hooks-compile',['g++','-std=c++17','-O2','-I'+str(source/'common'),out/'hooks.cpp','-o',out/'hooks.exe'])
run('actual-request-boundary-hooks',[out/'hooks.exe'])
includes=[source/'common',source/'ggml/include',source/'include',source/'vendor',source/'src']
run('full-speculative-compile',['g++','-std=c++17',*['-I'+str(i) for i in includes],'-c',source/'common/speculative.cpp','-o',out/'speculative.o'])
