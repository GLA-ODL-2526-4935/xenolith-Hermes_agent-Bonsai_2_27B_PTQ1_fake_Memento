"""Model-free tests of the actual edited source hooks plus compilation checks."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import argparse,sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='Exact source checkout AFTER applying this delta')
parser.add_argument('--build-dir',type=Path,default=Path('lifecycle-test-output'))
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
BASE=args.source.resolve(strict=True)
BUILD=args.build_dir.resolve();BUILD.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'manifest.json').read_text())
for name,sha in {**manifest['dependencies'],**{n:d['after_sha256'] for n,d in manifest['files'].items()}}.items():
 if hashlib.sha256((BASE/name).read_text(encoding='utf8').encode()).hexdigest()!=sha:raise SystemExit('Test source differs: '+name)
RESULTS=[]
def run(name,args,env=None):
 start=time.monotonic();p=subprocess.run([str(x) for x in args],capture_output=True,text=True,env=env)
 (BUILD/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf8')
 RESULTS.append({'name':name,'returncode':p.returncode,'seconds':round(time.monotonic()-start,3),'command':[str(x) for x in args]})
 print(name,'PASS' if p.returncode==0 else 'FAIL',flush=True)
 if p.returncode:raise RuntimeError((name,(p.stdout+p.stderr)[-3500:]))
def compile_run(name,text):
 source=BUILD/(name+'.cpp');source.write_text(text,encoding='utf8')
 exe=BUILD/(name+'.exe')
 run(name+'-compile',['g++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-pthread',source,'-o',exe])
 run(name,[exe])
try:
 run('policy-compile',['g++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-pthread',ROOT/'test-policy.cpp','-I'+str(BASE/'ggml/src'),'-o',BUILD/'policy.exe'])
 env=os.environ.copy();env['GGML_RUNTIME_TELEMETRY']='1'
 run('policy',[BUILD/'policy.exe'],env)

 # Execute the edited graph selection statements with a tiny get_rows oracle.
 prefix='''#include <vector>\n#include <cassert>\n#include <cstdio>\nusing V=std::vector<int>; struct ggml_tensor{V rows;};
struct C{bool embeddings_nextn_masked;}; struct R{ggml_tensor *t_h_nextn;};
ggml_tensor *ggml_get_rows(void*,ggml_tensor*x,ggml_tensor*ids){auto*y=new ggml_tensor;for(int i:ids->rows)y->rows.push_back(x->rows.at(i));return y;}
void cb(ggml_tensor*,const char*,int){}\n'''
 funcs=[]
 for version in ['before','after']:
  for model in ['qwen35','qwen35moe']:
   s=(ROOT/'fixtures/graph-before.inc').read_text() if version=='before' else (BASE/'src/models'/f'{model}.cpp').read_text()
   start=s.rindex('    cb(cur, "h_nextn", -1);');end=s.index('    cb(cur, "mtp_shared_head_norm", -1);',start)
   funcs.append('std::pair<V,V> '+version+'_'+model+'(const V &rows,const V &ids,bool masked){ggml_tensor x{rows},out{ids};auto*cur=&x;auto*inp_out_ids=&out;void*ctx0=nullptr;C cparams{masked};R storage{};auto*res=&storage;\n'+s[start:end]+ '\nauto answer=std::make_pair(res->t_h_nextn->rows,cur->rows);if(cur!=&x)delete cur;return answer;}\n')
 main='''int main(){int old_bad=0;for(int n=1;n<=9;++n){V rows;for(int i=0;i<n;++i)rows.push_back(100+i*71);
for(unsigned bits=0;bits<(1u<<n);++bits){V ids,expected;for(int i=n-1;i>=0;--i)if(bits&(1u<<i)){ids.push_back(i);expected.push_back(rows[i]);}
for(bool masked:{false,true}){auto a=after_qwen35(rows,ids,masked),b=after_qwen35moe(rows,ids,masked);assert(a.second==expected&&b.second==expected);assert(a.first==(masked?expected:rows)&&b.first==(masked?expected:rows));
auto old=before_qwen35(rows,ids,masked);if(masked&&old.first!=expected)++old_bad;}}}assert(old_bad>0);std::printf("actual graph slices: PASS; pre-fix masked mismatches=%d\\n",old_bad);}
'''
 compile_run('masked-graph',prefix+'\n'.join(funcs)+main)

 s=(BASE/'common/sampling.cpp').read_text();start=s.index('std::vector<llama_token> common_sampler_sample_and_accept_n(');end=s.index('\nstd::vector<llama_token> common_sampler_sample_and_accept_n(',start+1)
 compile_run('actual-sampler', '''#include <vector>\n#include <cassert>\n#include <cstdio>\n#define GGML_ASSERT assert
using llama_token=int;using llama_tokens=std::vector<int>;struct common_sampler{llama_tokens history,indices;};struct llama_context{llama_tokens targets;};
int common_sampler_sample(common_sampler*s,llama_context*c,int i,bool){s->indices.push_back(i);return c->targets.at(i);}
void common_sampler_accept(common_sampler*s,int id,bool){s->history.push_back(id);}
'''+s[start:end]+'''
int main(){for(int n=0;n<=16;++n)for(int reject=0;reject<=n;++reject){common_sampler sampler;llama_context ctx;ctx.targets.resize(100,0);std::vector<int>idx;llama_tokens draft;
for(int i=0;i<=n;++i){idx.push_back(2*i+3);ctx.targets[idx.back()]=100+i;if(i<n)draft.push_back(i==reject?-100:100+i);}
auto result=common_sampler_sample_and_accept_n(&sampler,&ctx,idx,draft,false);int count=reject<n?reject+1:n+1;
assert(result.size()==(size_t)count&&sampler.history==result&&sampler.indices.size()==result.size());for(int i=0;i<count;++i){assert(result[i]==100+i);assert(sampler.indices[i]==idx[i]);}}
std::puts("actual verifier: all rejection positions, full acceptance, bonus and sampler history PASS");}
''')

 s=(BASE/'ggml/src/ggml-alloc.c').read_text();start=s.index('static bool ggml_gallocr_memory_preflight(');end=s.index('\nstatic bool ggml_gallocr_reserve_n_impl(',start)
 gate_start=s.index('bool ggml_gallocr_alloc_graph(');gate_end=s.index('    if (ggml_gallocr_needs_realloc(',gate_start)
 gate=s[gate_start:gate_end]+'    return true; }\n'
 compile_run('actual-budget-preflight', '#include "'+(BASE/'ggml/src/runtime-memory-policy.h').resolve().as_posix()+'"\n'+'''
#include <cassert>
#include <cstdio>
struct Alloc{int n_chunks;size_t sizes[3];};struct Buf{size_t bytes;};
struct Graph{int n_buffers;Alloc**buf_tallocs;Buf**buffers;bool memory_config_valid;size_t compute_limit;bool memory_reservation_failed=false;};using ggml_gallocr_t=Graph*;
size_t ggml_dyn_tallocr_max_size(Alloc*a,int c){return a->sizes[c];}size_t ggml_vbuffer_size(Buf*b){return b->bytes;}
'''+s[start:end]+gate+'''
int main(){Alloc a{2,{128,128}},b{1,{64}};Alloc*allocs[]={&a,&a,&b};Buf olda{512},oldb{32};Buf*bufs[]={&olda,&olda,&oldb};Graph g{3,allocs,bufs,true,576};size_t total=0;
assert(ggml_gallocr_memory_preflight(&g,&total)&&total==576);g.compute_limit=575;assert(!ggml_gallocr_memory_preflight(&g,&total));assert(olda.bytes==512&&oldb.bytes==32);
g.compute_limit=0;assert(ggml_gallocr_memory_preflight(&g,&total)&&total==576);g.memory_config_valid=false;assert(!ggml_gallocr_memory_preflight(&g,&total));g.memory_config_valid=true;
g.memory_reservation_failed=true;assert(!ggml_gallocr_alloc_graph(&g,nullptr));g.memory_reservation_failed=false;assert(ggml_gallocr_alloc_graph(&g,nullptr));a.sizes[0]=SIZE_MAX;a.sizes[1]=1;assert(!ggml_gallocr_memory_preflight(&g,&total));
std::puts("actual budget preflight: alias dedup, retained capacity, exact cap, invalid config and overflow PASS");}
''')

 s=(BASE/'ggml/src/ggml-backend.cpp').read_text();start=s.index('            if (input->flags & GGML_TENSOR_FLAG_INPUT) {');end=s.index('                // wait for the split backend',start)
 hook=s[start:end]+ ' /* fake non-user async path */ }\n'
 compile_run('actual-input-wait', '''#include <cassert>\n#include <cstdio>\n#include <initializer_list>\n#define GGML_TENSOR_FLAG_INPUT 1
struct Tensor{int flags;};struct S{void*events[1][1];int cur_copy=0;};int waits=0,copies=0;
void ggml_backend_event_synchronize(void*){++waits;}void ggml_backend_synchronize(void*){++waits;}void ggml_backend_tensor_copy(Tensor*,Tensor*){++copies;}
void test(unsigned bits,bool sync_once,bool events){S storage{};auto*sched=&storage;int split_backend_id=0;void*split_backend=nullptr;int event=0;storage.events[0][0]=events?&event:nullptr;bool user_synced=false;waits=copies=0;int expected_waits=0,expected_copies=0;bool previous=false;
for(int i=0;i<8;++i){Tensor t{(bits&(1u<<i))?1:0},dst{};auto*input=&t;auto*input_cpy=&dst;bool user=t.flags!=0;if(user){++expected_copies;if(!sync_once||!previous)++expected_waits;}previous=user;
'''+hook+'''}assert(copies==expected_copies&&waits==expected_waits);}
int main(){for(unsigned bits=0;bits<256;++bits)for(bool enabled:{false,true})for(bool events:{false,true})test(bits,enabled,events);std::puts("actual input wait hook: consecutive runs and reset across non-user inputs PASS");}
''')

 # Full edited translation units, not only helper scaffolds. This is a compile
 # check, not a linked runtime or model/backend validation.
 inc=['-I'+str(BASE/'ggml/src'),'-I'+str(BASE/'ggml/include'),'-I'+str(BASE/'ggml/src')]
 run('full-allocator-compile',['gcc','-std=c11',*inc,'-c',BASE/'ggml/src/ggml-alloc.c','-o',BUILD/'alloc.o'])
 run('full-backend-compile',['g++','-std=c++17',*inc,'-c',BASE/'ggml/src/ggml-backend.cpp','-o',BUILD/'backend.o'])
 for model in ['qwen35','qwen35moe']:
  run('full-'+model+'-compile',['g++','-std=c++17',*inc,'-I'+str(BASE/'include'),'-I'+str(BASE/'src'),'-I'+str(BASE/'src/models'),'-c',BASE/'src/models'/f'{model}.cpp','-o',BUILD/(model+'.o')])
 for mode,flags in [('off',[]),('on',['-DGGML_CPU_CACHELINE_SCHEDULE']),('qigen',['-DGGML_CPU_CACHELINE_SCHEDULE','-DGGML_PTQ1_ENABLE_QIGEN','-mavx2'])]:
  # MinGW's Windows SDK lacks thread power-throttling declarations. Select
  # the existing older-Windows priority branch for this compile check only.
  run('full-cpu-'+mode+'-compile',['gcc','-std=gnu11','-D_GNU_SOURCE','-D_WIN32_WINNT=0x0601',*flags,*inc,'-I'+str(BASE/'ggml/src/ggml-cpu'),'-c',BASE/'ggml/src/ggml-cpu/ggml-cpu.c','-o',BUILD/('cpu-'+mode+'.o')])
 run('full-server-compile',['g++','-std=c++17',*(['-Wa,-mbig-obj'] if os.name=='nt' else []),*inc,'-I'+str(BASE/'include'),'-I'+str(BASE/'tools/server'),'-I'+str(BASE/'tools/mtmd'),'-I'+str(BASE/'common'),'-I'+str(BASE/'vendor'),'-c',BASE/'tools/server/server-context.cpp','-o',BUILD/'server.o'])
finally:
 (BUILD/'verification.json').write_text(json.dumps({'scope':'model-free policy/hooks executed; full edited allocator/backend/CPU/models compile only','results':RESULTS},indent=2),encoding='utf8')

