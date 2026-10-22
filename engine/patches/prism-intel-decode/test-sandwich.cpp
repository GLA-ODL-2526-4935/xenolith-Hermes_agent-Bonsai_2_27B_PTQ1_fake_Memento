// Future target mock-API test; compile-only on preparation PC.
#include "sandwich-compile-prefix.inc"
static void env(const char * key,const char * value) {
#if defined(_WIN32)
 _putenv_s(key,value);
#else
 if(*value) setenv(key,value,1); else unsetenv(key);
#endif
}
int main() {
 struct token {int id_slot;bool is_prompt;};
 std::vector<token> prompt={{0,true}},decode={{0,false}},verify={{0,false},{0,false}},mixed={{0,true},{0,false}},multi={{0,false},{1,false}};
 assert(common_sandwich_classify(prompt,0,1)==common_cpu_phase::prefill);
 assert(common_sandwich_classify(decode,0,1)==common_cpu_phase::decode);
 assert(common_sandwich_classify(verify,0,2)==common_cpu_phase::verify);
 assert(common_sandwich_classify(mixed,0,2)==common_cpu_phase::original);
 assert(common_sandwich_classify(multi,0,2)==common_cpu_phase::original);
 assert(common_sandwich_classify(prompt,1,1)==common_cpu_phase::original);
 for(const char * bad:{"0","65","-1","1x","999999999999999999"}) {bool caught=false;try {common_sandwich_threads(bad);} catch(const std::invalid_argument &) {caught=true;}assert(caught);}
 env("LLAMA_SANDWICH_VERIFY_THREADS","");env("LLAMA_SANDWICH_VERIFY_MASK","");
 {common_threadpools pools;common_params params;llama_context ctx;pools.init(&ctx,params);pools.init_sandwich(&ctx,params);events.clear();pools.select_phase(&ctx,common_cpu_phase::prefill);assert(events.empty());}
 assert(allocated==freed);
 env("LLAMA_SANDWICH_VERIFY_THREADS","2");env("LLAMA_SANDWICH_VERIFY_MASK","5");
 {common_threadpools pools;common_params params;params.cpuparams_batch.n_threads=6;llama_context ctx;pools.init(&ctx,params);pools.init_sandwich(&ctx,params);
  for(auto phase:{common_cpu_phase::prefill,common_cpu_phase::verify,common_cpu_phase::decode,common_cpu_phase::original}) {
   events.clear();pools.select_phase(&ctx,phase);assert(events.front()=="sync");assert(events.back()=="attach");
   assert(ctx.one->count==ctx.n && ctx.batch->count==ctx.nb);
   if(phase==common_cpu_phase::verify) assert(ctx.n==2 && ctx.nb==2);
   events.clear();pools.select_phase(&ctx,phase);assert(events.empty());
  }
 }
 assert(allocated==freed);
 {common_threadpools pools;common_params params;params.n_gpu_layers=1;llama_context ctx;bool caught=false;
  try {pools.init(&ctx,params);pools.init_sandwich(&ctx,params);} catch(const std::invalid_argument &) {caught=true;}assert(caught);}
 assert(allocated==freed);
 env("LLAMA_SANDWICH_VERIFY_THREADS","");env("LLAMA_SANDWICH_VERIFY_MASK","");
}
