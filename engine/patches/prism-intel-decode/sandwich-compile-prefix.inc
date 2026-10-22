#include <cassert>
#include <cstring>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "sandwich-policy.h"
#define GGML_MAX_N_THREADS 512
#define GGML_ASSERT assert
#define COM_INF(...) ((void)0)
#define COM_WRN(...) ((void)0)
struct ggml_threadpool { int count; int id; bool paused; };
using ggml_threadpool_t=ggml_threadpool *;
struct ggml_threadpool_params { bool cpumask[512]; int n_threads; int prio; uint32_t poll; bool strict_cpu; bool paused; };
struct common_cpu_params { int n_threads=4; bool cpumask[512]={}; bool mask_valid=false; int priority=0; uint32_t poll=0; bool strict_cpu=false; };
struct common_params { common_cpu_params cpuparams,cpuparams_batch; int n_gpu_layers=0,n_parallel=1; bool embedding=false; };
static std::vector<std::string> events;
static int allocated=0,freed=0;
struct llama_context { int n=0,nb=0; ggml_threadpool * one=nullptr,*batch=nullptr; };
static void ggml_threadpool_params_init(ggml_threadpool_params * p,int n) { *p={};p->n_threads=n; }
static bool ggml_threadpool_params_match(const ggml_threadpool_params * a,const ggml_threadpool_params * b) {return a->n_threads==b->n_threads && a->prio==b->prio && a->poll==b->poll && a->strict_cpu==b->strict_cpu && !std::memcmp(a->cpumask,b->cpumask,sizeof(a->cpumask));}
static ggml_threadpool * ggml_threadpool_new(ggml_threadpool_params * p) {return new ggml_threadpool{p->n_threads,++allocated,p->paused};}
static void ggml_threadpool_free(ggml_threadpool * p) {if(p){++freed;delete p;}}
static void ggml_threadpool_pause(ggml_threadpool * p) {assert(p);p->paused=true;events.push_back("pause");}
static void llama_synchronize(llama_context *) {events.push_back("sync");}
static void llama_set_n_threads(llama_context * c,int n,int nb) {c->n=n;c->nb=nb;events.push_back("counts");}
static void llama_attach_threadpool(llama_context * c,ggml_threadpool * a,ggml_threadpool * b) {c->one=a;c->batch=b?b:a;events.push_back("attach");}
enum {GGML_BACKEND_DEVICE_TYPE_CPU};
static int * ggml_backend_dev_by_type(int) {static int device;return &device;}
static int * ggml_backend_dev_backend_reg(int * d) {return d;}
static void * ggml_backend_reg_get_proc_address(int *,const char * name) {
 if(!std::strcmp(name,"ggml_threadpool_new")) return (void *)ggml_threadpool_new;
 if(!std::strcmp(name,"ggml_threadpool_free")) return (void *)ggml_threadpool_free;
 if(!std::strcmp(name,"ggml_threadpool_pause")) return (void *)ggml_threadpool_pause;
 return nullptr;
}
struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}
struct common_threadpools {
    common_threadpools() = default;
    ~common_threadpools();

    common_threadpools(const common_threadpools &) = delete;
    common_threadpools & operator=(const common_threadpools &) = delete;

    void init(llama_context * ctx, const common_params & params);
    void init_sandwich(llama_context * ctx, const common_params & params);
    void select_phase(llama_context * ctx, common_cpu_phase phase);

private:
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    ggml_threadpool * threadpool_verify = nullptr;
    int sandwich_counts[3] = {0,0,0};
    common_cpu_phase sandwich_phase = common_cpu_phase::original;
    decltype(ggml_threadpool_pause) * pause_fn = nullptr;
    decltype(ggml_threadpool_free) * free_fn = nullptr;
};
struct common_init_result {
 struct impl { common_threadpools threadpools; std::unique_ptr<llama_context> context; };
 std::unique_ptr<impl> pimpl;
 void select_cpu_phase(common_cpu_phase);
 void init_cpu_phases(const common_params &);
};
common_threadpools::~common_threadpools() {
    if (!free_fn) {
        return;
    }
    free_fn(threadpool);
    free_fn(threadpool_batch);
    if (threadpool_verify) free_fn(threadpool_verify);
}

void common_threadpools::init(llama_context * ctx, const common_params & params) {
    GGML_ASSERT(!threadpool);
    GGML_ASSERT(!threadpool_batch);

    COM_INF("llama threadpool init, n_threads = %d\n", (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        COM_WRN("%s", "no CPU backend found\n");
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    // each pool needs to match the respective n_threads exactly
    // see: https://github.com/ggml-org/llama.cpp/pull/27138#issuecomment-5332307332
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            COM_WRN("batch threadpool create failed : n_threads %d\n", tpp_batch.n_threads);
            return;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        COM_WRN("threadpool create failed : n_threads %d\n", tpp.n_threads);
        free_fn(threadpool_batch);
        threadpool_batch = nullptr;
        return;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
}

#include "sandwich-impl.inc"
