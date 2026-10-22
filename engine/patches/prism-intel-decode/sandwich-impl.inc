// Included by common.cpp: the original pools retain ownership and fallback.
void common_threadpools::init_sandwich(llama_context * ctx,const common_params & params) {
    if (threadpool_verify) throw std::logic_error("Sandwich pools already initialized");
    const char * value=std::getenv("LLAMA_SANDWICH_VERIFY_THREADS");
    const char * mask=std::getenv("LLAMA_SANDWICH_VERIFY_MASK");
    if ((!value || !*value) && mask && *mask) throw std::invalid_argument("Sandwich mask requires explicit verify threads");
    const int n=common_sandwich_threads(value);
    if (!n) return;
    if (params.n_gpu_layers!=0 || params.n_parallel!=1 || params.embedding)
        throw std::invalid_argument("Sandwich candidate requires CPU-only, one slot, text generation");
    if (!threadpool || !free_fn) throw std::runtime_error("Sandwich requires a valid CPU pool");
    if (params.cpuparams.n_threads<1 || params.cpuparams_batch.n_threads<1)
        throw std::invalid_argument("Sandwich requires resolved positive decode/prefill thread counts");
    auto * cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * reg=ggml_backend_dev_backend_reg(cpu_dev);
    auto * new_fn=(decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_new");
    pause_fn=(decltype(ggml_threadpool_pause) *)ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_pause");
    if (!new_fn || !pause_fn) throw std::runtime_error("Sandwich CPU pool API unavailable");
    auto config=params.cpuparams;
    config.n_threads=n;
    if (common_sandwich_mask(mask,config.cpumask,n)) { config.mask_valid=true; config.strict_cpu=true; }
    // An inherited decode mask also needs enough permitted CPUs.
    if (config.mask_valid) {
        int permitted=0;
        for (bool bit:config.cpumask) permitted+=bit;
        if (permitted<n) throw std::invalid_argument("Inherited CPU mask is too small for verify threads");
    }
    auto tpp=ggml_threadpool_params_from_cpu_params(config);
    tpp.paused=true;
    threadpool_verify=new_fn(&tpp);
    if (!threadpool_verify) throw std::runtime_error("Sandwich verify pool creation failed");
    sandwich_counts[0]=params.cpuparams.n_threads;
    sandwich_counts[1]=params.cpuparams_batch.n_threads;
    sandwich_counts[2]=n;
    sandwich_phase=common_cpu_phase::original;
    COM_INF("Sandwich source candidate enabled: decode=%d prefill=%d verify=%d; target parity pending\n",
        sandwich_counts[0],sandwich_counts[1],sandwich_counts[2]);
    (void)ctx;
}

void common_threadpools::select_phase(llama_context * ctx,common_cpu_phase phase) {
    if (!threadpool_verify || phase==sandwich_phase) return;
    // Drain every outstanding graph before changing pools or counts. No KV edits.
    llama_synchronize(ctx);
    pause_fn(threadpool);
    if (threadpool_batch) pause_fn(threadpool_batch);
    pause_fn(threadpool_verify);
    if (phase==common_cpu_phase::original) {
        llama_set_n_threads(ctx,sandwich_counts[0],sandwich_counts[1]);
        llama_attach_threadpool(ctx,threadpool,threadpool_batch);
    } else {
        auto * selected=phase==common_cpu_phase::verify ? threadpool_verify :
                       phase==common_cpu_phase::prefill && threadpool_batch ? threadpool_batch : threadpool;
        const int count=phase==common_cpu_phase::verify ? sandwich_counts[2] :
                        phase==common_cpu_phase::prefill ? sandwich_counts[1] : sandwich_counts[0];
        // Both handles/counts match: a one-token prompt is still prefill, and
        // short verification cannot be forced onto the prefill plan by n_tokens.
        llama_set_n_threads(ctx,count,count);
        llama_attach_threadpool(ctx,selected,selected);
    }
    sandwich_phase=phase;
}

void common_init_result::select_cpu_phase(common_cpu_phase phase) {
    if (pimpl->context) pimpl->threadpools.select_phase(pimpl->context.get(),phase);
}

void common_init_result::init_cpu_phases(const common_params & params) {
    if (pimpl->context) pimpl->threadpools.init_sandwich(pimpl->context.get(),params);
}
