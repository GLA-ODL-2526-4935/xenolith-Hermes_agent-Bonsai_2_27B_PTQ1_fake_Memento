#include "source-policy.h"
#include <memory>
#include <string>
#include <cassert>
#include <cinttypes>
#define GGML_ASSERT assert
#define SPC_DBG(...) ((void)0)
using llama_seq_id=int;
using llama_tokens=std::vector<int>;
enum common_speculative_type { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
static std::string common_speculative_type_to_str(common_speculative_type) { return "mtp"; }
struct common_speculative_draft_params { bool drafting=false;int n_max=-1,n_past=0,id_last=0;const llama_tokens * prompt=nullptr;llama_tokens * result=nullptr; };
using common_speculative_draft_params_vec=std::vector<common_speculative_draft_params>;
struct common_time_meas { common_time_meas(int64_t &,bool){} };
static int64_t ggml_time_us(){return 1;}
static void llama_synchronize(void *){}
struct common_speculative_impl {
 common_speculative_type type=COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
 int64_t t_draft_us=0,t_accept_us=0;bool gen_perf=true;
 size_t n_call_draft=0,n_gen_drafts=0,n_gen_tokens=0,n_acc_drafts=0,n_acc_tokens=0,n_call_accept=0;
 std::vector<size_t> n_acc_tokens_per_pos;
 virtual ~common_speculative_impl()=default;
 virtual void draft(common_speculative_draft_params_vec &){}
 virtual void accept(int,uint16_t,bool){}
};
struct common_speculative_impl_draft_mtp:common_speculative_impl {struct {void *ctx_dft=nullptr;} params;};
struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
    bool source_policy = false;
    bool suffix_enabled = false;
    int source_cap = 0;
    int source_force = -1; // explicit later-run A/B control, not an autotuner
    common_source_policy policy; // single-slot experimental server mode only

};
void common_speculative_cycle_begin(common_speculative * spec, llama_seq_id seq_id, int cap, bool greedy, int context) {
    if (!spec || !spec->source_policy || seq_id!=0) return;
    if (!greedy) { spec->policy.active=false; return; } // legacy path for stochastic requests
    const bool replay=spec->policy.active;
    cap=std::max(0,std::min(cap,spec->source_cap));
    spec->policy.begin(cap,spec->suffix_enabled,ggml_time_us(),context/4096);
    if (!replay && spec->source_force>=0) {
        const int a=spec->source_force;
        spec->policy.arm=(a>4?a-4:a)<=cap?a:0;
    }
}
void common_speculative_cycle_finish(common_speculative * spec, llama_seq_id seq_id, int committed) {
    if (!spec || !spec->source_policy || seq_id!=0) return;
    // process() may enqueue draft catch-up asynchronously. Synchronize before
    // rewarding any source, including warm off, rather than charge it later.
    if (spec->policy.active) {
        for (auto & impl:spec->impls) {
            auto * mtp=static_cast<common_speculative_impl_draft_mtp *>(impl.get());
            llama_synchronize(mtp->params.ctx_dft);
        }
    }
    [[maybe_unused]] const int arm=spec->policy.arm;
    const int64_t now=ggml_time_us();
    [[maybe_unused]] const int64_t elapsed=now-spec->policy.started;
    if (spec->policy.finish(committed,now)) {
        SPC_DBG("source arm=%d unique_tokens=%d elapsed_us=%" PRId64 " bucket=%d cycles=%" PRIu64 " warm_off=1\n",
                arm,committed,elapsed,spec->policy.context_bucket,spec->policy.cycles);
    }
}
void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    if (spec->source_policy && spec->policy.active) {
        auto & dp=dparams[0];
        spec->impl_last[0]=nullptr; // no stale MTP ownership on off/lookup miss
        [[maybe_unused]] const int arm=spec->policy.arm;
        if (arm==0 || arm>4) {
            if (arm>4 && dp.drafting) {
                const int cap=dp.n_max<0?arm-4:std::min(arm-4,dp.n_max);
                *dp.result=common_suffix_chain(*dp.prompt,dp.id_last,cap);
            }
            dp.drafting=false;
            return;
        }
        dp.n_max=dp.n_max<0?arm:std::min(arm,dp.n_max);
        // Exactly one MTP implementation; ordinary verification and hooks below.
    }
    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}
void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        if (spec->source_policy && spec->policy.active) {
            // Suffix drafts still advance the warm MTP carryover at the exact
            // accepted hidden row. Do not skip reconciliation on source switches.
            for (auto & other:spec->impls) other->accept(seq_id,n_accepted,true);
        } else {
            GGML_ASSERT(n_accepted == 0);
        }
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}
