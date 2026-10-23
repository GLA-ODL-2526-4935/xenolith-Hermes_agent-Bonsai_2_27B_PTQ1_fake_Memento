// Offline native contract test for later execution on the target. No model.
#include "source-profile.h"
#include <cassert>
#include <chrono>

int main() {
    using profiles=common_source_profiles;
    using json=profiles::json;
    const auto dir=std::filesystem::temp_directory_path()/
        ("source-profile-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(std::filesystem::create_directory(dir));
    const auto ident=dir/"identity.json",out=dir/"profile.json";
    json identity={{"algorithm","hermes-source-v2"}};
    for (auto key:{"weights_sha256","tokenizer_artifact_sha256","head_artifact_sha256",
            "runtime_sha256","hardware_sha256","settings_sha256","workload_sha256"}) identity[key]=std::string(64,'a');
    { std::ofstream f(ident); f<<identity; }
    {
        profiles bank(ident.string(),"",out.string());common_source_policy p;
        bank.seed(p,0);p.begin(4,true,10,0);assert(p.arm==0);p.finish(1,40);bank.observe(p);bank.save();
        bool locked=false;
        try { profiles conflict(ident.string(),out.string(),out.string()); } catch (...) { locked=true; }
        assert(locked);
    }
    {
        profiles resumed(ident.string(),out.string(),out.string());common_source_policy p;
        resumed.seed(p,0);assert(p.stats[0].tokens==1 && p.stats[0].us==30 && p.cycles==1);
        const auto before=resumed.snapshot();auto bad=before;bad["buckets"][0]["cycles"]=7;
        bool rejected=false;try { resumed.load(bad); } catch (...) { rejected=true; }
        assert(rejected && resumed.snapshot()==before);
    }
    {
        profiles frozen(ident.string(),out.string(),"",true);common_source_policy p;
        frozen.seed(p,0);p.begin(4,true,10,0);p.finish(1,100);frozen.observe(p);assert(!frozen.dirty);
    }
    const auto duplicate=dir/"duplicate.json";
    { std::ofstream f(duplicate);f<<"{\"x\":1,\"x\":2}"; }
    bool rejected=false;try { profiles::read(duplicate); } catch (...) { rejected=true; }assert(rejected);
    common_source_policy choice;choice.cycles=2;choice.stats[0]={2,200,2};choice.stats[2]={6,200,2};
    assert(choice.choose(4,true,false)==2); // frozen mode never probes
    for (auto path:{ident,out,duplicate}) assert(std::filesystem::remove(path));
    assert(std::filesystem::remove(dir));
}
