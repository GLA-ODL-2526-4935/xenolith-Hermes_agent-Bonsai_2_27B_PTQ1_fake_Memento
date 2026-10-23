// Private Prism/Xenolith profile adapter. Upstream llama.cpp MIT license applies.
#pragma once
#include "source-policy.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct common_source_profiles {
    using json = nlohmann::json;
    struct bucket { bool present=false; common_source_policy value; };
    std::array<bucket,64> buckets{};
    json identity;
    std::filesystem::path output, lock;
    bool dirty=false, owns_lock=false, frozen=false;

    static json read(const std::filesystem::path & path) {
        if (std::filesystem::file_size(path)>262144) throw std::runtime_error("source profile too large");
        std::ifstream in(path, std::ios::binary);
        std::string text(262145,'\0');in.read(text.data(),(std::streamsize)text.size());text.resize((size_t)in.gcount());
        if (text.size()>262144) throw std::runtime_error("source profile too large");
        std::map<int,std::set<std::string>> keys;
        auto callback=[&](int depth,json::parse_event_t event,json & value) {
            if (event==json::parse_event_t::object_start) keys[depth+1].clear();
            if (event==json::parse_event_t::key && !keys[depth].insert(value.get<std::string>()).second)
                throw std::runtime_error("duplicate source profile key");
            return true;
        };
        return json::parse(text,callback);
    }
    static bool hash(const json & value) {
        if (!value.is_string()) return false;
        const auto s=value.get<std::string>();
        return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
    }
    static void validate_identity(const json & value) {
        const char * names[]={"weights_sha256","tokenizer_artifact_sha256","head_artifact_sha256",
            "runtime_sha256","hardware_sha256","settings_sha256","workload_sha256"};
        if (!value.is_object() || value.size()!=8 || value.value("algorithm","")!="hermes-source-v2")
            throw std::runtime_error("invalid source profile identity");
        for (auto name:names) if (!value.contains(name) || !hash(value[name]))
            throw std::runtime_error("invalid source profile identity hash");
    }
    static uint64_t count(const json & value) {
        if (!value.is_number_unsigned() && !value.is_number_integer()) throw std::runtime_error("invalid profile count");
        if (value.is_number_integer() && value.get<int64_t>()<0) throw std::runtime_error("negative profile count");
        const auto n=value.get<uint64_t>();
        if (n>1000000000000ULL) throw std::runtime_error("profile count exceeds bound");
        return n;
    }
    void load(const json & data) {
        if (!data.is_object() || data.size()!=3 || data.value("format","")!="hermes-source-profile-v2" ||
                data.at("identity")!=identity || !data.at("buckets").is_array() || data.at("buckets").size()>64)
            throw std::runtime_error("incompatible source profile");
        std::array<bucket,64> checked{};
        for (const auto & b:data.at("buckets")) {
            if (!b.is_object() || b.size()!=3 || !b.at("arms").is_array() || b.at("arms").size()!=9)
                throw std::runtime_error("invalid source profile bucket");
            const auto id=count(b.at("context_bucket"));
            if (id>=checked.size() || checked[id].present) throw std::runtime_error("duplicate/invalid context bucket");
            auto & dst=checked[id];dst.present=true;dst.value.context_bucket=(int)id;
            dst.value.cycles=count(b.at("cycles"));uint64_t sum=0;
            for (size_t a=0;a<9;++a) {
                const auto & s=b.at("arms")[a];
                if (!s.is_object() || s.size()!=3) throw std::runtime_error("invalid source profile arm");
                auto & t=dst.value.stats[a];
                t.tokens=count(s.at("tokens"));t.us=count(s.at("elapsed_us"));t.samples=count(s.at("samples"));
                if ((!t.samples && (t.tokens || t.us)) || (t.samples && (t.tokens<t.samples || t.us<t.samples || t.tokens>t.samples*6)))
                    throw std::runtime_error("inconsistent source profile counters");
                sum+=t.samples;
            }
            if (sum!=dst.value.cycles) throw std::runtime_error("inconsistent source profile cycles");
        }
        buckets=checked;
    }
    json snapshot() const {
        json data={{"format","hermes-source-profile-v2"},{"identity",identity},{"buckets",json::array()}};
        for (size_t id=0;id<buckets.size();++id) {
            if (!buckets[id].present) continue;
            const auto & v=buckets[id].value;json arms=json::array();
            for (const auto & s:v.stats) arms.push_back({{"tokens",s.tokens},{"elapsed_us",s.us},{"samples",s.samples}});
            data["buckets"].push_back({{"context_bucket",id},{"cycles",v.cycles},{"arms",arms}});
        }
        return data;
    }
    common_source_profiles(const std::string & expected,const std::string & input,const std::string & destination,bool freeze=false): frozen(freeze) {
        identity=read(expected);validate_identity(identity);
        try {
            if (!destination.empty()) {
                output=std::filesystem::absolute(destination);
                if (!std::filesystem::is_directory(output.parent_path())) throw std::runtime_error("profile parent missing");
                if (std::filesystem::exists(output) && (input.empty() || !std::filesystem::equivalent(output,input)))
                    throw std::runtime_error("profile output exists; resume it explicitly or choose a new file");
                lock=output;lock += ".lock";
                if (!std::filesystem::create_directory(lock)) throw std::runtime_error("source profile output already owned");
                owns_lock=true;
            }
            if (!input.empty()) load(read(input));
        } catch (...) { release();throw; }
    }
    common_source_profiles(const common_source_profiles &)=delete;
    common_source_profiles & operator=(const common_source_profiles &)=delete;
    ~common_source_profiles() { release(); }
    void release() noexcept {
        if (owns_lock) { std::error_code error;std::filesystem::remove(lock/"snapshot.tmp",error);std::filesystem::remove(lock,error);owns_lock=false; }
    }
    void seed(common_source_policy & policy,int context) const {
        policy.reset();policy.context_bucket=context;
        if (context>=0 && context<(int)buckets.size() && buckets[context].present) policy=buckets[context].value;
    }
    void observe(const common_source_policy & policy) {
        if (frozen) return;
        const int id=policy.context_bucket;
        if (id<0 || id>=(int)buckets.size()) return;
        auto & b=buckets[id];b.present=true;b.value=policy;b.value.active=false;b.value.started=0;dirty=true;
        // Bound long-running priors before the portable integer limit is reached.
        if (std::any_of(b.value.stats.begin(),b.value.stats.end(),[](const common_source_policy::stat & s) {
                return s.us>500000000000ULL || s.samples>50000000000ULL;
            })) {
            b.value.reset();b.value.context_bucket=id;
        }
    }
    void save() {
        if (!dirty || output.empty()) return;
        const auto temp=lock/"snapshot.tmp";
        const auto data=snapshot().dump(2)+"\n";
        { std::ofstream file(temp,std::ios::binary|std::ios::trunc);file.write(data.data(),(std::streamsize)data.size());file.flush();
          if (!file) throw std::runtime_error("source profile write failed"); }
#if defined(_WIN32)
        if (!MoveFileExW(temp.c_str(),output.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("source profile atomic replace failed");
#else
        std::filesystem::rename(temp,output);
#endif
        dirty=false;
    }
};
