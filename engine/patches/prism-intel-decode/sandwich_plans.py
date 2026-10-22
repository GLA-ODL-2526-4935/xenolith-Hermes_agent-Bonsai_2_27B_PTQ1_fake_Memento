"""Read-only Intel/Linux topology and bounded dry-run phase plans. Never starts inference.

Supply a saved topology JSON, or explicitly inspect the Linux deployment target.
P/E labels are accepted only from reported core_type; frequency is not core type.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics


def cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        if not part:
            continue
        bounds = part.split('-')
        if len(bounds) > 2:
            raise ValueError('Invalid CPU list')
        a, b = int(bounds[0]), int(bounds[-1])
        if a < 0 or b < a or b > 4095:
            raise ValueError('Invalid CPU range')
        result.update(range(a,b+1))
    return sorted(result)


def inspect_target(sysfs=Path('/sys/devices/system/cpu'), allowed=None):
    if allowed is None:
        if platform.system() != 'Linux':
            raise ValueError('Inspect the Linux target, not the preparation PC')
        allowed = sorted(os.sched_getaffinity(0))
    allowed = set(allowed)
    online = set(cpu_list((sysfs/'online').read_text()))
    records = []
    for cpu in sorted(allowed & online):
        base = sysfs/f'cpu{cpu}'
        def value(name, fallback=None):
            p = base/name
            return p.read_text().strip() if p.exists() else fallback
        core = value('topology/core_id')
        package = value('topology/physical_package_id')
        if core is None or package is None:
            raise ValueError('Missing core/package topology')
        caches = []
        for p in sorted((base/'cache').glob('index*')):
            caches.append({'level':(p/'level').read_text().strip(),
                           'shared':cpu_list((p/'shared_cpu_list').read_text())})
        records.append({'cpu':cpu,'core':int(core),'package':int(package),
            'core_type':value('topology/core_type','unknown'),
            'siblings':cpu_list(value('topology/thread_siblings_list',str(cpu))),
            'caches':caches})
    return {'schema_version':1,'source':'Linux sysfs + process affinity',
            'cpus':records,'isa':'unknown; CPUID validation required',
            'ram_reserve':'target decision pending; no safeguard changed'}


def mask(cpus):
    return format(sum(1 << c for c in cpus),'x')


def candidates(topology):
    records = topology['cpus']
    if not records or len(records)>64:
        raise ValueError('Expected 1..64 available logical CPUs')
    ids = [r['cpu'] for r in records]
    if len(ids)!=len(set(ids)) or any(type(c) is not int or not 0<=c<512 for c in ids):
        raise ValueError('Invalid/duplicate CPU IDs for pinned 512-bit masks')
    core_map = {}
    for r in records:
        core_map.setdefault((r['package'],r['core']),[]).append(r)
    physical = sorted(min(v,key=lambda r:r['cpu'])['cpu'] for v in core_map.values())
    types = {}
    for v in core_map.values():
        # Inconsistent sibling classification is unknown, never silently P/E.
        labels = {str(r['core_type']) for r in v}
        label = next(iter(labels)) if len(labels)==1 else 'unknown'
        types.setdefault(label,[]).append(min(r['cpu'] for r in v))
    sets = [('physical',physical),('logical',sorted(ids))]
    for label, members in sorted(types.items()):
        if label!='unknown':
            sets.append(('type-'+label,sorted(members)))
    shared_groups=set()
    for r in records:
        for cache in r.get('caches',[]):
            if str(cache['level']) not in {'2','3'} or r['cpu'] not in cache['shared']:
                continue
            members=tuple(c for c in physical if c in cache['shared'])
            if 1<len(members)<len(physical):
                shared_groups.add((str(cache['level']),members))
    for level,members in sorted(shared_groups,key=lambda x:(x[0],len(x[1]),x[1])):
        sets.append(('L'+level+'-'+mask(members),list(members)))
    if len(physical)>4:
        sets.append(('physical-first4',physical[:4]))
    unique=[]
    for label, cpus in sets:
        if cpus and cpus not in [c['cpus'] for c in unique]:
            unique.append({'id':label,'cpus':cpus,'threads':len(cpus),'mask':mask(cpus)})
    return unique


def render(topology, key, limit=16):
    required={'model_sha256','head_sha256','runtime_build','kernel_flags','kv_type','context_bucket','sampler','mtp_policy','workload_sha256'}
    if set(key)!=required or any(v is None or v=='' for v in key.values()):
        raise ValueError('Cost key must identify model/head/runtime/kernel/KV/context/sampler/MTP policy')
    if type(limit) is not int or not 1<=limit<=32:
        raise ValueError('Bound the plan search to 1..32 plans')
    choices=candidates(topology)
    plans=[]
    # Same-set plans first, then balanced heterogeneous combinations. No model is
    # duplicated, no thread/affinity choice is called fastest without measurement.
    triples=[(c,c,c) for c in choices]
    # Cover different decode/core-class choices before a Cartesian product can
    # consume the budget tuning only the first decode choice.
    for prefill in choices[:2]:
        for decode in choices:
            triples.extend([(decode,prefill,decode),(decode,prefill,choices[0])])
    triples+=list(itertools.product(choices,repeat=3))
    seen=set()
    for decode,prefill,verify in triples:
        identity=(decode['mask'],prefill['mask'],verify['mask'])
        if identity in seen: continue
        seen.add(identity)
        config={'decode':decode,'prefill':prefill,'verify':verify}
        digest=hashlib.sha256(json.dumps({'key':key,'topology':topology,'config':config},sort_keys=True).encode()).hexdigest()
        plans.append({'id':digest,'phases':config,'argv':['-t',str(decode['threads']),'-tb',str(prefill['threads']),
            '--cpu-mask',decode['mask'],'--cpu-mask-batch',prefill['mask'],'--cpu-strict','1','--cpu-strict-batch','1',
            '--poll','0','--poll-batch','0'],
            'environment':{'LLAMA_SANDWICH_VERIFY_THREADS':str(verify['threads']),'LLAMA_SANDWICH_VERIFY_MASK':verify['mask']}})
        if len(plans)>=limit: break
    return {'schema_version':1,'cost_key':key,'topology':topology,'plans':plans,'launch':False,
            'activation':'Requires patched runtime, explicit target acceptance and owner launch; no profiles changed.'}


def select(plans, results, baseline):
    known={p['id'] for p in plans['plans']}
    if baseline not in known:
        raise ValueError('Baseline must be a rendered plan')
    groups={}
    for r in results:
        if r['id'] not in known or r['cost_key']!=plans['cost_key']:
            raise ValueError('Result is not from this fixed cost key/plan set')
        samples=r['whole_task_us']
        if r['id'] in groups or len(samples)<3 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
            raise ValueError('Need distinct plans with at least 3 finite positive whole-task samples')
        if r.get('correctness_passed') is not True or r.get('memory_passed') is not True:
            continue
        groups[r['id']]=samples
    if baseline not in groups:
        raise ValueError('No passing baseline measurements')
    # Conservative no-noise-overlap gate. MTP policy is fixed in the key: this
    # controller never competes with a second online adaptive controller.
    eligible=[p for p,s in groups.items() if p!=baseline and max(s)<min(groups[baseline])]
    best=min(eligible,key=lambda p:statistics.median(groups[p])) if eligible else baseline
    return {'selected':best,'baseline':baseline,'applied':False,
            'reason':'repeatable whole-task improvement' if best!=baseline else 'retain baseline; no clear passing improvement'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect-target',action='store_true')
    group.add_argument('--topology',type=Path)
    p.add_argument('--cost-key',type=Path)
    p.add_argument('--limit',type=int,default=16)
    p.add_argument('--results',type=Path,help='Measured result array; only selects a plan, never applies it')
    p.add_argument('--baseline',help='Baseline plan ID, required with --results')
    args=p.parse_args()
    topology=inspect_target() if args.inspect_target else json.loads(args.topology.read_text())
    if bool(args.results)!=bool(args.baseline) or (args.results and not args.cost_key):
        p.error('--results requires --baseline and --cost-key')
    result=render(topology,json.loads(args.cost_key.read_text()),args.limit) if args.cost_key else topology
    if args.results:
        result=select(result,json.loads(args.results.read_text()),args.baseline)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
