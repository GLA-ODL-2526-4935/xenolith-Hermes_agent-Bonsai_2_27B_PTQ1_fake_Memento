"""Frozen-state replay only. Dry run by default; never launch a model or execute tools."""
import argparse,json,time,urllib.request,urllib.parse
from pathlib import Path

def score(row,content):
    try:v=json.loads(content)
    except (ValueError,TypeError):return {'format':False,'checks':False}
    o=row['oracle'];phase=o['phase'];ok=False
    if phase=='plan':ok=type(v) is dict and type(v.get('steps')) is list and 1<=len(v['steps'])<=4 and all(isinstance(s,str) and s.strip() for s in v['steps'])
    elif phase=='tool':ok=type(v) is dict and v.get('type')=='tool' and v.get('tool')==o['tool'] and v.get('arguments')==o['arguments']
    else:ok=type(v) is dict and v.get('type')=='finish' and isinstance(v.get('answer'),str) and all(s.lower() in v['answer'].lower() for s in o['required'])
    return {'format':True,'checks':ok}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload',type=Path,default=Path(__file__).with_name('hermes-workflows.jsonl'))
    p.add_argument('--split',choices=['calibration','heldout'],required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8080/v1/chat/completions')
    p.add_argument('--plan',type=Path,required=True,help='Exact recorded backend dry-run plan for the running owned server')
    p.add_argument('--mode',choices=['baseline','learn','frozen'],required=True)
    p.add_argument('--out',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    url=urllib.parse.urlparse(a.endpoint)
    if url.scheme!='http' or url.hostname!='127.0.0.1' or url.username or url.password or url.query or url.fragment or url.path!='/v1/chat/completions':raise ValueError('Use an explicit loopback chat-completions endpoint')
    rows=[json.loads(s) for s in a.workload.read_text(encoding='utf-8').splitlines() if s.strip()]
    selected=[r for r in rows if r['split']==a.split]
    plan=json.loads(a.plan.read_text(encoding='utf-8'));defaults=plan['request_defaults'];env=plan['environment']
    if a.endpoint!=plan['endpoint'].rstrip('/')+'/chat/completions':raise ValueError('Endpoint must match recorded server plan')
    if a.mode=='baseline':
        if env.get('LLAMA_SPEC_SOURCE_POLICY')=='1' or plan['argv'][plan['argv'].index('--spec-type')+1]!='none':raise ValueError('Baseline requires no MTP and no selector')
    else:
        if env.get('LLAMA_SPEC_SOURCE_POLICY')!='1' or not env.get('LLAMA_SPEC_IDENTITY'):raise ValueError('Portable selector plan required')
        if a.mode=='learn' and (a.split!='calibration' or not env.get('LLAMA_SPEC_PROFILE_OUT') or env.get('LLAMA_SPEC_PROFILE_FROZEN')=='1'):raise ValueError('Learn only on calibration with profile output')
        if a.mode=='frozen' and (a.split!='heldout' or env.get('LLAMA_SPEC_PROFILE_FROZEN')!='1' or not env.get('LLAMA_SPEC_PROFILE_IN') or env.get('LLAMA_SPEC_PROFILE_OUT')):raise ValueError('Held-out selector evaluation requires frozen input only')
    if defaults.get('temperature')!=0 or defaults.get('top_k')!=1 or not 1<=defaults.get('max_tokens',0)<=4096:raise ValueError('Greedy defaults and 1..4096 output tokens required')
    if any(k in defaults for k in ('messages','tools','stream','n')):raise ValueError('Do not override workload messages or enable tool execution/streaming')
    if not a.run:
        print(json.dumps({'mode':'dry-run','selector_mode':a.mode,'split':a.split,'requests':len(selected),'endpoint':a.endpoint,'tools_executed':False}));return
    if not a.out:raise ValueError('--out required for explicit run')
    if a.mode!='baseline':
        import source_profiles as profiles
        expected=profiles.identity(profiles.read(env['LLAMA_SPEC_IDENTITY']))
        profiles.check_runtime(expected,plan['argv'][0],plan['selector_settings'],a.workload)
        if expected['weights_sha256']!=plan['required_model_sha256']:raise ValueError('Wrong model identity')
        if env.get('LLAMA_SPEC_PROFILE_IN'):profiles.validate(profiles.read(env['LLAMA_SPEC_PROFILE_IN']),expected)
    baseline={}
    if a.baseline:
        baseline={r['id']:r for r in map(json.loads,a.baseline.read_text(encoding='utf-8').splitlines())}
    # Exclusive output: preserve earlier results. Completed records are flushed request by request.
    with a.out.open('x',encoding='utf-8') as out:
        for row in selected:
            body={**defaults,'messages':row['messages'],'stream':False}
            req=urllib.request.Request(a.endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            started=time.perf_counter()
            with urllib.request.urlopen(req,timeout=900) as response:result=json.load(response)
            elapsed=time.perf_counter()-started;choice=result['choices'][0];message=choice['message']
            content=message.get('content') or ''
            record={'id':row['id'],'split':a.split,'elapsed_s':elapsed,'finish_reason':choice.get('finish_reason'),
                    'content':content,'usage':result.get('usage',{}),'timings':result.get('timings',{}),
                    'score':score(row,content),'token_capped':choice.get('finish_reason')=='length'}
            if row['id'] in baseline:record['exact_baseline_agreement']=content==baseline[row['id']]['content']
            out.write(json.dumps(record)+'\n');out.flush()
            print(row['id'],record['finish_reason'],round(elapsed,3),flush=True)
if __name__=='__main__':main()
