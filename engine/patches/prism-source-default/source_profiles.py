"""Portable selector counters, not model weights. No inference or downloads."""
import argparse, hashlib, json, os, platform
from pathlib import Path

FORMAT='hermes-source-profile-v2'
ALGORITHM='hermes-source-v2'
HASH_FIELDS=('weights_sha256','tokenizer_artifact_sha256','head_artifact_sha256',
             'runtime_sha256','hardware_sha256','settings_sha256','workload_sha256')
MAX_COUNT=10**12
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def runtime_digest(path):
    """Bind adjacent runtime libraries too; llama-server.exe can be a small loader."""
    path=Path(path)
    files={path.name:path}
    for pattern in ('*.dll','*.so','*.so.*','*.dylib'):
        for item in path.parent.glob(pattern):
            if item.is_file():files[item.name]=item
    return digest({name:file_digest(item) for name,item in sorted(files.items())})
def unique(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result
def read(path):
    with Path(path).open('rb') as f:data=f.read(262145)
    if len(data)>262144:raise ValueError('profile/identity exceeds 256 KiB')
    return json.loads(data.decode('utf-8'),object_pairs_hook=unique,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def identity(value):
    if type(value) is not dict or set(value)!={'algorithm',*HASH_FIELDS} or value['algorithm']!=ALGORITHM:
        raise ValueError('invalid selector identity schema')
    for name in HASH_FIELDS:
        v=value[name]
        if type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('invalid identity digest: '+name)
    return value
def count(value):
    if type(value) is not int or not 0<=value<=MAX_COUNT:raise ValueError('invalid profile counter')
    return value
def validate(data,expected=None):
    if type(data) is not dict or set(data)!={'format','identity','buckets'} or data['format']!=FORMAT:
        raise ValueError('invalid selector profile schema')
    identity(data['identity'])
    if expected is not None and data['identity']!=identity(expected):raise ValueError('incompatible selector identity; recalibrate')
    if type(data['buckets']) is not list or len(data['buckets'])>64:raise ValueError('invalid context buckets')
    seen=set()
    for b in data['buckets']:
        if type(b) is not dict or set(b)!={'context_bucket','cycles','arms'}:raise ValueError('invalid bucket schema')
        i=count(b['context_bucket'])
        if i>=64 or i in seen:raise ValueError('duplicate/invalid context bucket')
        seen.add(i);cycles=count(b['cycles'])
        if type(b['arms']) is not list or len(b['arms'])!=9:raise ValueError('nine arms required')
        total=0
        for arm in b['arms']:
            if type(arm) is not dict or set(arm)!={'tokens','elapsed_us','samples'}:raise ValueError('invalid arm schema')
            t,u,s=map(count,(arm['tokens'],arm['elapsed_us'],arm['samples']))
            if (s==0 and (t or u)) or (s and (t<s or u<s or t>6*s)):raise ValueError('inconsistent arm counters')
            total+=s
        if total!=cycles:raise ValueError('inconsistent cycle total')
    return data
def hardware():
    result={'system':platform.system(),'release':platform.release(),'machine':platform.machine(),
            'processor':platform.processor(),'logical_cpus':os.cpu_count()}
    if platform.system()=='Linux':
        cpu=Path('/proc/cpuinfo').read_text().split('\n\n')[0]
        result['cpu']={k.strip():v.strip() for line in cpu.splitlines() if ':' in line
                       for k,v in [line.split(':',1)] if k.strip() in {'vendor_id','model name','cpu family','model','stepping','flags'}}
        result['drm']=[]
        for dev in sorted(Path('/sys/class/drm').glob('renderD*/device')):
            entry={}
            for name in ('vendor','device','revision'):
                f=dev/name
                if f.exists():entry[name]=f.read_text().strip()
            driver=dev/'driver'
            if driver.exists():entry['driver']=driver.resolve().name
            result['drm'].append(entry)
        result['memory_kib']=next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:'))
    elif platform.system()=='Windows':
        # Registry reads only. No GPU utility or runtime is initialized.
        import winreg
        result['display_drivers']=[]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\Video') as video:
            for i in range(winreg.QueryInfoKey(video)[0]):
                name=winreg.EnumKey(video,i)
                try:
                    with winreg.OpenKey(video,name+r'\0000') as adapter:
                        fields={}
                        for field in ('DriverDesc','DriverVersion','ProviderName','MatchingDeviceId'):
                            try:
                                v=winreg.QueryValueEx(adapter,field)[0]
                                if isinstance(v,str):fields[field]=v
                            except FileNotFoundError:pass
                        if fields:result['display_drivers'].append(fields)
                except FileNotFoundError:pass
        result['display_drivers'].sort(key=lambda x:canonical(x))
    else:raise ValueError('hardware identity supports Linux/Windows only')
    return result
def check_runtime(expected,engine,settings,workload=None):
    expected=identity(expected)
    checks={'runtime_sha256':runtime_digest(engine),'hardware_sha256':digest(hardware()),'settings_sha256':digest(settings)}
    if workload is not None:checks['workload_sha256']=file_digest(workload)
    for name,value in checks.items():
        if expected[name]!=value:raise ValueError('selector provenance mismatch: '+name)
def write_new(path,value):
    path=Path(path)
    if path.exists() or Path(str(path)+'.lock').exists():raise ValueError('output exists or is owned')
    # Exclusive create: a concurrent writer can never be overwritten.
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def settings_for_plan(profile,plan):
    keys=('device','context','threads','threads_batch','batch','ubatch','kv_type','reasoning_effort',
          'max_output_tokens','mtp_tokens','mtp_probability','cpu_mask','cpu_mask_batch','poll','poll_batch',
          'prefix_cache','cache_ram_mib','draft_backend_sampling','suffix_drafting')
    return {'profile':{k:profile[k] for k in keys},'request_defaults':plan['request_defaults'],
            'performance_environment_sha256':performance_environment()}
def performance_environment():
    ignore=('LLAMA_ARG_','LLAMA_SPEC_PROFILE_','LLAMA_SPEC_SOURCE_')
    values={k:v for k,v in os.environ.items() if
            (k.startswith(('GGML_','LLAMA_','OMP_','MKL_','OPENBLAS_')) or k in {'CUDA_VISIBLE_DEVICES','VK_ICD_FILENAMES','ZE_AFFINITY_MASK'})
            and not k.startswith(ignore) and k not in {'LLAMA_SPEC_IDENTITY','LLAMA_SPEC_SUFFIX'}}
    return digest(values)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('identity');make.add_argument('--runtime',required=True);make.add_argument('--settings',required=True)
    make.add_argument('--workload',required=True);make.add_argument('--weights-sha256',required=True)
    make.add_argument('--out',required=True)
    for command in ('validate','copy','retarget'):
        x=sub.add_parser(command);x.add_argument('--profile',required=True);x.add_argument('--identity',required=True)
        if command!='validate':x.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.command=='identity':
        settings=read(a.settings)
        if isinstance(settings,dict) and 'selector_settings' in settings:settings=settings['selector_settings']
        data={'algorithm':ALGORITHM,**{k:a.weights_sha256 for k in ('weights_sha256','tokenizer_artifact_sha256','head_artifact_sha256')},
              'runtime_sha256':runtime_digest(a.runtime),'hardware_sha256':digest(hardware()),
              'settings_sha256':digest(settings),'workload_sha256':file_digest(a.workload)}
        write_new(a.out,identity(data));print('Identity created; no model opened');return
    expected=identity(read(a.identity));data=validate(read(a.profile))
    if a.command=='retarget':
        for k in ('weights_sha256','tokenizer_artifact_sha256','head_artifact_sha256'):
            if data['identity'][k]!=expected[k]:raise ValueError('retarget requires same model/tokenizer/head artifacts')
        write_new(a.out,{'format':FORMAT,'identity':expected,'buckets':[]})
        print('Fresh target profile created; foreign timing counters were cleared');return
    validate(data,expected)
    if a.command=='copy':write_new(a.out,data)
    print('Compatible profile; counters='+str(sum(b['cycles'] for b in data['buckets'])))
if __name__=='__main__':main()
