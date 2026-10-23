"""Offline contracts only; never execute a model, native binary or tool call."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import source_profiles as p
import hermes_replay as replay
spec=importlib.util.spec_from_file_location('portable_backend',HERE/'backendcheck/engine/backends/prism.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class Profiles(unittest.TestCase):
    def setUp(self):
        self.identity={'algorithm':p.ALGORITHM,**{key:'a'*64 for key in p.HASH_FIELDS}}
        self.data={'format':p.FORMAT,'identity':self.identity,'buckets':[{'context_bucket':0,'cycles':2,
                 'arms':[{'tokens':2,'elapsed_us':300,'samples':2}]+[{'tokens':0,'elapsed_us':0,'samples':0} for _ in range(8)]}]}
    def test_roundtrip(self):self.assertEqual(p.validate(json.loads(json.dumps(self.data)),self.identity),self.data)
    def test_each_provenance_mismatch(self):
        for key in p.HASH_FIELDS:
            other={**self.identity,key:'b'*64}
            with self.assertRaises(ValueError):p.validate(self.data,other)
    def test_malformed_counters(self):
        for value in [-1,True,0.3,10**12+1]:
            bad=copy.deepcopy(self.data);bad['buckets'][0]['arms'][0]['samples']=value
            with self.assertRaises(ValueError):p.validate(bad)
    def test_cycles_and_tokens(self):
        for key,value in [('tokens',13),('elapsed_us',0),('samples',1)]:
            bad=copy.deepcopy(self.data);bad['buckets'][0]['arms'][0][key]=value
            with self.assertRaises(ValueError):p.validate(bad)
    def test_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'x.json';f.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):p.read(f)
        bad=copy.deepcopy(self.data);bad['buckets']*=2
        with self.assertRaises(ValueError):p.validate(bad)
    def test_exclusive_create_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'x.json';p.write_new(f,self.data)
            with self.assertRaises(ValueError):p.write_new(f,self.data)
            g=Path(tmp)/'g.json';Path(str(g)+'.lock').mkdir()
            with self.assertRaises(ValueError):p.write_new(g,self.data)
    def test_copy_and_retarget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'s.json';ident=root/'i.json';out=root/'o.json'
            p.write_new(source,self.data);p.write_new(ident,self.identity)
            p.main(['copy','--profile',str(source),'--identity',str(ident),'--out',str(out)])
            self.assertEqual(p.read(out),self.data)
            new={**self.identity,'hardware_sha256':'b'*64};target=root/'t.json';p.write_new(target,new)
            dest=root/'new.json';p.main(['retarget','--profile',str(source),'--identity',str(target),'--out',str(dest)])
            self.assertEqual(p.read(dest)['buckets'],[])
    def test_runtime_guard(self):
        with patch.object(p,'file_digest',return_value='a'*64),patch.object(p,'hardware',return_value={}),patch.object(p,'digest',return_value='a'*64):
            p.check_runtime(self.identity,'/fake',{})
            with self.assertRaises(ValueError):p.check_runtime({**self.identity,'hardware_sha256':'b'*64},'/fake',{})

class Backend(unittest.TestCase):
    def setUp(self):
        self.c=b.load_profile(HERE/'backendcheck/engine/profiles/bonsai27-linux-cpu.json')
        self.c.update(source_policy=True,suffix_drafting=True,adaptive_mtp=False,mtp_tokens=4,
                      runtime_requirement='ff41412 + intel-source-profile-v2',policy_identity='/tmp/id.json',policy_output='/tmp/profile.json')
    def test_default_still_off(self):
        old=b.load_profile(HERE/'backendcheck/engine/profiles/bonsai27-linux-cpu.json')
        self.assertEqual(b.make_plan(old,'/fake','/'+b.MODEL)['environment'],{})
    def test_flags_and_identity_settings(self):
        plan=b.make_plan(self.c,'/fake','/'+b.MODEL)
        self.assertEqual(plan['environment']['LLAMA_SPEC_SOURCE_POLICY'],'1')
        self.assertEqual(plan['environment']['LLAMA_SPEC_PROFILE_OUT'],'/tmp/profile.json')
        self.assertTrue(plan['selector_settings']['profile']['suffix_drafting'])
    def test_old_tuner_conflict(self):
        self.c['adaptive_mtp']=True
        with self.assertRaises(ValueError):b.validate(self.c)
    def test_frozen(self):
        self.c.update(policy_frozen=True,policy_input='/tmp/input.json',policy_output='')
        self.assertEqual(b.make_plan(self.c,'/fake','/'+b.MODEL)['environment']['LLAMA_SPEC_PROFILE_FROZEN'],'1')
        self.c['policy_output']='/tmp/out.json'
        with self.assertRaises(ValueError):b.validate(self.c)
    def test_no_spawn_on_windows(self):
        with patch.object(b.platform,'system',return_value='Windows'),patch.object(b.subprocess,'Popen') as spawn:
            with self.assertRaises(RuntimeError):b.serve(b.make_plan(self.c,'/fake','/'+b.MODEL),self.c)
            spawn.assert_not_called()

class Workflows(unittest.TestCase):
    def test_split_oracles_and_no_leak(self):
        rows=list(map(json.loads,(HERE/'hermes-workflows.jsonl').read_text(encoding='utf-8').splitlines()))
        self.assertEqual(len(rows),72)
        groups={s:{r['workflow'] for r in rows if r['split']==s} for s in ('calibration','heldout')}
        self.assertFalse(groups['calibration'] & groups['heldout'])
        for row in rows:
            self.assertNotIn('oracle',json.loads(row['messages'][1]['content']))
            o=row['oracle']
            response={'steps':['read record','answer']} if o['phase']=='plan' else (
                {'type':'tool','tool':o['tool'],'arguments':o['arguments']} if o['phase']=='tool' else
                {'type':'finish','answer':' '.join(o['required'])})
            self.assertTrue(replay.score(row,json.dumps(response))['checks'])
            self.assertFalse(replay.score(row,'invalid JSON')['checks'])
if __name__=='__main__':unittest.main()
