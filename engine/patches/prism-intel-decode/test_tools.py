"""Model-free tests: actual packing analyzer, GGUF bounds, topology and cost gate."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import bitcos_cost as bitcos
import sandwich_plans as sandwich


def gguf(blocks, rows=1):
    name=b'test.weight'
    header=b'GGUF'+struct.pack('<IQQ',3,1,0)
    header+=struct.pack('<Q',len(name))+name+struct.pack('<IQQIQ',2,len(blocks)//rows*128,rows,143,0)
    return header+b'\0'*((-len(header))%32)+b''.join(blocks)


class Costs(unittest.TestCase):
    def test_encodings_and_bitcos_floor(self):
        for code in range(256):
            block=bytes([code])*26+b'\x00\x3c'
            digits=bitcos.trits(block)
            self.assertEqual(len(digits),128)
            self.assertTrue(set(digits)<={-1,0,1})
            # Independent iterative-byte oracle includes noncanonical encodings.
            state=code
            for j in range(5):
                self.assertEqual(digits[16*j],state*3//256-1)
                state=state*3%256
            size,z=bitcos.block_bytes(block)
            self.assertEqual(z,digits.count(0))
            self.assertEqual(size,18+(128-z+7)//8)
        self.assertEqual(bitcos.block_bytes(bytes([128])*26+b'\x00\x3c'),(18,128))
        self.assertEqual(bitcos.block_bytes(bytes(26)+b'\x00\x3c'),(34,0))

    def test_exact_vs_sample_gate(self):
        blocks=[bytes([128])*26+b'\x00\x3c']*4
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            p=Path(temp)/'test.gguf'; p.write_bytes(gguf(blocks,2))
            exact=bitcos.analyze(p)['tensors'][0]
            self.assertEqual(exact['bitcos_bytes_floor'],4*18+2*3*4)
            self.assertTrue(exact['candidate_for_target_benchmark'])
            sampled=bitcos.analyze(p,1)['tensors'][0]
            self.assertFalse(sampled['exact'])
            self.assertFalse(sampled['candidate_for_target_benchmark'])
            self.assertEqual(p.read_bytes(),gguf(blocks,2))
            p.write_bytes(gguf([bytes(26)+b'\x00\x3c']*4,2))
            self.assertFalse(bitcos.analyze(p)['tensors'][0]['candidate_for_target_benchmark'])

    def test_reader_rejects_truncation_and_lengths(self):
        data=gguf([bytes(28)])
        for end in (0,4,8,20,35,len(data)-1):
            with self.assertRaises((ValueError,UnicodeError)):
                bitcos.inventory(io.BytesIO(data[:end]))
        bad=data[:24]+struct.pack('<Q',2**63)+data[32:]
        with self.assertRaises(ValueError): bitcos.inventory(io.BytesIO(bad))


class Plans(unittest.TestCase):
    def setUp(self):
        self.topology={'schema_version':1,'cpus':[
            {'cpu':0,'core':0,'package':0,'core_type':'2'},
            {'cpu':1,'core':0,'package':0,'core_type':'2'},
            {'cpu':2,'core':1,'package':0,'core_type':'2'},
            {'cpu':3,'core':1,'package':0,'core_type':'2'},
            {'cpu':4,'core':2,'package':0,'core_type':'1'},
            {'cpu':7,'core':3,'package':0,'core_type':'1'}]}
        self.key={'model_sha256':'model','head_sha256':'head','runtime_build':'build',
            'kernel_flags':'exact-avx2','kv_type':'q4_0','context_bucket':8192,
            'sampler':'greedy','mtp_policy':'fixed4','workload_sha256':'same-inputs-output-budget'}

    def test_topology_not_power_of_two_and_no_smt_double_count(self):
        c=sandwich.candidates(self.topology)
        self.assertEqual(c[0]['cpus'],[0,2,4,7])
        self.assertIn([4,7],[x['cpus'] for x in c])
        for x in c:
            self.assertEqual(int(x['mask'],16),sum(1<<i for i in x['cpus']))
        cache_topology=copy.deepcopy(self.topology)
        for r in cache_topology['cpus']:
            r['core_type']='unknown'
            r['caches']=[{'level':'2','shared':[0,1,2,3]}] if r['cpu']<4 else []
        self.assertIn([0,2],[x['cpus'] for x in sandwich.candidates(cache_topology)])

    def test_bounded_stable_and_key_invalidates(self):
        a=sandwich.render(self.topology,self.key)
        self.assertEqual(a,sandwich.render(self.topology,self.key))
        self.assertLessEqual(len(a['plans']),16)
        self.assertFalse(a['launch'])
        changed=copy.deepcopy(self.key); changed['runtime_build']='new'
        self.assertNotEqual(a['plans'][0]['id'],sandwich.render(self.topology,changed)['plans'][0]['id'])
        for p in a['plans']:
            self.assertNotIn('--start',p['argv'])
            self.assertEqual(p['environment']['LLAMA_SANDWICH_VERIFY_THREADS'],str(p['phases']['verify']['threads']))

    def test_cost_gate_rejects_noise_wrongness_and_wrong_key(self):
        p=sandwich.render(self.topology,self.key)
        a,b=p['plans'][:2]
        def result(plan,samples,correct=True):
            return {'id':plan['id'],'cost_key':self.key,'whole_task_us':samples,
                    'correctness_passed':correct,'memory_passed':True}
        baseline=result(a,[100,102,104])
        self.assertEqual(sandwich.select(p,[baseline,result(b,[99,103,105])],a['id'])['selected'],a['id'])
        self.assertEqual(sandwich.select(p,[baseline,result(b,[90,91,92],False)],a['id'])['selected'],a['id'])
        self.assertEqual(sandwich.select(p,[baseline,result(b,[90,91,92])],a['id'])['selected'],b['id'])
        wrong=result(b,[90,91,92]); wrong['cost_key']={**self.key,'mtp_policy':'adaptive'}
        with self.assertRaises(ValueError): sandwich.select(p,[baseline,wrong],a['id'])

    def test_fixture_sysfs_obeys_allowed_and_online(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root=Path(temp); (root/'online').write_text('0-3,7')
            for cpu in (0,2,7):
                top=root/f'cpu{cpu}'/'topology'; top.mkdir(parents=True)
                (top/'core_id').write_text(str(cpu))
                (top/'physical_package_id').write_text('0')
                (top/'thread_siblings_list').write_text(str(cpu))
            found=sandwich.inspect_target(root,[0,2,7,9])
            self.assertEqual([c['cpu'] for c in found['cpus']],[0,2,7])
            self.assertTrue(all(c['core_type']=='unknown' for c in found['cpus']))


if __name__=='__main__': unittest.main()
