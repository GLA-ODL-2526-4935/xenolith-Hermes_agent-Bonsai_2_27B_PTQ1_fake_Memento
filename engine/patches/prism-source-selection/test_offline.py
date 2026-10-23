"""Offline tool/contract tests; not native inference correctness tests."""
import importlib.util,json,math,unittest
from pathlib import Path
p=Path(__file__).parent
spec=importlib.util.spec_from_file_location('qigen_select',p/'qigen-select.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
def data():
    return {'target_key':'fixture-only:cpu/build/threads/cache','cases':[
        {'columns':c,'mode':1,'correct':True,'times_us':[100,100,100,100,100]} for c in range(1,6)]}
class Offline(unittest.TestCase):
    def test_baseline_only(self):
        result=tool.select(data());self.assertEqual(result['GGML_PTQ1_QIGEN_SCHEDULE'],'11111');self.assertFalse(result['activated'])
    def test_per_width_tuning(self):
        d=data()
        d['cases'] += [{'columns':c,'mode':4,'correct':True,'times_us':[70]*5} for c in range(2,6)]
        self.assertEqual(tool.select(d)['GGML_PTQ1_QIGEN_SCHEDULE'],'14444')
    def test_numerically_failed_candidate(self):
        d=data();d['cases'].append({'columns':1,'mode':3,'correct':False,'times_us':[1]*5})
        self.assertEqual(tool.select(d)['GGML_PTQ1_QIGEN_SCHEDULE'],'11111')
    def test_tail_latency_and_small_gain(self):
        d=data();d['cases'] += [{'columns':1,'mode':3,'correct':True,'times_us':[50,50,50,50,200]},
            {'columns':2,'mode':4,'correct':True,'times_us':[96]*5}]
        self.assertEqual(tool.select(d)['GGML_PTQ1_QIGEN_SCHEDULE'],'11111')
    def test_reject_invalid_measurements(self):
        for times in ([1]*4,[1,2,3,4,float('nan')],[1,2,3,4,float('inf')],[1,2,3,4,0]):
            d=data();d['cases'][0]['times_us']=times
            with self.assertRaises(ValueError):tool.select(d)
    def test_reject_ambiguous_input(self):
        for change in ('key','missing','duplicate','batch1'):
            d=data()
            if change=='key':d['target_key']=''
            if change=='missing':d['cases'].pop()
            if change=='duplicate':d['cases'].append(dict(d['cases'][0]))
            if change=='batch1':d['cases'][0]['mode']=2
            with self.assertRaises(ValueError):tool.select(d)
    def test_source_scope_and_lifecycle_contract(self):
        m=json.loads((p/'manifest.json').read_text())
        self.assertEqual(m['requires'],['intel-decode-v3','intel-decode-increment-v1'])
        patch=(p/'source-selection.patch').read_text(encoding='utf-8')
        self.assertNotIn('vulkan', '\n'.join(m['files']))
        self.assertIn('other->accept(seq_id,n_accepted,true)',patch)
        self.assertIn('configs[0].params.draft.adaptive=false',patch)
        self.assertIn('qigen_mode==3?8:16',patch)
        self.assertIn('slot.task->params.sampling.temp==0.0f',patch)
        self.assertIn('llama_synchronize(mtp->params.ctx_dft)',patch)
if __name__=='__main__':unittest.main()
