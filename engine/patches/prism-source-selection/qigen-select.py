"""Choose a conservative explicit five-column schedule from offline target measurements.

Input JSON: {"target_key": "hardware/build/thread/flags signature", "cases": [
  {"columns": 1, "mode": 1, "correct": true, "times_us": [10,11,10]}, ...]}
Modes 1=dot, 2=batch (2..5 cols), 3=LUT tile8, 4=LUT tile16.
Never runs a benchmark or changes environment/profile. Requires repeated baseline
and candidate timings on the same shape, thread, cache and thermal conditions.
"""
import argparse,json,math,statistics
def select(data):
    if not isinstance(data.get('target_key'),str) or not data['target_key'].strip():
        raise ValueError('nonempty target_key required')
    cases={}
    for c in data['cases']:
        col,mode=c['columns'],c['mode'];times=c['times_us']
        if col not in range(1,6) or mode not in range(1,5) or (col==1 and mode==2):
            raise ValueError('unsupported shape/mode')
        if (col,mode) in cases: raise ValueError('duplicate shape/mode')
        if c.get('correct') is not True: continue
        if len(times)<5 or any(not math.isfinite(v) or v<=0 for v in times):
            raise ValueError('at least five positive finite measurements required')
        cases[col,mode]=times
    schedule=[]
    for col in range(1,6):
        if (col,1) not in cases: raise ValueError('baseline missing for column '+str(col))
        base=cases[col,1];winner=1;best=statistics.median(base)
        for mode in (2,3,4):
            if (col,mode) not in cases: continue
            times=cases[col,mode];median=statistics.median(times)
            # Demand >=5% median gain and no p90 regression. Conservative,
            # empirical engineering rule, not a QIGen/theoretical speed promise.
            p90=lambda a:sorted(a)[math.ceil(0.9*len(a))-1]
            if median<statistics.median(base)*0.95 and median<best and p90(times)<=p90(base): winner=mode;best=median
        schedule.append(str(winner))
    return {'target_key':data['target_key'],'GGML_PTQ1_QIGEN':'1',
            'GGML_PTQ1_QIGEN_SCHEDULE':''.join(schedule),'activated':False}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('measurements')
    a=p.parse_args()
    print(json.dumps(select(json.loads(open(a.measurements,encoding='utf-8').read())),indent=2))
