import csv
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT.parent

def load(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def validate_access(path=ROOT/'output'/'SCHEDULE_ACCESS.csv'):
    acts={r['activity_id']:r for r in load(DATA/'08_ACTIVITY_DETAILS.csv')}
    projs={r['contract_number']:r for r in load(DATA/'07_PROJECT_DETAILS.csv')}
    sched=load(path); errors=[]; by_a=defaultdict(list)
    for r in sched: by_a[r['activity_id']].append(r)
    # Workload conservation: sum(1 + .5*ECLO) >= total_accesses
    for aid,a in acts.items():
        y=sum(1+0.5*int(r['eclo']) for r in by_a[aid])
        if y < int(a['total_accesses']): errors.append(f'workload {aid}: {y} < {a["total_accesses"]}')
    # Weekly allocation and workfronts.
    nights=defaultdict(set); fronts=defaultdict(set)
    for r in sched:
        a=acts[r['activity_id']]; c=a['contract_number']; w=int(r['week']); n=int(r['access_night'])
        nights[c,w].add(n); fronts[c,w,n].add(r['activity_id'])
    for (c,w),ns in nights.items():
        if len(ns)>int(projs[c]['number_of_maximum_access_per_week']): errors.append(f'weekly allocation {c} wk{w}')
    for (c,w,n),aa in fronts.items():
        if len(aa)>int(projs[c]['number_of_workfronts']): errors.append(f'workfront {c} wk{w} n{n}')
    # Predecessor strict later-week rule.
    for aid,a in acts.items():
        pred=a['predecessor_activity_id']
        if pred and by_a[aid] and by_a[pred]:
            if min(int(r['week']) for r in by_a[aid]) <= max(int(r['week']) for r in by_a[pred]): errors.append(f'predecessor {pred}->{aid}')
    return errors

if __name__=='__main__':
    e=validate_access(); print('PASS' if not e else 'FAIL'); [print('-',x) for x in e]
