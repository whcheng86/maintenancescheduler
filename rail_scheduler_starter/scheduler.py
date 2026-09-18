from __future__ import annotations
import csv, math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from collections import defaultdict
from ortools.sat.python import cp_model

DATA = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(exist_ok=True)

def rows(name):
    with open(DATA/name, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

@dataclass
class Activity:
    aid:str; contract:str; total:int; start_week:int; priority:int; pred:str|None

class RailModel:
    def __init__(self, scenario='A', time_limit=30):
        self.scenario=scenario.upper(); self.time_limit=time_limit
        self.params={r['key']:r['value'] for r in rows('06_PARAMETERS.csv')}
        self.h0=date.fromisoformat(self.params['horizon_start']); self.W=int(self.params['horizon_weeks'])
        self.projects={r['contract_number']:r for r in rows('07_PROJECT_DETAILS.csv')}
        self.raw_acts=rows('08_ACTIVITY_DETAILS.csv')
        self.acts={}
        for r in self.raw_acts:
            sw=max(1,(date.fromisoformat(r['planned_start_date'])-self.h0).days//7+1)
            self.acts[r['activity_id']]=Activity(r['activity_id'],r['contract_number'],int(r['total_accesses']),sw,int(r['activity_priority']),r['predecessor_activity_id'] or None)
        self.model=cp_model.CpModel(); self.x={}; self.e={}; self.last={}

    def build(self):
        # Baseline temporal/access model. Spatial possession constraints are the next module.
        for a in self.acts.values():
            maxn=int(self.projects[a.contract]['number_of_maximum_access_per_week'])
            for w in range(a.start_week,self.W+1):
                for n in range(1,maxn+1):
                    self.x[a.aid,w,n]=self.model.NewBoolVar(f'x_{a.aid}_{w}_{n}')
                    if self.scenario in ('B','C'):
                        self.e[a.aid,w,n]=self.model.NewBoolVar(f'e_{a.aid}_{w}_{n}')
                        self.model.Add(self.e[a.aid,w,n] <= self.x[a.aid,w,n])
            self.last[a.aid]=self.model.NewIntVar(a.start_week,self.W,f'last_{a.aid}')
            for (aid,w,n),v in self.x.items():
                if aid==a.aid: self.model.Add(self.last[aid] >= w).OnlyEnforceIf(v)
            # Workload: scaled by 2 so CP-SAT stays integer. Standard=2, ECLO=3.
            std=sum(v for (aid,w,n),v in self.x.items() if aid==a.aid)
            ec=sum(v for (aid,w,n),v in self.e.items() if aid==a.aid) if self.scenario in ('B','C') else 0
            self.model.Add(2*std + ec >= 2*a.total)

        # Workfront: max distinct activities of contract+type on a local access-night.
        for c,p in self.projects.items():
            wf=int(p['number_of_workfronts']); maxn=int(p['number_of_maximum_access_per_week'])
            aids=[a.aid for a in self.acts.values() if a.contract==c]
            for w in range(1,self.W+1):
                for n in range(1,maxn+1):
                    vs=[self.x[a,w,n] for a in aids if (a,w,n) in self.x]
                    if vs: self.model.Add(sum(vs)<=wf)

        # Finish-to-start predecessor, zero lag: successor starts in a strictly later week.
        for a in self.acts.values():
            if a.pred:
                for (aid,w,n),v in self.x.items():
                    if aid==a.aid:
                        self.model.Add(w >= self.last[a.pred] + 1).OnlyEnforceIf(v)

        # Scenario A forbids ECLO by construction. B requires completion by planned date.
        if self.scenario=='B':
            for c,p in self.projects.items():
                due=min(self.W, max(1,(date.fromisoformat(p['planned_completion_date'])-self.h0).days//7+1))
                for a in self.acts.values():
                    if a.contract==c: self.model.Add(self.last[a.aid] <= due)

        # Baseline objective: weighted activity completion week; approximates urgency until
        # exact contract overrun/capacity terms are added with the spatial model.
        terms=[]
        cw={1:100,2:10,3:1}; nudge={1:13,2:12,3:10} # /10 scaling
        for a in self.acts.values():
            cp=int(self.projects[a.contract]['contract_priority'])
            terms.append(cw[cp]*nudge[a.priority]*self.last[a.aid])
        if self.scenario in ('B','C'):
            terms.append(5*10*sum(self.e.values()))
        self.model.Minimize(sum(terms))
        return self

    def solve(self):
        s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=self.time_limit; s.parameters.num_search_workers=8
        status=s.Solve(self.model)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE): raise RuntimeError(f'No solution: {s.StatusName(status)}')
        rec=[]
        for (aid,w,n),v in self.x.items():
            if s.Value(v): rec.append({'activity_id':aid,'week':w,'access_night':n,'eclo':s.Value(self.e[aid,w,n]) if (aid,w,n) in self.e else 0})
        rec.sort(key=lambda r:(r['week'],r['access_night'],r['activity_id']))
        with open(OUT/'SCHEDULE_ACCESS.csv','w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['activity_id','access_seq','week','eclo','access_night']); wr.writeheader()
            seq=defaultdict(int)
            for r in rec:
                seq[r['activity_id']]+=1; wr.writerow({'activity_id':r['activity_id'],'access_seq':seq[r['activity_id']],**{k:r[k] for k in ('week','eclo','access_night')}})
        print(s.StatusName(status), 'objective=',s.ObjectiveValue(), 'accesses=',len(rec))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',choices=['A','B','C'],default='A'); ap.add_argument('--time-limit',type=int,default=30)
    args=ap.parse_args(); RailModel(args.scenario,args.time_limit).build().solve()
