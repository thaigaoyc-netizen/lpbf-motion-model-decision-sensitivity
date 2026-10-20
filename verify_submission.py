"""Independent consistency checks and continuous-interval recalculation.

Does not train, tune, or select cases. Fails instead of silently dropping rows.
"""
import json
import math
import hashlib
import time
import sys
from pathlib import Path
import numpy as np
from pilot import ROOT,Jobs
from geometry_portfolio import assignment_of,hatch,sections
from prospective_geometry import new_sections
from motion_fidelity import motion_setup,numerical_checks as phase_checks
from motion_portfolio_sensitivity import cases,numerical_checks as cubic_checks
from interrupted_motion import interruption_cost,interruption_check
from execution_cost import verify_execution
from revision_validation import exact_two,split_check
from submission_revision import OUT,load,dump,check_sources,select,route_cost

def continuous(jobs,plans):
    names=list(plans);vs=np.median(np.linalg.norm(jobs.b-jobs.a,axis=1)/jobs.p)
    A=np.array([[sum(jobs.p[q]) for q in plans[l]] for l in names]);B=np.array([[sum(np.linalg.norm(jobs.a[j]-jobs.b[i])/vs for i,j in zip(q[:-1],q[1:])) for q in plans[l]] for l in names])
    aa,bb=A.ravel(),B.ravel();points=[.05,.2]
    for i in range(len(aa)):
        den=bb[i]-bb[i+1:];good=np.abs(den)>1e-13;q=(aa[i+1:][good]-aa[i])/den[good];points.extend(q[(q>.05)&(q<.2)].tolist())
    q=np.unique(points);v=(A[:,:,None]+B[:,:,None]*q).max(1);regret=100*(v/v.min(0)-1);worst=regret.max(1)
    # Fixed name only when worst-regret values are equal within numerical tolerance.
    selected=min(np.flatnonzero(worst<=worst.min()+1e-10),key=lambda i:names[i])
    dense=np.linspace(.05,.2,2003);vd=(A[:,:,None]+B[:,:,None]*dense).max(1)
    assert np.max((100*(vd/vd.min(0)-1)).max(1)-worst)<1e-8
    mid=(q[:-1]+q[1:])/2;winner=np.argmin((A[:,:,None]+B[:,:,None]*mid).max(1),axis=0)
    switches=[float(q[i+1]) for i in range(len(winner)-1) if winner[i]!=winner[i+1]]
    return dict(plan_names=names,A=A.tolist(),B=B.tolist(),q=q.tolist(),regret=regret[selected].tolist(),selected_plan=names[selected],worst_regret_pct=float(worst[selected]),switches=switches)

def main():
    begin=time.perf_counter();check_sources();d=load('primary_results.json');ex=load('exact_results.json');sr=load('search_results.json');mech=load('mechanism_results.json')
    assert all(x['complete'] for x in [d,ex,sr,mech]);assert len(ex['rows'])==300;assert len(sr['rows'])==104;assert len(mech['rows'])==104
    out={'phase_checks':phase_checks(),'cubic_checks':cubic_checks(),'result_counts':{'plan_rows':len(d['rows']),'comparisons':len(d['comparisons']),'exact':len(ex['rows']),'search':len(sr['rows'])}}
    # A pure selector cannot inspect reference outcomes or inherited starts.
    assert select({'B':1.,'A':1.+1e-9},1.)[0]=='A'
    assert select({'B':1.,'A':1.+1e-6},1.)[0]=='B'
    assert all(r.get('retimer')!='previous_reference_schedule' for r in d['rows'] if r['model']=='constant_speed')
    table={(r['case'],r['lasers'],r['radius'],r['kappa'],r['alpha'],r['model'],r['plan']):r for r in d['rows']}
    for c in d['comparisons']:
        k=(c['case'],c['lasers'],c['radius'],c['kappa']);rc=table[(*k,c['alpha'],c['model'],c['selected_plan'])]['makespan'];cv=table[(*k,None,'constant_speed',c['selected_plan'])]['makespan']
        assert c['selected_plan']==min(c['tied_labels'])
        assert abs(c['selected_cost']-rc)<1e-10 and abs(c['simplified_estimate']-cv)<1e-10
        assert abs(c['selection_loss_pct']-100*(rc/c['best_cost']-1))<1e-8
    schedule=np.load(OUT/'separation_schedules.npz');checked=0;cont=[];geom_error=0.
    shapes={**sections(),**new_sections()}
    for stage,name,m,jobs,plans in cases():
        if m==2 and name not in ['Layer 10','Nozzle']:
            shape,angle=name.rsplit('_',1);regen=hatch(shapes[shape],float(angle))
            for field in ['a','b','p']:
                err=float(np.max(np.abs(getattr(regen,field)-getattr(jobs,field))));geom_error=max(geom_error,err);assert err<1e-10
        for c in [z for z in d['comparisons'] if z['case']==name and z['lasers']==m and z['radius']==8]:
            for role in ['selected','alternative']:
                qs=plans[c[role+'_plan']];st=schedule[f'{name}_m{m}_a{int(c["alpha"])}_{role}'];setup=motion_setup(jobs,qs,10.,c['alpha'])[0]
                metric=verify_execution(jobs,st,assignment_of(qs,len(jobs.p)),qs,setup,8.)
                assert not interruption_check(jobs,qs,st,interruption_cost(jobs,qs,c['alpha']))[0]
                expected=c['selected_cost'] if role=='selected' else c['best_cost'];assert abs(metric['makespan']-expected)<1e-9
                checked+=1
        cont.append(dict(case=name,stage=stage,lasers=m,**continuous(jobs,plans)))
    # Known zero-edge and dedicated-set limits, including an empty second queue.
    zero=np.zeros((3,3));p=np.array([1.,2.,3.]);x,_=exact_two(p,zero);assert x==3.
    x,_=exact_two(p,zero,np.ones(3,bool));assert x==6.
    out['zero_edges_free_initial_and_all_dedicated']='PASS';out['selected_separation_schedule_replays']=checked
    out['constructed_geometry_regeneration_max_error']=geom_error
    out['segmentation']=split_check()
    out['continuous']={s:{'n':sum(c['stage']==s for c in cont),'maximum_pct':max(c['worst_regret_pct'] for c in cont if c['stage']==s),'zero':sum(c['worst_regret_pct']<=1e-6 for c in cont if c['stage']==s)} for s in ['earlier','prospective']}
    old=json.loads((ROOT/'interrupted_motion_results.json').read_text());old_table={(r['case'],r['lasers'],r['plan'],r['alpha']):r for r in old['rows']};changes=[]
    for r in d['rows']:
        if r['radius']!=8:continue
        k=(r['case'],r['lasers'],r['plan'],r['alpha']);delta=r['makespan']-old_table[k]['makespan']
        if abs(delta)>1e-10:changes.append(dict(case=k[0],lasers=k[1],plan=k[2],alpha=k[3],old=old_table[k]['makespan'],new=r['makespan']))
    assert all(r['alpha'] is None for r in changes)
    out['retiming_information_audit']={'cv_cost_rows_changed':len(changes),'reference_cost_rows_changed':0,'changes':changes}
    out['complete']=True;out['wall_s']=time.perf_counter()-begin
    out['runner_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['submission_revision.py','verify_submission.py']}
    dump('continuous_results.json',dict(complete=True,rows=cont));dump('verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['segmentation','retiming_information_audit']},indent=2))

if __name__=='__main__':main()
