"""Correct interruption timing at formerly continuous exposure joins.

An uninterrupted collinear join costs zero. If delayed, the same declared
brake/rest/ramp convention requires a nonzero return loop before re-entry.
This is a physical-consistency repair of the frozen model, not a new optimizer.
Historical results remain untouched; every separation setting is recomputed.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from pilot import ROOT, Jobs, closest_distance, forbidden_interval, read_jobs
from geometry_portfolio import assignment_of
from execution_cost import verify_execution
from motion_fidelity import motion_setup, motion_phases, resource_guard

EPS = 1e-10


def interruption_cost(jobs, queues, alpha):
    cost = np.zeros(len(jobs.p))
    if math.isinf(alpha):
        return cost
    v = (jobs.b-jobs.a)/jobs.p[:, None]
    speeds = np.linalg.norm(v, axis=1)
    vs = float(np.median(speeds))
    acc = alpha*vs/float(np.median(jobs.p))
    for q in queues:
        for i, j in zip(q[:-1], q[1:]):
            if np.linalg.norm(jobs.b[i]-jobs.a[j]) < 1e-9 and np.linalg.norm(v[i]-v[j]) < 1e-8*vs:
                # Outgoing brake, rest-to-rest backward move, incoming ramp.
                # d = speed^2 / a, below the triangular-profile threshold.
                cost[j] = 4*speeds[i]/acc
    return cost


def interruption_check(jobs, queues, starts, pause):
    bad = []; count = 0
    for q in queues:
        for i, j in zip(q[:-1], q[1:]):
            gap = float(starts[j]-starts[i]-jobs.p[i])
            if pause[j] > 0 and gap > EPS:
                count += 1
                if gap < pause[j]-EPS:
                    bad.append((int(i), int(j), gap, float(pause[j])))
    return bad, count


def dispatch(jobs, queues, setup, pause, radius, continuous_release, cache, deadline):
    starts = np.zeros(len(jobs.p)); active = [None]*len(queues)
    cursor = np.zeros(len(queues), int)
    ready = np.array([setup[q[0]] if q else math.inf for q in queues])
    t, assigned, steps = 0., 0, 0
    while assigned < len(jobs.p):
        steps += 1
        if steps % 128 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError("Interruption correction budget exhausted")
        for ell, job in enumerate(active):
            if job is not None and starts[job]+jobs.p[job] <= t+1e-12:
                active[ell] = None
        releases = []
        for ell, q in enumerate(queues):
            if active[ell] is not None or cursor[ell] == len(q) or ready[ell] > t+1e-12:
                continue
            j = q[cursor[ell]]
            previous_end = starts[q[cursor[ell]-1]]+jobs.p[q[cursor[ell]-1]] if cursor[ell] else -math.inf
            if pause[j] > 0 and t > previous_end+EPS and t < previous_end+pause[j]-EPS:
                ready[ell] = previous_end+pause[j]
                continue
            proposed = t
            while True:
                blocked = False
                for i in active:
                    if i is None or closest_distance(jobs, j, i, proposed, starts[i]) >= radius-1e-9:
                        continue
                    blocked = True
                    if continuous_release:
                        key = (i,j)
                        if key not in cache:
                            cache[key] = forbidden_interval(jobs.a[i],jobs.b[i],jobs.p[i],jobs.a[j],jobs.b[j],jobs.p[j],radius)
                        assert cache[key] is not None
                        new = starts[i]+cache[key][1]
                        if new <= proposed+1e-13:
                            raise ArithmeticError("Non-forward conflict release")
                        proposed = max(new, previous_end+pause[j]) if pause[j] > 0 else new
                    break
                if not blocked or not continuous_release:
                    break
            if proposed <= t+1e-12 and not blocked:
                starts[j], active[ell] = t, j
                cursor[ell] += 1; assigned += 1
                ready[ell] = t+jobs.p[j]+setup[q[cursor[ell]]] if cursor[ell] < len(q) else math.inf
            else:
                if pause[j] > 0:
                    ready[ell] = max(ready[ell], previous_end+pause[j])
                if continuous_release:
                    releases.append(proposed)
        if assigned < len(jobs.p):
            events = [starts[j]+jobs.p[j] for j in active if j is not None]
            events.extend(ready[ell] for ell,q in enumerate(queues)
                          if active[ell] is None and cursor[ell] < len(q) and ready[ell] > t+1e-12)
            events.extend(releases)
            events = [x for x in events if x > t+1e-12 and math.isfinite(x)]
            if not events:
                raise RuntimeError("No forward event")
            t = min(events)
    assert not interruption_check(jobs,queues,starts,pause)[0]
    return starts


def checks():
    j = Jobs(np.arange(2), np.array([[0.,0.],[1.,0.]]), np.array([[1.,0.],[2.,0.]]), np.ones(2), np.zeros(2,bool))
    q = [[0,1]]; pause = interruption_cost(j,q,10.)
    np.testing.assert_allclose(pause,[0.,.4])
    assert not interruption_check(j,q,np.array([0.,1.]),pause)[0]
    assert interruption_check(j,q,np.array([0.,1.2]),pause)[0]
    assert not interruption_check(j,q,np.array([0.,1.4]),pause)[0]
    # Independently integrate the stop / return / restart motion, including a
    # hold at rest. The general helper skips a continuous join, so specify the
    # phases directly here instead of invoking that special case.
    position=np.array([1.,0.]); velocity=np.array([1.,0.]); elapsed=0.
    phases=[(.1,[-10.,0.]),(.1,[-10.,0.]),(.1,[10.,0.]),(.1,[10.,0.])]
    for dt,acc in phases:
        acc=np.array(acc); position+=velocity*dt+.5*acc*dt*dt;velocity+=acc*dt;elapsed+=dt
    np.testing.assert_allclose(position,[1.,0.],atol=1e-12)
    np.testing.assert_allclose(velocity,[1.,0.],atol=1e-12)
    assert abs(elapsed-.4)<1e-12
    # A saved old beam-on schedule passes old geometry checks but fails the
    # newly required velocity-continuity condition at three short pauses.
    w=np.load(ROOT/"selection_witness.npz"); old=json.loads((ROOT/"selection_witness.json").read_text())
    jj=Jobs(np.arange(len(w['p'])),w['a'],w['b'],w['p'],np.zeros(len(w['p']),bool))
    row=next(r for r in old['plans'] if r['role']=='alternative')
    bad, count=interruption_check(jj,row['queues'],w['alternative_starts'],interruption_cost(jj,row['queues'],10.))
    assert len(bad)==3
    return {"closed_form_pause_test":"PASS","independent_phase_integration":"PASS",
            "old_counterexample_alternative_short_pauses":bad,"old_alternative_total_interrupted_joins":count}


def run():
    protocol={"reason":"Three short pauses at zero-setup continuous joins were found in the old 15.88% separation counterexample's alternative. All separation settings are recomputed; no case selection.",
              "rule":"Uninterrupted equal-velocity coincident join costs zero; a delayed join uses the existing brake/rest/ramp convention with minimum loop duration 4*v/a.",
              "unchanged":"All geometries, queues, kappa, alpha, radius, tie rule and two scheduling heuristics remain unchanged. No-spacing results are unchanged.",
              "scope":"r=8; kappa=10; two and four lasers; alpha=10,100,1000,infinity; both prior and prospective sets.",
              "predicted_count":{"plan_evaluations":2704,"comparisons":156},"budget":{"wall_s":600.,"rss_MB":512.},
              "initial_motion":"Each first exposure is initialized at its prescribed position and velocity at its chosen start; initial positioning excluded.",
              "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ["interrupted_motion.py","motion_fidelity_results.json","prospective_geometry_results.json","execution_cost.py","motion_fidelity.py"]},
              "frozen_before_recomputation":True}
    p=ROOT/"interrupted_motion_protocol.json"
    if p.exists(): assert json.loads(p.read_text())==protocol
    else: p.write_text(json.dumps(protocol,indent=2))
    begin=time.perf_counter();deadline=begin+600
    out={"protocol":protocol,"checks":checks(),"rows":[],"comparisons":[],"complete":False}
    saved={}; peak=0.; audits=0
    try:
        for stage,result_file,plan_file,input_file in [
            ('earlier','motion_fidelity_results.json','geometry_portfolio_plans.json','controlled_scan_geometries.npz'),
            ('prospective','prospective_geometry_results.json','prospective_geometry_plans.json','prospective_geometry_inputs.npz')]:
            original=json.loads((ROOT/result_file).read_text()); allplans=json.loads((ROOT/plan_file).read_text()); arrays=np.load(ROOT/input_file)
            names=list(dict.fromkeys(r['case'] for r in original['rows']))
            for name in names:
                if name in ['Layer 10','Nozzle']: jobs=read_jobs(ROOT/'inputs'/(name+' - SharpCornerCons - less5mm.xlsx'))
                else:
                    f=[arrays[name+'_'+k] for k in ['a','b','p','dedicated']];jobs=Jobs(np.arange(len(f[2])),*f)
                for m in [2,4]:
                    plans=allplans[name+f'_m{m}'];labels=list(plans); table={};starts_table={};cache={}
                    for label,q in plans.items():
                        prior=[];assignment=assignment_of(q,len(jobs.p))
                        for av in [10.,100.,1000.,None]:
                            peak=max(peak,resource_guard(deadline,512)); alpha=math.inf if av is None else av
                            setup,_=motion_setup(jobs,q,10.,alpha);pause=interruption_cost(jobs,q,alpha)
                            options=[]
                            for continuous in [False,True]:
                                st=dispatch(jobs,q,setup,pause,8.,continuous,cache,deadline)
                                verify_execution(jobs,st,assignment,q,setup,8.);audits+=1
                                options.append(('continuous_release' if continuous else 'whole_job_wait',st))
                            for st in prior:
                                feasible=all(st[j]-st[i]-jobs.p[i]-setup[j]>=-EPS for z in q for i,j in zip(z[:-1],z[1:]))
                                if feasible and not interruption_check(jobs,q,st,pause)[0]:options.append(('previous_feasible_starts',st))
                            retimer,starts=min(options,key=lambda z:max(z[1]+jobs.p))
                            metric=verify_execution(jobs,starts,assignment,q,setup,8.)
                            bad,n=interruption_check(jobs,q,starts,pause);assert not bad
                            row={"stage":stage,"case":name,"lasers":m,"radius":8.,"kappa":10.,"alpha":av,"plan":label,
                                 "retimer":retimer,"interrupted_continuous_joins":n,"short_pause_violations":0,**metric}
                            out['rows'].append(row);table[(label,av)]=row;starts_table[(label,av)]=starts;prior.append(starts.copy())
                    tol=1e-8*float(sum(jobs.p));low=min(table[(k,None)]['makespan'] for k in labels)
                    choices=[k for k in labels if table[(k,None)]['makespan']<=low+tol]
                    for av in [10.,100.,1000.]:
                        selected=min(choices,key=lambda k:table[(k,av)]['makespan']);best=min(labels,key=lambda k:table[(k,av)]['makespan'])
                        value=table[(selected,av)]['makespan'];ub=table[(best,av)]['makespan'];lb=table[(selected,av)]['fixed_order_lower_bound']
                        c={"stage":stage,"case":name,"lasers":m,"radius":8.,"kappa":10.,"alpha":av,"selected_plan":selected,"alternative_plan":best,
                           "selected_cost":value,"alternative_cost":ub,"selected_order_lower_bound":lb,
                           "time_extra_vs_simplified_pct":100*(value/table[(selected,None)]['makespan']-1),
                           "observed_selection_loss_pct":max(0.,100*(value/ub-1)),"selection_loss_lower_bound_pct":max(0.,100*((lb-EPS)/ub-1))}
                        out['comparisons'].append(c)
                        if c['selection_loss_lower_bound_pct']>1e-6:
                            key=f"{name}_m{m}_a{int(av)}"
                            for role,label in [('selected',selected),('alternative',best)]:
                                saved[key+f'_{role}_starts']=starts_table[(label,av)]
                                saved[key+f'_{role}_assignment']=assignment_of(plans[label],len(jobs.p))
                    print(json.dumps({"stage":stage,"case":name,"lasers":m,"rows":len(out['rows']),"wall_s":time.perf_counter()-begin}),flush=True)
                    (ROOT/'interrupted_motion_results.json').write_text(json.dumps(out,separators=(',',':')))
        out['complete']=len(out['rows'])==2704 and len(out['comparisons'])==156
    except Exception as exc:
        out['error']={"type":type(exc).__name__,"message":str(exc)};raise
    finally:
        out['wall_s']=time.perf_counter()-begin;out['peak_rss_MB']=peak;out['raw_schedule_audits']=audits
        (ROOT/'interrupted_motion_results.json').write_text(json.dumps(out,separators=(',',':')))
        np.savez_compressed(ROOT/'interrupted_motion_witnesses.npz',**saved)
    print('COMPLETE',out['complete'],out['wall_s'],flush=True)


if __name__=='__main__':run()
