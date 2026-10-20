"""Pre-submission numerical revision. Frozen kernels and historical results stay intact.

All times/coordinates use stored units. No equipment calibration is implied.
Run --freeze, then --primary, --exact, --mechanism, --search, --summarize.
"""
import argparse
import csv
import hashlib
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
from pilot import ROOT, Jobs
from geometry_portfolio import assignment_of, portfolio, free_starts
from execution_cost import verify_execution
from motion_fidelity import motion_setup, resource_guard
from interrupted_motion import dispatch, interruption_cost, interruption_check
from motion_portfolio_sensitivity import (cases, groups, motion_parameters, brake_edges,
    hermite_duration, motion_nearest_neighbor, numerical_checks)
from revision_validation import matrices, costs, exact_two, exhaustive_check

OUT=ROOT/'submission'
OUT.mkdir(exist_ok=True)
EPS=1e-6
TARGETS=['rectangle_30','concave_L_30','annulus_30','four_islands_30',
         'U_bracket_45','dogbone_45','perforated_plate_45','unequal_islands_45','Layer 10','Nozzle']

def dump(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))

def load(name):return json.loads((OUT/name).read_text())

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    names=['pilot.py','geometry_portfolio.py','execution_cost.py','motion_fidelity.py',
           'interrupted_motion.py','motion_portfolio_sensitivity.py','revision_validation.py',
           'controlled_scan_geometries.npz','prospective_geometry_inputs.npz',
           'geometry_portfolio_plans.json','prospective_geometry_plans.json','motion_portfolio_added_plans.json']
    names += ['inputs/Layer 10 - SharpCornerCons - less5mm.xlsx','inputs/Nozzle - SharpCornerCons - less5mm.xlsx']
    p={'version':'submission-20261017-1','selection':'Case-sensitive lexicographic plan label among CV costs within 1e-8*sum(p) of minimum. No reference outcome is passed to selector.',
       'primary':'Recompute all 20748 r=0 plan costs and 2704 corrected r=8 schedules. CV retiming uses only its own two dispatchers; no finite-acceleration schedule inheritance. Reference retains verified earlier reference schedules only.',
       'exact':{'cases':TARGETS,'sizes':[8,12,16],'centres':'Two seeds at x-quantiles .25/.75 of midpoint lexicographic order, then nearest n midpoints to seed. Whole directed vectors retained, compact spatial windows; not rehatched full layers.',
                'parameters':{'lasers':2,'radius':0,'kappa':10,'alpha':[10,100]},'edge_models':['constant_speed','brake_rest_ramp','optional_cubic'],'expected_comparisons':300,
                'baselines':'13/19 portfolio, model-driven multistart relocate/swap/reverse search; exact open-path DP + all feasible partitions.',
                'search_proposals_per_start':1200,'search_starts':3,'seed':620171},
       'full_search':{'cases':'all 26 retained cases, m=2/4','radius':0,'kappa':10,'alpha':10,'models':['constant_speed','brake_rest_ramp'],
                      'starts':3,'proposals_per_start':6000,'seed':620171,'common_candidates':'Union of original19 and both objectives\u2019 searched plans; same set for both selectors. No global-optimum claim.'},
       'mechanism':'All 52 configurations at kappa=10 alpha=10 and at fixed acceleration a=720000 stored-coordinate/stored-time^2, V=6000; original13; every critical-chain diagnostic retained.',
       'budget_seconds':{'primary':900,'exact':900,'mechanism':180,'search':1200},'memory_MB':1400,
       'source_sha256':{n:sha(ROOT/n) for n in names},'protocol_frozen_before_outcomes':True,
       'scope':'Directed exposures, free initial position/velocity, no final parking, no thermal/quality claim. Existing motion conventions, no new physics or AI.'}
    path=OUT/'protocol.json'
    if path.exists():assert load('protocol.json')==p
    else:dump('protocol.json',p)
    print('Protocol frozen',flush=True)

def check_sources():
    p=load('protocol.json')
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    return p

def select(cv,total):
    low=min(cv.values());ties=sorted(k for k,v in cv.items() if v<=low+1e-8*total)
    return ties[0],ties

def comparison(meta,cv,rc,total,lower=None):
    selected,ties=select(cv,total);best=min(rc,key=lambda k:(rc[k],k))
    lo=min(ties,key=lambda k:(rc[k],k));hi=max(ties,key=lambda k:(rc[k],k))
    val=rc[selected];ub=rc[best];lb=val if lower is None else lower[selected]
    return dict(meta,selected_plan=selected,alternative_plan=best,tied_labels=ties,
                selected_cost=val,best_cost=ub,simplified_estimate=cv[selected],selected_lower_bound=lb,
                selection_loss_pct=max(0.,100*(val/ub-1)),time_bias_pct=100*(val/cv[selected]-1),
                lower_bound_pct=max(0.,100*((lb-1e-10)/ub-1)),
                optimistic_loss_pct=max(0.,100*(rc[lo]/ub-1)),pessimistic_loss_pct=max(0.,100*(rc[hi]/ub-1)))

def evaluate_plans(jobs,plans,kappa,alpha,model):
    table={};details={}
    for label,qs in plans.items():
        setup=motion_setup(jobs,qs,kappa,math.inf if model=='constant_speed' else alpha)[0]
        if model=='optional_cubic':
            vel,sp,cap,acc=motion_parameters(jobs,kappa,alpha)
            for q in qs:
                if len(q)<2:continue
                i,j=np.array(q[:-1]),np.array(q[1:]);delta=jobs.a[j]-jobs.b[i]
                cubic=hermite_duration(delta,vel[i],vel[j],cap,acc)
                setup[j]=np.minimum(setup[j],cubic)
        loads=[float(sum(jobs.p[q])+sum(setup[q])) for q in qs]
        table[label]=max(loads);details[label]=(setup,loads)
    return table,details

def primary():
    p=check_sources();start=time.perf_counter();deadline=start+p['budget_seconds']['primary']
    added=json.loads((ROOT/'motion_portfolio_added_plans.json').read_text())
    old=json.loads((ROOT/'motion_portfolio_sensitivity_results.json').read_text())
    oldcost={(r['case'],r['lasers'],r['kappa'],r['alpha'],r['model'],r['plan']):r['makespan'] for r in old['rows']}
    result={'complete':False,'rows':[],'comparisons':[],'max_r0_reproduction_error':0.,'schedule_audits':0}
    starts_out={}
    for stage,name,m,jobs,original in cases():
        resource_guard(deadline,p['memory_MB']);plans=dict(original,**added[name+f'_m{m}']);total=float(jobs.p.sum())
        for k in [5.,10.,20.]:
            cv,cd=evaluate_plans(jobs,plans,k,None,'constant_speed')
            for label,value in cv.items():
                result['rows'].append(dict(stage=stage,case=name,lasers=m,radius=0,kappa=k,alpha=None,model='constant_speed',plan=label,makespan=value,chain_loads=cd[label][1]))
            for a in [10.,100.,1000.]:
                for model in ['brake_rest_ramp','optional_cubic']:
                    rc,dd=evaluate_plans(jobs,plans,k,a,model)
                    for label,value in rc.items():
                        err=abs(value-oldcost[(name,m,k,a,model,label)])
                        result['max_r0_reproduction_error']=max(err,result['max_r0_reproduction_error']);assert err<1e-9
                        result['rows'].append(dict(stage=stage,case=name,lasers=m,radius=0,kappa=k,alpha=a,model=model,plan=label,makespan=value,chain_loads=dd[label][1]))
                    for group,labels in groups(list(plans)).items():
                        result['comparisons'].append(comparison(dict(stage=stage,case=name,lasers=m,radius=0,kappa=k,alpha=a,model=model,portfolio=group),{l:cv[l] for l in labels},{l:rc[l] for l in labels},total))
        # Separation: selector is independent of reference-model schedules.
        table={};st_table={};cache={}
        for label,q in original.items():
            prior_reference=[];assignment=assignment_of(q,len(jobs.p))
            for av in [None,10.,100.,1000.]:
                resource_guard(deadline,p['memory_MB']);a=math.inf if av is None else av
                setup=motion_setup(jobs,q,10.,a)[0];pause=interruption_cost(jobs,q,a);options=[]
                for cont in [False,True]:
                    st=dispatch(jobs,q,setup,pause,8.,cont,cache,deadline)
                    verify_execution(jobs,st,assignment,q,setup,8.);result['schedule_audits']+=1
                    assert not interruption_check(jobs,q,st,pause)[0]
                    options.append(('continuous_release' if cont else 'whole_job_wait',st))
                if av is not None:
                    for st in prior_reference:
                        if all(st[j]-st[i]-jobs.p[i]-setup[j]>=-1e-10 for z in q for i,j in zip(z[:-1],z[1:])) and not interruption_check(jobs,q,st,pause)[0]:
                            options.append(('previous_reference_schedule',st))
                tag,st=min(options,key=lambda z:float(max(z[1]+jobs.p)))
                metric=verify_execution(jobs,st,assignment,q,setup,8.)
                if av is not None:prior_reference.append(st.copy())
                table[(label,av)]=metric;st_table[(label,av)]=st
                result['rows'].append(dict(stage=stage,case=name,lasers=m,radius=8,kappa=10.,alpha=av,model='constant_speed' if av is None else 'brake_rest_ramp',plan=label,retimer=tag,**metric))
        cv={l:table[(l,None)]['makespan'] for l in original}
        for a in [10.,100.,1000.]:
            rc={l:table[(l,a)]['makespan'] for l in original};lb={l:table[(l,a)]['fixed_order_lower_bound'] for l in original}
            c=comparison(dict(stage=stage,case=name,lasers=m,radius=8,kappa=10.,alpha=a,model='brake_rest_ramp',portfolio='original_13'),cv,rc,total,lb)
            result['comparisons'].append(c)
            for role,label in [('selected',c['selected_plan']),('alternative',c['alternative_plan'])]:
                starts_out[f'{name}_m{m}_a{int(a)}_{role}']=st_table[(label,a)]
        print(json.dumps({'stage':'primary','case':name,'m':m,'wall_s':round(time.perf_counter()-start,2)}),flush=True)
    assert len(result['rows'])==23452 and len(result['comparisons'])==5772
    result.update(complete=True,wall_s=time.perf_counter()-start,peak_MB=resource_guard(deadline,p['memory_MB']))
    dump('primary_results.json',result);np.savez_compressed(OUT/'separation_schedules.npz',**starts_out)

def route_cost(q,p,edge):
    return float(sum(p[q])+sum(edge[i,j] for i,j in zip(q[:-1],q[1:]))) if q else 0.

class Edges:
    """Bounded on-demand cache; no quadratic dense matrix on full layers."""
    def __init__(self,jobs,model,alpha=10.):
        self.j=jobs;self.model=model;self.alpha=alpha;self.cache={}
        self.v,self.s,self.cap,self.acc=motion_parameters(jobs,10.,alpha)
    def __getitem__(self,pair):
        i,j=pair
        if i is None or j is None:return 0.
        key=(i,j)
        if key not in self.cache:
            d=self.j.a[j]-self.j.b[i];dist=math.hypot(*d)
            if self.model=='constant_speed':val=dist/self.cap
            elif dist<1e-9 and np.linalg.norm(self.v[i]-self.v[j])<1e-8*np.median(self.s):val=0.
            else:
                z=d-(self.v[i]*self.s[i]+self.v[j]*self.s[j])/(2*self.acc)
                gap=math.hypot(*z)
                val=(self.s[i]+self.s[j])/self.acc+(2*math.sqrt(gap/self.acc) if gap<=self.cap**2/self.acc else gap/self.cap+self.cap/self.acc)
            self.cache[key]=val
        return self.cache[key]

def improve(jobs,edge,initial,proposals,seed):
    """Standard multineighborhood descent: relocate, swap, bounded reversal.

    Fixed proposal count, deterministic seed, lexicographic (max load,total load)
    acceptance. No target-result tuning or claim to a new planner.
    """
    rng=np.random.default_rng(seed);q=[list(x) for x in initial];m=len(q);p=jobs.p
    loads=np.array([route_cost(z,p,edge) for z in q]);accepted=0
    def score(z):return (round(float(max(z)),12),round(float(sum(z)),12))
    for step in range(proposals):
        l=int(rng.integers(m));h=int(rng.integers(m))
        if not q[l]:continue
        i=int(rng.integers(len(q[l])));x=q[l][i];kind=step%3
        newloads=loads.copy()
        if kind==0:
            if jobs.dedicated[x] and h!=0:continue
            src=q[l];before=src[i-1] if i else None;after=src[i+1] if i+1<len(src) else None
            removal=-p[x]-edge[before,x]-edge[x,after]+edge[before,after]
            new_src=src[:i]+src[i+1:];dst=new_src if h==l else q[h]
            t=int(rng.integers(len(dst)+1));a=dst[t-1] if t else None;b=dst[t] if t<len(dst) else None
            insertion=p[x]+edge[a,x]+edge[x,b]-edge[a,b]
            newloads[l]+=removal;newloads[h]+=insertion
            updates={l:new_src,h:dst[:t]+[x]+dst[t:]} if h!=l else {l:dst[:t]+[x]+dst[t:]}
        elif kind==1:
            if not q[h]:continue
            t=int(rng.integers(len(q[h])));y=q[h][t]
            if x==y or (jobs.dedicated[x] and h!=0) or (jobs.dedicated[y] and l!=0):continue
            updates={z:q[z].copy() for z in {l,h}};updates[l][i]=y;updates[h][t]=x
            for z in updates:
                changed={i} if z==l else set()
                if z==h:changed.add(t)
                positions={a for b in changed for a in [b-1,b] if 0<=a<len(q[z])-1}
                delta=sum(edge[updates[z][a],updates[z][a+1]]-edge[q[z][a],q[z][a+1]] for a in positions)
                delta+=sum(p[updates[z][b]]-p[q[z][b]] for b in changed)
                newloads[z]+=delta
        else:
            if len(q[l])<2:continue
            t=min(len(q[l])-1,i+int(rng.integers(1,min(32,len(q[l])))))
            if t==i:continue
            nq=q[l].copy();nq[i:t+1]=reversed(nq[i:t+1]);updates={l:nq}
            newloads[l]+=sum(edge[nq[a],nq[a+1]]-edge[q[l][a],q[l][a+1]] for a in range(max(0,i-1),min(t+1,len(nq)-1)))
        if score(newloads)<score(loads):
            for z,nq in updates.items():q[z]=nq
            loads=newloads;accepted+=1
    measured=np.array([route_cost(z,p,edge) for z in q]);np.testing.assert_allclose(loads,measured,atol=1e-9,rtol=1e-9)
    assert sorted(j for z in q for j in z)==list(range(len(p)))
    assert np.all(assignment_of(q,len(p))[jobs.dedicated]==0)
    return q,float(max(measured)),accepted

def multistart(jobs,edge,plans,proposals,seed):
    if isinstance(edge,np.ndarray):
        matrix=edge
        class DenseEdges:
            def __getitem__(self,pair):
                i,j=pair
                return 0. if i is None or j is None else float(matrix[i,j])
        edge=DenseEdges()
    ordered=sorted(plans,key=lambda l:(max(route_cost(q,jobs.p,edge) for q in plans[l]),l))
    seen=set();starts=[]
    for l in ordered:
        sig=tuple(tuple(q) for q in plans[l])
        if sig not in seen:seen.add(sig);starts.append(l)
        if len(starts)==3:break
    results=[]
    for k,l in enumerate(starts):
        q,value,accepted=improve(jobs,edge,plans[l],proposals,seed+k)
        results.append({'start':l,'queues':q,'cost':value,'accepted':accepted})
    return min(results,key=lambda r:r['cost']),results

def spatial_indices(jobs,size,fraction):
    xy=(jobs.a+jobs.b)/2
    order=np.lexsort((jobs.ids,xy[:,1],xy[:,0]));seed=order[int(fraction*(len(order)-1))]
    distance=np.sum((xy-xy[seed])**2,axis=1)
    index=np.lexsort((jobs.ids,distance))[:size]
    return np.sort(index),int(seed),float(np.sqrt(distance[index].max()))

def exact():
    p=check_sources();start=time.perf_counter();deadline=start+p['budget_seconds']['exact'];out=[]
    checks=exhaustive_check()
    for _,name,m,full,_ in cases():
        if m!=2 or name not in TARGETS:continue
        for fraction in [.25,.75]:
            for size in [8,12,16]:
                resource_guard(deadline,p['memory_MB'])
                ix,seed,radius=spatial_indices(full,size,fraction);j=full.take(ix);plans=portfolio(j,2)
                for base,short in [('LPT_tie_stable','LPT'),('region_x_file','x'),('region_y_file','y')]:
                    for a in [10.,100.]:plans[f'motion_NN_{short}_a{int(a)}']=motion_nearest_neighbor(j,plans[base],a,deadline)
                # Preserve parent dimensional acceleration, rather than silently changing it when taking a subset.
                for a in [10.,100.]:
                    parent_acc=motion_parameters(full,10.,a)[3]
                    sub_alpha=parent_acc*np.median(j.p)/np.median(np.linalg.norm(j.b-j.a,axis=1)/j.p)
                    mm=matrices(j,10.,sub_alpha)
                    for model,edge in mm.items():
                        if model=='constant_speed' and a==100.:continue
                        tick=time.perf_counter();opt,qs=exact_two(j.p,edge,j.dedicated);dt=time.perf_counter()-tick
                        pc=costs(j.p,edge,plans);b13=min(v for l,v in pc.items() if not l.startswith('motion_NN_'));b19=min(pc.values())
                        best,trials=multistart(j,edge,plans,1200,620171)
                        assert best['cost']>=opt-1e-9 and best['cost']<=b19+1e-9
                        out.append(dict(case=name,fraction=fraction,size=size,source_vectors=len(full.p),subset_indices=ix.tolist(),seed_index=seed,window_radius=radius,dedicated_count=int(j.dedicated.sum()),model=model,parent_alpha=None if model=='constant_speed' else a,subset_alpha=None if model=='constant_speed' else float(sub_alpha),acceleration=None if model=='constant_speed' else parent_acc,
                                        exact_cost=opt,exact_queues=qs,dp_wall_s=dt,portfolio13=b13,portfolio19=b19,search_cost=best['cost'],search_queues=best['queues'],search_trials=trials,
                                        gap13_pct=100*(b13/opt-1),gap19_pct=100*(b19/opt-1),gap_search_pct=max(0.,100*(best['cost']/opt-1))))
        print(json.dumps({'stage':'exact','case':name,'comparisons':len(out),'wall_s':round(time.perf_counter()-start,2)}),flush=True)
    assert len(out)==300
    dump('exact_results.json',dict(complete=True,checks=checks,rows=out,wall_s=time.perf_counter()-start,peak_MB=resource_guard(deadline,p['memory_MB'])))

def chain_diagnostics(jobs,q,kappa,alpha):
    vel,sp,cap,acc=motion_parameters(jobs,kappa,alpha);setup=motion_setup(jobs,q,kappa,alpha)[0];cv=motion_setup(jobs,q,kappa,math.inf)[0]
    data=[]
    for laser,z in enumerate(q):
        i,j=np.array(z[:-1],int),np.array(z[1:],int);work=float(sum(jobs.p[z]));motion=float(sum(setup[z]));nonzero=setup[j]>1e-10
        direct=np.linalg.norm(jobs.a[j]-jobs.b[i],axis=1);dv=np.linalg.norm(vel[j]-vel[i],axis=1)
        brake_end=float(sum((sp[i][nonzero]+sp[j][nonzero])/acc))
        data.append(dict(laser=laser,exposure=work,off_time=motion,cv_off=float(sum(cv[z])),cost=work+motion,jumps=int(nonzero.sum()),direct_distance=float(direct.sum()),endpoint_dv_sum=float(dv.sum()),endpoint_change_count=int((dv>1e-8*np.median(sp)).sum()),brake_ramp_time=brake_end,rest_move_time=motion-brake_end))
    crit=max(range(len(q)),key=lambda k:data[k]['cost']);cvcrit=max(range(len(q)),key=lambda k:data[k]['exposure']+data[k]['cv_off'])
    return dict(critical=crit,cv_critical=cvcrit,critical_changed=crit!=cvcrit,chains=data,**{'critical_chain':data[crit]})

def mechanism():
    p=check_sources();start=time.perf_counter();deadline=start+p['budget_seconds']['mechanism'];out=[]
    for stage,name,m,j,plans in cases():
        for mode in ['alpha10','fixed_acceleration']:
            vs=float(np.median(np.linalg.norm(j.b-j.a,axis=1)/j.p));a=10. if mode=='alpha10' else 720000.*np.median(j.p)/vs
            k=10. if mode=='alpha10' else 6000./vs
            cv,_=evaluate_plans(j,plans,k,None,'constant_speed');rc,_=evaluate_plans(j,plans,k,a,'brake_rest_ramp')
            c=comparison(dict(stage=stage,case=name,lasers=m,mode=mode,kappa=k,alpha=float(a),acceleration=motion_parameters(j,k,a)[3]),cv,rc,float(sum(j.p)))
            c['selected_diagnostics']=chain_diagnostics(j,plans[c['selected_plan']],k,a)
            c['alternative_diagnostics']=chain_diagnostics(j,plans[c['alternative_plan']],k,a)
            out.append(c);resource_guard(deadline,p['memory_MB'])
    dump('mechanism_results.json',dict(complete=True,rows=out,wall_s=time.perf_counter()-start))

def search():
    p=check_sources();start=time.perf_counter();deadline=start+p['budget_seconds']['search'];out=[];allplans={}
    added=json.loads((ROOT/'motion_portfolio_added_plans.json').read_text())
    for stage,name,m,j,original in cases():
        plans=dict(original,**added[name+f'_m{m}']);expanded=dict(plans);search_info=[]
        for model in ['constant_speed','brake_rest_ramp']:
            edge=Edges(j,model);best,trials=multistart(j,edge,plans,6000,620171)
            for trial_id,r in enumerate(trials):expanded[f'search_{model}_{trial_id}']=r['queues']
            search_info.append(dict(model=model,cache_edges=len(edge.cache),trials=trials))
            resource_guard(deadline,p['memory_MB'])
        cv,_=evaluate_plans(j,expanded,10,None,'constant_speed')
        for model in ['brake_rest_ramp','optional_cubic']:
            rc,_=evaluate_plans(j,expanded,10,10,model)
            old=comparison({}, {l:cv[l] for l in plans},{l:rc[l] for l in plans},float(sum(j.p)))
            c=comparison(dict(stage=stage,case=name,lasers=m,model=model,old_portfolio=old,search_trials=search_info),cv,rc,float(sum(j.p)))
            c['reference_improvement_pct']=100*(1-c['best_cost']/old['best_cost'])
            c['rows']=[dict(plan=l,cv=cv[l],reference=rc[l]) for l in expanded];out.append(c)
        allplans[name+f'_m{m}']={l:q for l,q in expanded.items() if l.startswith('search_')}
        print(json.dumps({'stage':'search','case':name,'m':m,'wall_s':round(time.perf_counter()-start,2)}),flush=True)
    dump('search_plans.json',allplans)
    dump('search_results.json',dict(complete=True,rows=out,wall_s=time.perf_counter()-start,peak_MB=resource_guard(deadline,p['memory_MB'])))

def stats(rows,key='selection_loss_pct'):
    v=np.array([r[key] for r in rows]);return dict(n=len(v),positive=int((v>EPS).sum()),zero=int((v<=EPS).sum()),median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(v.max()),over5=int((v>5).sum()))

def summarize():
    d=load('primary_results.json');comp=d['comparisons'];main=[r for r in comp if r['portfolio']=='original_13' and r['model']=='brake_rest_ramp'];assert len(main)==624
    summary={'primary':{'wall_s':d['wall_s'],'schedule_audits':d['schedule_audits'],'max_r0_reproduction_error':d['max_r0_reproduction_error']},'table2':[],'table3':[],'table4':[],'pairs':{},'ties':[]}
    for radius in [0,8]:
        for a in [10,100,1000]:
            r=[c for c in main if c['stage']=='prospective' and c['radius']==radius and c['alpha']==a]
            summary['table2'].append(dict(radius=radius,alpha=a,loss=stats(r),bias=stats(r,'time_bias_pct'),bound=stats(r,'lower_bound_pct')))
    for model in ['brake_rest_ramp','optional_cubic']:
        for port in ['original_13','expanded_19']:
            r=[c for c in comp if c['radius']==0 and c['model']==model and c['portfolio']==port]
            summary['table3'].append(dict(model=model,portfolio=port,**stats(r)))
            summary['ties'].append(dict(model=model,portfolio=port,lexicographic=stats(r),optimistic=stats(r,'optimistic_loss_pct'),pessimistic=stats(r,'pessimistic_loss_pct')))
    for port in groups(list(json.loads((ROOT/'geometry_portfolio_plans.json').read_text()).values())[0]):
        r=[c for c in comp if c['radius']==0 and c['model']=='optional_cubic' and c['portfolio']==port]
        if r:summary['table4'].append(dict(portfolio=port,**stats(r)))
    def key(r):return r['case'],r['lasers'],r['kappa'],r['alpha']
    for label,mm,pp in [('motion',['brake_rest_ramp','optional_cubic'],['original_13']*2),('portfolio',['optional_cubic']*2,['original_13','expanded_19'])]:
        a={key(r):r for r in comp if r['radius']==0 and r['model']==mm[0] and r['portfolio']==pp[0]};b={key(r):r for r in comp if r['radius']==0 and r['model']==mm[1] and r['portfolio']==pp[1]}
        matrix=np.zeros((2,2),int);rows=[]
        for k in sorted(a):
            va,vb=a[k]['selection_loss_pct'],b[k]['selection_loss_pct'];matrix[int(va>EPS),int(vb>EPS)]+=1
            rows.append(dict(case=k[0],lasers=k[1],kappa=k[2],alpha=k[3],before=va,after=vb,delta=vb-va))
        summary['pairs'][label]={'matrix':matrix.tolist(),'rows':rows,'unchanged':sum(abs(r['delta'])<=EPS for r in rows)}
    ex=load('exact_results.json');summary['exact']=[]
    for size in [8,12,16]:
        for model in ['constant_speed','brake_rest_ramp','optional_cubic']:
            rr=[r for r in ex['rows'] if r['size']==size and r['model']==model]
            summary['exact'].append(dict(size=size,model=model,portfolio13=stats(rr,'gap13_pct'),portfolio19=stats(rr,'gap19_pct'),search=stats(rr,'gap_search_pct')))
    mech=load('mechanism_results.json')['rows'];summary['mechanism']=[]
    for mode in ['alpha10','fixed_acceleration']:
        r=[c for c in mech if c['mode']==mode];positive=[c for c in r if c['selection_loss_pct']>EPS]
        diff=lambda c,k:c['selected_diagnostics']['critical_chain'][k]-c['alternative_diagnostics']['critical_chain'][k]
        summary['mechanism'].append(dict(mode=mode,**stats(r),positive_more_jumps=sum(diff(c,'jumps')>0 for c in positive),positive_more_distance=sum(diff(c,'direct_distance')>1e-8 for c in positive),positive_more_endpoint_change=sum(diff(c,'endpoint_dv_sum')>1e-8 for c in positive),positive_more_motion=sum(diff(c,'off_time')>1e-10 for c in positive),selected_critical_switch=sum(c['selected_diagnostics']['critical_changed'] for c in r)))
    sr=load('search_results.json');summary['search']=[]
    for model in ['brake_rest_ramp','optional_cubic']:
        rr=[r for r in sr['rows'] if r['model']==model]
        summary['search'].append(dict(model=model,before=stats([r['old_portfolio'] for r in rr]),after=stats(rr),reference_improvement=stats(rr,'reference_improvement_pct')))
    summary['runtimes']={name:load(name+'_results.json')['wall_s'] for name in ['primary','exact','mechanism','search']}
    dump('summary.json',summary)
    for name,rows in [('primary_comparisons',comp),('exact_comparisons',ex['rows']),('mechanism_comparisons',mech),('search_comparisons',sr['rows'])]:
        flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows];keys=list(dict.fromkeys(k for r in flat for k in r))
        with (OUT/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['pairs','ties']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','primary','exact','mechanism','search','summarize'])
    globals()[parser.parse_args().stage]()
