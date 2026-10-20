"""Bounded reviewer checks: exact tiny schedules, ties, segmentation invariance.

Uses only retained geometries and established dynamic programming. These are
computational checks of the manuscript, not a proposed scheduling algorithm.
"""
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pilot import ROOT, Jobs
from geometry_portfolio import portfolio, assignment_of
from motion_fidelity import motion_setup, resource_guard
from motion_portfolio_sensitivity import cases, brake_edges, hermite_duration, motion_parameters, motion_nearest_neighbor


def exact_two(p, edge, dedicated=None):
    """Held--Karp open-path costs, then exhaustive two-laser subset partition.

    Fixed directed exposures. Either laser may be empty. Free initial
    preposition, no final parking, r=0. Dedicated jobs must be on laser 0.
    """
    n=len(p); full=(1<<n)-1
    dp=np.full((1<<n,n),np.inf);prev=np.full((1<<n,n),-1,np.int16)
    for i in range(n):dp[1<<i,i]=p[i]
    members=[np.array([j for j in range(n) if mask>>j&1],int) for mask in range(1<<n)]
    for mask in range(1,1<<n):
        for j in members[mask]:
            rest=mask^(1<<j)
            if not rest:continue
            ix=members[rest];values=dp[rest,ix]+edge[ix,j]+p[j]
            k=int(np.argmin(values));dp[mask,j]=values[k];prev[mask,j]=ix[k]
    cost=np.min(dp,axis=1);cost[0]=0.
    dedicated=np.zeros(n,bool) if dedicated is None else dedicated
    must=sum(1<<i for i in np.flatnonzero(dedicated))
    best=np.inf;best_mask=None
    for mask in range(1<<n):
        if mask&must!=must:continue
        c=max(cost[mask],cost[full^mask])
        if c<best:best=c;best_mask=mask
    def route(mask):
        if not mask:return []
        j=int(np.argmin(dp[mask]));path=[]
        while mask:
            path.append(j);k=int(prev[mask,j]);mask^=1<<j;j=k
        return path[::-1]
    queues=[route(best_mask),route(full^best_mask)]
    assert sorted(j for q in queues for j in q)==list(range(n))
    assert np.all(assignment_of(queues,n)[dedicated]==0)
    check=max(sum(p[q])+sum(edge[i,j] for i,j in zip(q[:-1],q[1:])) for q in queues)
    assert abs(check-best)<1e-10
    return float(best),queues


def exhaustive_check():
    rng=np.random.default_rng(82491);errors=[]
    for trial in range(3):
        n=6;p=rng.uniform(.1,1,n);edge=rng.uniform(0,.7,(n,n));np.fill_diagonal(edge,0)
        dedicated=np.array([True,False,False,False,False,False]) if trial==1 else np.zeros(n,bool)
        optimum,_=exact_two(p,edge,dedicated)
        brute=np.inf;count=0
        for order in itertools.permutations(range(n)):
            for cut in range(n+1):
                if any(dedicated[j] for j in order[cut:]):continue
                qs=[list(order[:cut]),list(order[cut:])]
                c=max(sum(p[q])+sum(edge[i,j] for i,j in zip(q[:-1],q[1:])) for q in qs)
                brute=min(brute,c);count+=1
        errors.append({"trial":trial,"enumerated_feasible_plans":count,"absolute_error":abs(brute-optimum)})
        assert abs(brute-optimum)<1e-12
    return errors


def matrices(jobs,kappa,alpha):
    n=len(jobs.p);i,j=np.indices((n,n));v,s,V,a=motion_parameters(jobs,kappa,alpha)
    delta=jobs.a[j]-jobs.b[i]
    cv=np.linalg.norm(delta,axis=-1)/V
    brake=brake_edges(jobs,i,j,kappa,alpha)
    cubic=hermite_duration(delta,v[i],v[j],V,a)
    continuous=(np.linalg.norm(delta,axis=-1)<1e-9)&(np.linalg.norm(v[i]-v[j],axis=-1)<1e-8*np.median(s))
    cubic[continuous]=0
    optional=np.minimum(brake,cubic)
    for z in [cv,brake,optional]:np.fill_diagonal(z,0.)
    return {"constant_speed":cv,"brake_rest_ramp":brake,"optional_cubic":optional}


def costs(p,edge,plans):
    return {name:float(max(sum(p[q])+sum(edge[i,j] for i,j in zip(q[:-1],q[1:])) for q in qs)) for name,qs in plans.items()}


def tiny_comparison(deadline):
    targets={"rectangle_30","concave_L_30","annulus_30","four_islands_30",
             "U_bracket_45","dogbone_45","perforated_plate_45","unequal_islands_45","Layer 10","Nozzle"}
    results=[]
    for stage,name,m,full,_ in cases():
        if m!=2 or name not in targets:continue
        resource_guard(deadline,512)
        index=np.linspace(0,len(full.p)-1,12,dtype=int)
        j=Jobs(full.ids[index],full.a[index],full.b[index],full.p[index],full.dedicated[index])
        plans=portfolio(j,2)
        for source in ["LPT_tie_stable","region_x_file","region_y_file"]:
            for alpha in [10.,100.]:
                plans[f"motion_NN_{source}_a{int(alpha)}"]=motion_nearest_neighbor(j,plans[source],alpha,deadline)
        for alpha in [10.,100.]:
            mm=matrices(j,10.,alpha);cv=costs(j.p,mm["constant_speed"],plans)
            for model,edge in mm.items():
                if model=="constant_speed" and alpha==100:continue
                optimum,queues=exact_two(j.p,edge,j.dedicated)
                pc=costs(j.p,edge,plans);original={k:v for k,v in pc.items() if not k.startswith("motion_NN_")}
                tied=[k for k,c in cv.items() if c<=min(cv.values())+1e-8*j.p.sum()]
                chosen=min(tied,key=lambda k:(pc[k],k))
                results.append({"case":name,"source_vectors":len(full.p),"subset_indices":index.tolist(),
                    "dedicated_count":int(j.dedicated.sum()),"lasers":2,"kappa":10.,"alpha":None if model=="constant_speed" else alpha,
                    "model":model,"exact_cost":optimum,"exact_queues":queues,
                    "portfolio_13_best":min(original.values()),"portfolio_19_best":min(pc.values()),
                    "gap_13_pct":100*(min(original.values())/optimum-1),"gap_19_pct":100*(min(pc.values())/optimum-1),
                    "simplified_selected_19":chosen,"selected_cost":pc[chosen],
                    "selection_loss_19_pct":100*(pc[chosen]/min(pc.values())-1),
                    "selected_global_gap_pct":100*(pc[chosen]/optimum-1)})
                resource_guard(deadline,512)
    assert len(results)==50
    return results


def tie_audit():
    data=json.loads((ROOT/"motion_portfolio_sensitivity_results.json").read_text())
    cv={};ref={}
    for row in data["rows"]:
        key=(row["case"],row["lasers"],row["kappa"])
        if row["model"]=="constant_speed":cv.setdefault(key,{})[row["plan"]]=row["makespan"]
        else:ref.setdefault((*key,row["alpha"],row["model"]),{})[row["plan"]]=row["makespan"]
    total={}
    for _,name,m,jobs,_ in cases():total[(name,m)]=float(jobs.p.sum())
    result=[]
    for key,rc in ref.items():
        for portfolio_name in ["original_13","expanded_19"]:
            cc={k:v for k,v in cv[key[:3]].items() if portfolio_name=="expanded_19" or not k.startswith("motion_NN_")}
            tied=[k for k,v in cc.items() if v<=min(cc.values())+1e-8*total[key[:2]]]
            best=min(rc[k] for k in cc)
            choices={"reference_best_tie":min(tied,key=lambda k:(rc[k],k)),
                     "lexicographic_tie":min(tied),"reference_worst_tie":max(tied,key=lambda k:(rc[k],k))}
            result.append({"case":key[0],"lasers":key[1],"kappa":key[2],"alpha":key[3],"model":key[4],
                "portfolio":portfolio_name,"tied_labels":tied,"choices":choices,
                "loss_pct":{k:100*(rc[v]/best-1) for k,v in choices.items()}})
    return result


def split_check():
    from revise_paper_figures import witness
    jobs,_,rr=witness();out=[]
    velocity,speed,cap,a=motion_parameters(jobs,10.,10.)
    for split in [1,2,4]:
        aa=np.concatenate([jobs.a+(jobs.b-jobs.a)*k/split for k in range(split)],axis=0)
        bb=np.concatenate([jobs.a+(jobs.b-jobs.a)*(k+1)/split for k in range(split)],axis=0)
        new=Jobs(np.arange(len(aa)),aa,bb,np.tile(jobs.p/split,split),np.tile(jobs.dedicated,split))
        new_alpha=a*np.median(new.p)/np.median(speed)
        for r in rr:
            qs=[[int(j+k*len(jobs.p)) for j in q for k in range(split)] for q in r["queues"]]
            fixed=motion_setup(new,qs,10.,new_alpha)[0];naive=motion_setup(new,qs,10.,10.)[0]
            fixed_cost=max(sum(new.p[q]+fixed[q]) for q in qs)
            naive_cost=max(sum(new.p[q]+naive[q]) for q in qs)
            original=max(r["work"]+r["fa_sum"])
            assert abs(fixed_cost-original)<1e-10
            out.append({"plan":r["role"],"split_each_vector_into":split,"vector_count":len(aa),
                "dimensional_acceleration":float(a),"adjusted_alpha":float(new_alpha),
                "fixed_acceleration_cost":float(fixed_cost),"fixed_alpha_cost":float(naive_cost),
                "fixed_alpha_acceleration_factor":split,"invariance_error":float(abs(fixed_cost-original))})
    return out


def main():
    begin=time.perf_counter();deadline=begin+600
    source=["geometry_portfolio.py","motion_fidelity.py","motion_portfolio_sensitivity.py",
            "geometry_portfolio_plans.json","prospective_geometry_plans.json","motion_portfolio_sensitivity_results.json"]
    protocol={"scope":"Reviewer-requested checks on existing inputs; no fitted model, CFD, or new data.",
        "frozen_before_execution":True,"budget_wall_s":600,"budget_peak_MB":512,
        "tiny_instances":"10 retained geometries, angle 30 or 45 for constructed cases; 12 indices from linspace(0,n-1,12); 2 lasers",
        "tiny_parameters":{"kappa":10,"alpha":[10,100],"r":0},
        "exact_scope":"All assignments and orders of directed subset jobs under declared edge times, free initial/final positions; not full layers or minimum-time control.",
        "tie_rules":["reference_best_tie (existing diagnostic)","lexicographic_tie (outcome-independent)","reference_worst_tie (bound)"],
        "segmentation":"Original unequal-island S/B queues: split every vector into 1/2/4 contiguous equal pieces; preserve queue and dimensional acceleration. Fixed alpha additionally diagnoses confounding.",
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/"revision_validation_protocol.json").write_text(json.dumps(protocol,indent=2))
    result={"protocol":protocol,"complete":False}
    try:
        result["dp_exhaustive_checks"]=exhaustive_check()
        result["small_exact_results"]=tiny_comparison(deadline)
        result["tie_sensitivity"]=tie_audit()
        result["segmentation"]=split_check()
        result["peak_rss_MB"]=resource_guard(deadline,512)
        result["wall_s"]=time.perf_counter()-begin;result["complete"]=True
    finally:
        (ROOT/"revision_validation_results.json").write_text(json.dumps(result,indent=2))
    compact={"complete":result["complete"],"wall_s":result["wall_s"],"peak_rss_MB":result["peak_rss_MB"],"exact_comparisons":len(result["small_exact_results"])}
    for model in ["constant_speed","brake_rest_ramp","optional_cubic"]:
        rows=[r for r in result["small_exact_results"] if r["model"]==model]
        compact[model]={"n":len(rows),"median_gap_13_pct":float(np.median([r["gap_13_pct"] for r in rows])),
                        "max_gap_13_pct":max(r["gap_13_pct"] for r in rows),
                        "median_gap_19_pct":float(np.median([r["gap_19_pct"] for r in rows])),
                        "max_gap_19_pct":max(r["gap_19_pct"] for r in rows)}
    compact["ties"]={}
    for model in ["brake_rest_ramp","optional_cubic"]:
        for portfolio_name in ["original_13","expanded_19"]:
            rows=[r for r in result["tie_sensitivity"] if r["model"]==model and r["portfolio"]==portfolio_name]
            compact["ties"][model+"_"+portfolio_name]={
                "multiple_tied_labels":sum(len(r["tied_labels"])>1 for r in rows),
                **{rule:{"positive":sum(r["loss_pct"][rule]>1e-6 for r in rows),
                         "maximum_pct":max(r["loss_pct"][rule] for r in rows),
                         "p95_pct":float(np.percentile([r["loss_pct"][rule] for r in rows],95))}
                   for rule in ["reference_best_tie","lexicographic_tie","reference_worst_tie"]}}
    (ROOT/"revision_validation_summary.json").write_text(json.dumps(compact,indent=2))
    print(json.dumps(compact,indent=2))


if __name__=="__main__":main()
