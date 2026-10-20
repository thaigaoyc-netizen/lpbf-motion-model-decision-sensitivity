"""Controlled geometry extension and equal-portfolio execution-cost comparison.

All constructors are conventional heuristics, not proposed new algorithms.
The comparison changes ONLY the objective used to select from the SAME plans.
Generated sections are computational cases, not physical printing experiments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

from pilot import ROOT, Jobs, lpt, np, read_jobs
from execution_cost import dispatch, ordered_queues, setup_times, verify_execution


def rectangle(x, y, w, h):
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], float)


def sections():
    angles = np.arange(128) * (2*np.pi/128)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    return {
        "rectangle": [rectangle(-18, -12, 36, 24)],
        "concave_L": [np.array([[-18,-18],[18,-18],[18,-6],[-6,-6],[-6,18],[-18,18]], float)],
        "annulus": [24*circle, 12*circle],
        "four_islands": [rectangle(x, y, 16, 16) for x in [-28,12] for y in [-28,12]],
    }


def inside_even_odd(points, polygons):
    inside = np.zeros(len(points), bool)
    x, y = points[:,0], points[:,1]
    for poly in polygons:
        for a,b in zip(poly, np.roll(poly,-1,axis=0)):
            if a[1] == b[1]:
                continue
            crossing = (a[1] > y) != (b[1] > y)
            xx = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            inside ^= crossing & (x < xx)
    return inside


def hatch(polygons, degrees, spacing=.8, max_length=5., scan_speed=600.):
    theta = np.deg2rad(degrees)
    rot = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
    aligned = [p @ rot for p in polygons]
    ymin = min(p[:,1].min() for p in aligned)
    ymax = max(p[:,1].max() for p in aligned)
    aa,bb = [],[]
    for line,y in enumerate(np.arange(ymin+spacing/2, ymax, spacing)):
        cuts=[]
        for poly in aligned:
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                    cuts.append(float(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])))
        cuts.sort()
        assert len(cuts)%2 == 0
        intervals=list(zip(cuts[::2],cuts[1::2]))
        if line%2:
            intervals=[(b,a) for a,b in reversed(intervals)]
        for a,b in intervals:
            length=abs(b-a)
            if length<1e-8:
                continue
            distance=np.r_[np.arange(0.,length,max_length),length]
            xx=a+np.sign(b-a)*distance
            for x0,x1 in zip(xx[:-1],xx[1:]):
                if abs(x1-x0)>1e-8:
                    aa.append([x0,y]);bb.append([x1,y])
    a,b=np.array(aa)@rot.T,np.array(bb)@rot.T
    length=np.linalg.norm(b-a,axis=1)
    assert len(length)>0 and np.all(length>0) and length.max()<max_length+1e-9
    for fraction in [.1,.5,.9]:
        assert inside_even_odd(a+fraction*(b-a),polygons).all()
    return Jobs(np.arange(len(a)),a,b,length/scan_speed,np.zeros(len(a),bool))


def assignment_of(queues,n):
    assignment=np.empty(n,int)
    assert sorted(j for q in queues for j in q)==list(range(n))
    for ell,q in enumerate(queues):
        assignment[q]=ell
    return assignment


def list_queues(jobs,m,order):
    queues=[[] for _ in range(m)]
    load=np.zeros(m)
    for j in order:
        ell=0 if jobs.dedicated[j] else int(np.argmin(load))
        queues[ell].append(int(j));load[ell]+=jobs.p[j]
    return queues


def spatial_queues(jobs,m,axis):
    center=(jobs.a+jobs.b)/2
    order=np.lexsort((jobs.ids,center[:,axis]))
    mass=np.cumsum(jobs.p[order])-jobs.p[order]/2
    assignment=np.empty(len(order),int)
    assignment[order]=np.minimum(m-1,(mass*m/jobs.p.sum()).astype(int))
    assignment[jobs.dedicated]=0
    return [np.where(assignment==ell)[0].tolist() for ell in range(m)]


def append_eft(jobs,m,ratio):
    """Standard greedy minimum completion time including sequence setup.

    No separation in this construction step; ALL plans receive the same
    independently checked separation retiming afterwards.
    """
    n=len(jobs.p)
    queues=[[] for _ in range(m)]
    load=np.zeros(m)
    remaining=np.ones(n,bool)
    speed=float(np.median(np.linalg.norm(jobs.b-jobs.a,axis=1)/jobs.p))
    scale=float(np.median(jobs.p))
    for _ in range(n):
        candidates=np.where(remaining)[0]
        best=None
        for ell in range(m):
            c=candidates if ell==0 else candidates[~jobs.dedicated[candidates]]
            if not len(c):
                continue
            travel=np.linalg.norm(jobs.a[c]-jobs.b[queues[ell][-1]],axis=1)/(speed*ratio) if queues[ell] else np.zeros(len(c))
            finish=load[ell]+travel+jobs.p[c]
            # Round only comparison keys, not scan or travel durations. Prevent
            # machine-epsilon ties in equal hatch lengths from choosing routes.
            order=np.lexsort((jobs.ids[c],np.round(finish/scale,10)))
            k=int(order[0]);j=int(c[k])
            item=(round(float(finish[k]/scale),10),int(jobs.ids[j]),ell,float(finish[k]),j)
            if best is None or item[:3]<best[:3]:
                best=item
        _,_,ell,finish,j=best
        queues[ell].append(j);load[ell]=finish;remaining[j]=False
    return queues


def portfolio(jobs,m):
    _,assn,lptq=lpt(jobs,m)
    rounded=np.round(jobs.p/np.median(jobs.p),10)
    plans={
        "LPT_raw":lptq,
        "LPT_tie_stable":list_queues(jobs,m,np.lexsort((jobs.ids,-rounded))),
        "SPT_tie_stable":list_queues(jobs,m,np.lexsort((jobs.ids,rounded))),
        "file_load_balance":list_queues(jobs,m,np.argsort(jobs.ids)),
        "LPT_nearest_neighbor":ordered_queues(jobs,assn,lptq,"nearest_neighbor"),
    }
    for axis,label in [(0,"x"),(1,"y")]:
        q=spatial_queues(jobs,m,axis)
        plans[f"region_{label}_file"]=q
        plans[f"region_{label}_nearest_neighbor"]=ordered_queues(jobs,assignment_of(q,len(jobs.p)),q,"nearest_neighbor")
    for ratio in [5.,10.,20.,math.inf]:
        label="inf" if math.isinf(ratio) else str(int(ratio))
        plans["append_EFT_"+label]=append_eft(jobs,m,ratio)
    for q in plans.values():
        assignment=assignment_of(q,len(jobs.p))
        assert np.all(assignment[jobs.dedicated]==0)
    return plans


def free_starts(jobs,queues,setup):
    starts=np.zeros(len(jobs.p))
    for q in queues:
        if q:
            starts[q]=np.cumsum(jobs.p[q]+setup[q])-jobs.p[q]
    return starts


def mathematical_checks():
    single=hatch(sections()["rectangle"],0)
    assert abs(np.linalg.norm(single.b-single.a,axis=1).sum()*.8-36*24)<1e-8
    tiny=Jobs(np.arange(3),np.array([[0.,0.],[3.,0.],[8.,0.]]),
              np.array([[1.,0.],[4.,0.],[10.,0.]]),np.array([1.,1.,2.]),np.zeros(3,bool))
    setup,_,_=setup_times(tiny,[[0,1,2]],2.,.1)
    np.testing.assert_allclose(free_starts(tiny,[[0,1,2]],setup),[0,2.1,5.2])
    # Exhaustive tiny two-machine queue enumeration: confirm the separable
    # chain-load formula against an independent sum of edges for every plan.
    import itertools
    maximum_error=0.;count=0
    for permutation in itertools.permutations(range(3)):
        for cut in range(4):
            queues=[list(permutation[:cut]),list(permutation[cut:])]
            setup,_,_=setup_times(tiny,queues,2.,0.)
            starts=free_starts(tiny,queues,setup)
            direct=max(sum(tiny.p[q])+sum(np.linalg.norm(tiny.b[i]-tiny.a[j])/2 for i,j in zip(q[:-1],q[1:])) for q in queues)
            maximum_error=max(maximum_error,abs(max(starts+tiny.p)-direct));count+=1
    assert maximum_error<1e-12
    return {"rectangle_hatch_area_check":"PASS","tiny_enumerated_plans":count,
            "chain_time_max_error":maximum_error,"segment_interior_checks":"three interior locations in every generated vector"}


def run():
    protocol={
        "question":"Does omission of laser-off travel change which plan is selected from an identical conventional portfolio, under an otherwise identical objective and constraints?",
        "generated_families":list(sections()),"hatch_angles_degrees":[0.,30.,60.],
        "hatch_spacing":.8,"max_scan_vector_length":5.,"generated_scan_speed":600.,
        "geometry_status":"Analytically defined polygon sections, no external CAD or physical validation; holes use even-odd clipping.",
        "source_geometries":["Layer 10","Nozzle"],"source_url":"https://zenodo.org/records/12747089",
        "lasers":[2,4],"separation_radii":[0.,8.],"jump_ratios_slow_to_fast":[5.,10.,20.,None],
        "None_ratio":"infinite jump speed","fixed_transition_delay":0.,
        "candidates":["LPT_raw","LPT_tie_stable","SPT_tie_stable","file_load_balance","LPT_nearest_neighbor",
                      "region_x_file","region_x_nearest_neighbor","region_y_file","region_y_nearest_neighbor",
                      "append_EFT_5","append_EFT_10","append_EFT_20","append_EFT_inf"],
        "same_portfolio":"All thirteen plans available to both scan-only and execution-aware selectors, including all four EFT parameterizations. No target outcome or physical labels.",
        "retiming":"Minimum makespan of whole-job-wait, continuous-release, and feasible slower-jump incumbent. This is a heuristic upper bound, not an optimal retiming claim.",
        "zero_radius":"Exact chain-load timing for each fixed queue; no separation constraint.",
        "nonzero_radius":"Same illustrative beam-on separation model as prior work, not a calibrated quality/safety requirement.",
        "selection_ties":"Declared candidate order, with duration comparison tolerance 1e-10 relative to median scan duration.",
        "global_wall_budget_s":900.,"memory_budget_MB":512,"compute_threads":1,
        "scope":"Single layer, fixed directed vectors, prepositioned first vector, no final park, acceleration, recoating, thermal history, machine timing or material-quality validation.",
        "independence":"Twelve deliberately constructed geometry-angle cases plus two previously studied source layers; parameter combinations are not independent manufacturing replicates.",
        "novel_algorithm_claim":False,"parameters_frozen_before_first_performance_run":True,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT/"geometry_portfolio_protocol.json").write_text(json.dumps(protocol,indent=2))
    begin=time.perf_counter();deadline=begin+protocol["global_wall_budget_s"]
    output={"protocol":protocol,"checks":mathematical_checks(),"geometry":[],"rows":[],"decisions":[],"complete":False}
    cases=[];stored={};plans_store={}
    for name,polygons in sections().items():
        for angle in protocol["hatch_angles_degrees"]:
            label=f"{name}_{int(angle)}"
            jobs=hatch(polygons,angle)
            cases.append((label,jobs,"controlled_polygon"))
            for field in ["a","b","p","dedicated"]:
                stored[label+"_"+field]=getattr(jobs,field)
            for k,p in enumerate(polygons):
                stored[label+f"_polygon_{k}"]=p
    np.savez_compressed(ROOT/"controlled_scan_geometries.npz",**stored)
    for name in protocol["source_geometries"]:
        cases.append((name,read_jobs(ROOT/"inputs"/(name+" - SharpCornerCons - less5mm.xlsx")),"public_source"))
    selected_store={}
    try:
        for name,jobs,origin in cases:
            output["geometry"].append({"case":name,"origin":origin,"n":len(jobs.p),"total_scan_time":float(jobs.p.sum()),
                                      "dedicated_jobs":int(jobs.dedicated.sum()),"bounding_box":np.vstack([jobs.a.min(axis=0),jobs.a.max(axis=0),jobs.b.min(axis=0),jobs.b.max(axis=0)]).tolist()})
            for m in protocol["lasers"]:
                tick=time.perf_counter();plans=portfolio(jobs,m)
                construction_s=time.perf_counter()-tick
                plans_store[name+f"_m{m}"]=plans
                assignment={label:assignment_of(q,len(jobs.p)) for label,q in plans.items()}
                global_lb=max(float(jobs.p.sum()/m),float(jobs.p[jobs.dedicated].sum()),float(jobs.p.max()))
                for radius in protocol["separation_radii"]:
                    cache={};scenario={};candidate_best={}
                    for label,q in plans.items():
                        incumbent=None
                        for value in protocol["jump_ratios_slow_to_fast"]:
                            if time.perf_counter()>deadline:
                                raise TimeoutError("Frozen global computation budget exhausted")
                            ratio=math.inf if value is None else value
                            setup,travel,_=setup_times(jobs,q,ratio,0.)
                            t=time.perf_counter()
                            options=[]
                            if radius==0:
                                options=[("exact_chain",free_starts(jobs,q,setup))]
                            else:
                                for continuous in [False,True]:
                                    st=dispatch(jobs,q,setup,radius,continuous,cache,deadline)
                                    # Audit every new candidate, not only the chosen schedule.
                                    verify_execution(jobs,st,assignment[label],q,setup,radius)
                                    options.append(("continuous_release" if continuous else "whole_job_wait",st))
                                if incumbent is not None:
                                    options.append(("slower_jump_feasible_incumbent",incumbent))
                            policy,starts=min(options,key=lambda item:float(max(item[1]+jobs.p)))
                            metrics=verify_execution(jobs,starts,assignment[label],q,setup,radius)
                            if incumbent is not None:
                                assert metrics["makespan"]<=float(max(incumbent+jobs.p))+1e-10
                            incumbent=starts.copy()
                            row={"case":name,"origin":origin,"n":len(jobs.p),"lasers":m,"radius":radius,
                                 "jump_ratio":value,"plan":label,"retimer":policy,"wall_s":time.perf_counter()-t,
                                 "plan_construction_s_all13":construction_s,"total_jump_distance":float(travel.sum()),
                                 "global_scan_workload_lower_bound":global_lb,
                                 "global_workload_gap_pct":100*(metrics["makespan"]/global_lb-1),**metrics}
                            output["rows"].append(row)
                            scenario[(label,value)]=row
                            # Retain only the best physical-cost plan per scenario, not all large time arrays.
                            if value not in candidate_best or metrics["makespan"]<candidate_best[value][0]:
                                candidate_best[value]=(metrics["makespan"],label,starts.copy())
                    # Separate the pre-declared selection rule from performance evaluation.
                    # Nominal and execution selectors receive precisely the same plans.
                    nominal=min(plans,key=lambda label:scenario[(label,None)]["makespan"])
                    for value in protocol["jump_ratios_slow_to_fast"]:
                        best_cost,best_label,best_starts=candidate_best[value]
                        nominal_cost=scenario[(nominal,value)]["makespan"]
                        zero=scenario[(nominal,None)]["makespan"]
                        ranking_reversals=0;comparable=0
                        names=list(plans)
                        for i,a in enumerate(names):
                            for b in names[i+1:]:
                                d0=scenario[(a,None)]["makespan"]-scenario[(b,None)]["makespan"]
                                d1=scenario[(a,value)]["makespan"]-scenario[(b,value)]["makespan"]
                                tol=1e-8*float(jobs.p.sum())
                                if abs(d0)>tol and abs(d1)>tol:
                                    comparable+=1;ranking_reversals+=int(d0*d1<0)
                        output["decisions"].append({"case":name,"origin":origin,"lasers":m,"radius":radius,"jump_ratio":value,
                            "scan_only_selected_plan":nominal,"execution_selected_plan":best_label,
                            "nominal_predicted_cost":zero,"nominal_plan_actual_cost":nominal_cost,"portfolio_best_actual_cost":best_cost,
                            "selection_regret_pct":max(0.,100*(nominal_cost/best_cost-1)),
                            "plan_changed":nominal!=best_label,"pairwise_ranking_reversals":ranking_reversals,
                            "strict_comparable_plan_pairs":comparable,
                            "best_plan_fixed_order_bound_gap_pct":scenario[(best_label,value)]["upper_bound_gap_pct"],
                            "best_global_workload_gap_pct":scenario[(best_label,value)]["global_workload_gap_pct"]})
                        # Main finite-ratio comparison and zero-cost reference retained for replay.
                        if value in [10.,None]:
                            key=name.replace(" ","_")+f"_m{m}_r{int(radius)}_k{value}"
                            selected_store[key+"_starts"]=best_starts
                            selected_store[key+"_assignment"]=assignment[best_label]
                print(json.dumps({"case":name,"lasers":m,"vectors":len(jobs.p),"rows":len(output["rows"]),"elapsed_s":time.perf_counter()-begin}),flush=True)
                (ROOT/"geometry_portfolio_results.json").write_text(json.dumps(output,indent=2))
                (ROOT/"geometry_portfolio_plans.json").write_text(json.dumps(plans_store,separators=(",",":")))
        output["complete"]=len(output["rows"])==len(cases)*2*2*4*13
    except Exception as exc:
        output["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        output["total_wall_s"]=time.perf_counter()-begin
        (ROOT/"geometry_portfolio_results.json").write_text(json.dumps(output,indent=2))
        (ROOT/"geometry_portfolio_plans.json").write_text(json.dumps(plans_store,separators=(",",":")))
        np.savez_compressed(ROOT/"geometry_portfolio_selected_schedules.npz",**selected_store)
        for key in ["rows","decisions"]:
            if output[key]:
                fields=[k for k,v in output[key][0].items() if not isinstance(v,list)]
                with (ROOT/f"geometry_portfolio_{key}.csv").open("w") as f:
                    writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
                    writer.writeheader();writer.writerows(output[key])
    print("COMPLETED",output["complete"],len(output["rows"]),output["total_wall_s"],flush=True)


if __name__=="__main__":
    run()
