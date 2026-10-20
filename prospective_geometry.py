"""Prospectively fixed geometric extension of the execution-model study.

No tuning, new optimizer, learned model, downloaded data or physical experiment.
Generated families are new computational cases, not external validation data.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from pilot import ROOT
from geometry_portfolio import rectangle, hatch, portfolio, assignment_of, free_starts
from execution_cost import dispatch, verify_execution
from motion_fidelity import motion_setup, resource_guard


def new_sections():
    phi = np.arange(64) * (2 * np.pi / 64)
    circle = np.column_stack([np.cos(phi), np.sin(phi)])
    return {
        "U_bracket": [np.array([[-24,-20],[24,-20],[24,20],[12,20],[12,-8],[-12,-8],[-12,20],[-24,20]], float)],
        "dogbone": [np.array([[-30,-14],[-16,-14],[-8,-5],[8,-5],[16,-14],[30,-14],[30,14],[16,14],[8,5],[-8,5],[-16,14],[-30,14]], float)],
        "perforated_plate": [rectangle(-28,-20,56,40)] + [5.5*circle + [x,0] for x in [-16,0,16]],
        "unequal_islands": [rectangle(-32,-22,12,28), rectangle(-10,-15,22,12),
                            rectangle(10,10,18,11), np.array([[16,-25],[31,-22],[25,-5],[14,-9]],float)]}


def continuous_bound(jobs, plans):
    labels = list(plans)
    speed = float(np.median(np.linalg.norm(jobs.b-jobs.a,axis=1)/jobs.p))
    a = np.array([[sum(jobs.p[q]) for q in plans[k]] for k in labels])
    b = np.array([[sum(np.linalg.norm(jobs.b[i]-jobs.a[j])/speed for i,j in zip(q[:-1],q[1:]))
                   for q in plans[k]] for k in labels])
    aa, bb = a.ravel(), b.ravel()
    points = [.05, .2]
    for i in range(len(aa)):
        den = bb[i]-bb[i+1:]
        good = np.abs(den)>1e-13
        q = (aa[i+1:][good]-aa[i])/den[good]
        points.extend(q[(q>.05)&(q<.2)].tolist())
    points = np.array(sorted(set(points)))
    times = (a[:,:,None]+b[:,:,None]*points[None,None,:]).max(axis=1)
    regret = times/times.min(axis=0)-1
    worst = regret.max(axis=1); selected = int(np.argmin(worst))
    dense = np.linspace(.05,.2,2003)
    times_dense = (a[:,:,None]+b[:,:,None]*dense[None,None,:]).max(axis=1)
    assert np.all((times_dense/times_dense.min(axis=0)-1).max(axis=1)<=worst+1e-10)
    return {"selected_plan":labels[selected],"worst_regret_pct":float(100*worst[selected]),
            "checked_breakpoints":len(points),"all_plan_worst_regret_pct":dict(zip(labels,(100*worst).tolist())),
            "scope":"Finite-portfolio constant-speed model, kappa in [5,20], no beam separation. Not a global optimum or physical guarantee."}


def feasible_setup(starts,jobs,queues,setup):
    for q in queues:
        if len(q)>1 and np.min(starts[q[1:]]-starts[q[:-1]]-jobs.p[q[:-1]]-setup[q[1:]]) < -1e-10:
            return False
    return True


def run():
    deps = ["geometry_portfolio.py","motion_fidelity.py","execution_cost.py","pilot.py"]
    protocol = {
        "purpose":"Prospective verification of the same computational modeling question, without method changes.",
        "families":{k:[p.tolist() for p in v] for k,v in new_sections().items()},
        "angles_degrees":[15,45,75],"lasers":[2,4],"hatch_spacing":.8,"max_vector_length":5.,"scan_speed":600.,
        "settings":[{"radius":0.,"kappa":[5.,10.,20.]},{"radius":8.,"kappa":[10.]}],
        "alpha":[10.,100.,1000.,None],"None_alpha":"infinite acceleration",
        "portfolio":"Same 13 conventional constructors; source hashes below. All selectors have identical candidates.",
        "retiming":"Same two retimers; retain any already computed feasible slower-model starts. No new optimization.",
        "tie_rule":"Optimistic finite-reference tie break within 1e-8 total scan duration, as before.",
        "independence":"New polygon families and angles fixed before performance evaluation. Same generator and modeling assumptions; not external CAD, machines or physical replicates.",
        "checks_to_report":["Timing bias versus selected-plan loss on every fixed condition", "Positive selected-order lower bound versus feasible alternative, including absent witnesses", "Continuous-speed finite-portfolio bounds on all new configurations, without assuming the prior 1.35% maximum transfers"],
        "thresholds":"1% and 5% are descriptive effect sizes, not industrial acceptance limits.",
        "no_exclusions":"Retain every predefined case. Abort/incomplete if time or memory budget prevents completion.",
        "parameter_status":"Dimensionless sensitivity settings, not calibrated devices. alpha uses the same median-vector-length reference as the preceding stage.",
        "motion_scope":"Same endpoint-velocity-preserving brake/rest-to-rest/ramp model; not time-optimal. No jerk, field limits, settling, initial positioning, parking, thermal effects or quality predictions.",
        "budget":{"wall_s":600.,"rss_MB":512.,"threads":1},
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in deps},
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_before_first_evaluation":True}
    protocol_path=ROOT/"prospective_geometry_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol, "Do not overwrite a frozen protocol with changed settings"
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2))
    begin=time.perf_counter();deadline=begin+600
    out={"protocol":protocol,"geometry":[],"rows":[],"comparisons":[],"continuous_bounds":[],"complete":False}
    stored={}; saved={}; plans_out={}; peak=0.; audits=0
    try:
        for family,polys in new_sections().items():
            for angle in protocol["angles_degrees"]:
                name=f"{family}_{angle}"
                jobs=hatch(polys,angle)
                out["geometry"].append({"case":name,"family":family,"angle":angle,"vectors":len(jobs.p),
                                        "total_scan_time":float(jobs.p.sum()),"median_vector_length":float(np.median(np.linalg.norm(jobs.b-jobs.a,axis=1)))})
                for key in ["a","b","p","dedicated"]: stored[name+"_"+key]=getattr(jobs,key)
                for i,p in enumerate(polys): stored[name+f"_polygon_{i}"]=p
                for m in protocol["lasers"]:
                    peak=max(peak,resource_guard(deadline,512))
                    plans=portfolio(jobs,m); labels=list(plans)
                    plans_out[name+f"_m{m}"]=plans
                    bound=continuous_bound(jobs,plans)
                    out["continuous_bounds"].append({"case":name,"lasers":m,**bound})
                    assn={k:assignment_of(q,len(jobs.p)) for k,q in plans.items()}
                    tol=1e-8*float(jobs.p.sum())
                    for setting in protocol["settings"]:
                        radius=setting["radius"]
                        for kappa in setting["kappa"]:
                            cache={};table={};all_starts={}
                            for label,q in plans.items():
                                prior=[]
                                for av in protocol["alpha"]:
                                    peak=max(peak,resource_guard(deadline,512))
                                    setup,_=motion_setup(jobs,q,kappa,math.inf if av is None else av)
                                    if radius==0:
                                        options=[("exact_chain",free_starts(jobs,q,setup))]
                                    else:
                                        options=[]
                                        for continuous in [False,True]:
                                            st=dispatch(jobs,q,setup,radius,continuous,cache,deadline)
                                            verify_execution(jobs,st,assn[label],q,setup,radius);audits+=1
                                            options.append(("continuous_release" if continuous else "whole_job_wait",st))
                                        options.extend(("previous_feasible_starts",st) for st in prior if feasible_setup(st,jobs,q,setup))
                                    retimer,starts=min(options,key=lambda z:float(max(z[1]+jobs.p)))
                                    metrics=verify_execution(jobs,starts,assn[label],q,setup,radius)
                                    if radius==0: assert abs(metrics["makespan"]-metrics["fixed_order_lower_bound"])<1e-9
                                    row={"case":name,"family":family,"angle":angle,"lasers":m,"radius":radius,
                                         "kappa":kappa,"alpha":av,"plan":label,"retimer":retimer,**metrics}
                                    out["rows"].append(row);table[(label,av)]=row
                                    prior.append(starts.copy());all_starts[(label,av)]=starts
                            cv_min=min(table[(k,None)]["makespan"] for k in labels)
                            cv_choices=[k for k in labels if table[(k,None)]["makespan"]<=cv_min+tol]
                            for av in protocol["alpha"][:-1]:
                                best=min(labels,key=lambda k:table[(k,av)]["makespan"])
                                selected=min(cv_choices,key=lambda k:table[(k,av)]["makespan"])
                                ub=table[(best,av)]["makespan"];value=table[(selected,av)]["makespan"]
                                lb=table[(selected,av)]["fixed_order_lower_bound"]
                                loss=max(0.,100*(value/ub-1))
                                witness=max(0.,100*((lb-1e-10)/ub-1))
                                c={"case":name,"family":family,"angle":angle,"lasers":m,"radius":radius,"kappa":kappa,"alpha":av,
                                   "selected_plan":selected,"alternative_plan":best,"selected_cost":value,"alternative_cost":ub,
                                   "time_extra_vs_simplified_pct":100*(value/table[(selected,None)]["makespan"]-1),
                                   "observed_selection_loss_pct":loss,"selected_order_lower_bound":lb,"selection_loss_lower_bound_pct":witness,
                                   "continuous_minimax_plan":bound["selected_plan"],
                                   "continuous_minimax_reference_loss_pct":max(0.,100*(table[(bound["selected_plan"],av)]["makespan"]/ub-1))}
                                out["comparisons"].append(c)
                                if witness>1e-6:
                                    key=f"{name}_m{m}_r{int(radius)}_k{int(kappa)}_a{int(av)}"
                                    for role,label in [("selected",selected),("alternative",best)]:
                                        saved[key+f"_{role}_starts"]=all_starts[(label,av)]
                                        saved[key+f"_{role}_assignment"]=assn[label]
                    print(json.dumps({"case":name,"lasers":m,"rows":len(out["rows"]),"elapsed_s":time.perf_counter()-begin}),flush=True)
                    (ROOT/"prospective_geometry_results.json").write_text(json.dumps(out,separators=(",",":")))
        out["complete"]=len(out["rows"])==4992 and len(out["comparisons"])==288
    except Exception as exc:
        out["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        out["wall_s"]=time.perf_counter()-begin;out["peak_rss_MB"]=peak;out["raw_separation_audits"]=audits
        (ROOT/"prospective_geometry_results.json").write_text(json.dumps(out,separators=(",",":")))
        (ROOT/"prospective_geometry_plans.json").write_text(json.dumps(plans_out,separators=(",",":")))
        np.savez_compressed(ROOT/"prospective_geometry_inputs.npz",**stored)
        np.savez_compressed(ROOT/"prospective_geometry_witnesses.npz",**saved)
        if out["comparisons"]:
            with (ROOT/"prospective_geometry_comparisons.csv").open("w") as f:
                writer=csv.DictWriter(f,fieldnames=list(out["comparisons"][0]));writer.writeheader();writer.writerows(out["comparisons"])
    print("COMPLETE",out["complete"],out["wall_s"],flush=True)


if __name__=="__main__": run()
