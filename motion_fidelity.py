"""Finite-portfolio decision bounds and feasible laser-off transition timing.

Conventional minimax regret, trapezoidal motion and brake/ramp transitions are
NOT new algorithms. No measured scanner calibration or thermal-quality claim.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from pilot import ROOT, Jobs, np, read_jobs
from execution_cost import dispatch, setup_times, verify_execution
from geometry_portfolio import assignment_of, free_starts


def rest_move_time(distance, speed_cap, acceleration):
    distance=np.asarray(distance,float)
    critical=speed_cap**2/acceleration
    return np.where(distance<=critical,2*np.sqrt(distance/acceleration),distance/speed_cap+speed_cap/acceleration)


def motion_setup(jobs,queues,kappa,alpha):
    velocity=(jobs.b-jobs.a)/jobs.p[:,None]
    speeds=np.linalg.norm(velocity,axis=1)
    scan_speed=float(np.median(speeds));reference_length=scan_speed*float(np.median(jobs.p))
    cap=kappa*scan_speed
    result=np.zeros(len(jobs.p));edge_upper=np.zeros_like(result)
    if math.isinf(alpha):
        return setup_times(jobs,queues,kappa,0.)[0],edge_upper
    acceleration=alpha*scan_speed**2/reference_length
    for q in queues:
        if len(q)<2:
            continue
        prev,nxt=np.array(q[:-1]),np.array(q[1:])
        vi,vj=velocity[prev],velocity[nxt]
        si,sj=speeds[prev],speeds[nxt]
        displacement=jobs.a[nxt]-jobs.b[prev]
        direct=np.linalg.norm(displacement,axis=1)
        continuous=(direct<1e-9)&(np.linalg.norm(vi-vj,axis=1)<1e-8*scan_speed)
        # Brake along the outgoing scan velocity, make a rest-to-rest move,
        # then accelerate along the incoming scan direction. Exposure vectors
        # and exposure durations remain unchanged, and velocities are continuous.
        rest0=jobs.b[prev]+vi*(si/(2*acceleration))[:,None]
        rest1=jobs.a[nxt]-vj*(sj/(2*acceleration))[:,None]
        gap=np.linalg.norm(rest1-rest0,axis=1)
        dt=(si+sj)/acceleration+rest_move_time(gap,cap,acceleration)
        dt[continuous]=0.
        upper=(si+sj)/acceleration+(si**2+sj**2)/(2*acceleration*cap)+cap/acceleration
        upper[continuous]=0.
        result[nxt]=dt;edge_upper[nxt]=upper
        # The direct constant-speed jump is a lower bound for this feasible
        # trajectory; a triangle inequality gives the stated overhead bound.
        assert np.min(dt-direct/cap)>-1e-10
        assert np.max(dt-direct/cap-upper)<1e-10
    return result,edge_upper


def motion_phases(end,velocity0,start,velocity1,cap,acceleration):
    """Independent phase representation used to verify the time formula."""
    phases=[]
    s0=float(np.linalg.norm(velocity0));s1=float(np.linalg.norm(velocity1))
    if np.linalg.norm(start-end)<1e-9 and np.linalg.norm(velocity0-velocity1)<1e-8:
        return phases
    tb=s0/acceleration;tr=s1/acceleration
    p0=end+velocity0*tb/2;p1=start-velocity1*tr/2
    phases.append((tb,-velocity0/tb))
    d=float(np.linalg.norm(p1-p0))
    if d>1e-15:
        direction=(p1-p0)/d
        peak=min(cap,math.sqrt(d*acceleration))
        ta=peak/acceleration;tc=max(0.,(d-peak**2/acceleration)/peak)
        phases.extend([(ta,direction*acceleration),(tc,np.zeros(2)),(ta,-direction*acceleration)])
    phases.append((tr,velocity1/tr))
    return phases


def numerical_checks():
    np.testing.assert_allclose(rest_move_time(np.array([0.,1.,4.,8.]),2.,1.),[0.,2.,4.,6.])
    rng=np.random.default_rng(17391)
    max_position_error=max_velocity_error=max_time_error=0.
    for trial in range(200):
        points=rng.normal(size=(4,2))*10
        v0=points[1]-points[0];v0=v0/np.linalg.norm(v0)*2
        v1=points[3]-points[2];v1=v1/np.linalg.norm(v1)*2
        if trial==0:
            points[2]=points[1];v1=v0.copy()
        jobs=Jobs(np.arange(2),np.array([points[1]-v0,points[2]]),np.array([points[1],points[2]+v1]),np.ones(2),np.zeros(2,bool))
        kappa=float(rng.choice([5.,10.,20.]));alpha=float(rng.choice([10.,100.,1000.]))
        a=alpha*2.
        setup,_=motion_setup(jobs,[[0,1]],kappa,alpha)
        phases=motion_phases(points[1],v0,points[2],v1,2*kappa,a)
        position=points[1].copy();velocity=v0.copy();duration=0.
        for dt,acc in phases:
            assert np.linalg.norm(acc)<=a+1e-10
            position+=velocity*dt+.5*acc*dt**2;velocity+=acc*dt;duration+=dt
            assert np.linalg.norm(velocity)<=2*kappa+1e-10
        max_position_error=max(max_position_error,float(np.linalg.norm(position-points[2])))
        max_velocity_error=max(max_velocity_error,float(np.linalg.norm(velocity-v1)))
        max_time_error=max(max_time_error,abs(duration-setup[1]))
    assert max_position_error<1e-9 and max_velocity_error<1e-9 and max_time_error<1e-10
    return {"motion_cases":200,"rest_motion_closed_forms":"PASS","max_endpoint_position_error":max_position_error,
            "max_endpoint_velocity_error":max_velocity_error,"max_duration_error":max_time_error}


def robust_constant_speed():
    old=json.loads((ROOT/"execution_cost_envelopes.json").read_text())
    result=[];curves={}
    for name,detail in old["details"].items():
        a,b=np.array(detail["A_scan"]),np.array(detail["B_jump"])
        names=detail["plan_names"]
        # The ratio of two affine functions has a constant derivative sign on
        # each cell; worst relative regret is attained at a cell endpoint.
        points={.05,.2}
        for cell in detail["cells"]:
            for q in [cell["q_low"],cell["q_high"]]:
                if .05<q<.2:
                    points.add(q)
        points=np.array(sorted(points))
        values=(a[:,:,None]+b[:,:,None]*points[None,None,:]).max(axis=1)
        regret=values/values.min(axis=0)-1
        worst=regret.max(axis=1)
        selected=int(np.argmin(worst));idx=int(np.argmax(regret[selected]))
        row={"instance":name,"plan":names[selected],"certified_worst_regret_pct":100*float(worst[selected]),
             "worst_kappa":float(1/points[idx]),"checked_breakpoints":len(points),"all_plan_worst_regret_pct":dict(zip(names,(100*worst).tolist()))}
        # Independent dense check can refute errors, but is NOT the certificate.
        dense=np.linspace(.05,.2,2003)
        vv=(a[:,:,None]+b[:,:,None]*dense[None,None,:]).max(axis=1)
        dense_worst=(vv/vv.min(axis=0)-1).max(axis=1)
        assert np.all(dense_worst<=worst+1e-10)
        result.append(row)
        curves[name]={"q":points.tolist(),"regret_selected_pct":(100*regret[selected]).tolist()}
    return result,curves


def resource_guard(deadline,memory_cap_mb):
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb=peak/(1024**2 if sys.platform=="darwin" else 1024)
    if peak_mb>memory_cap_mb:
        raise MemoryError(f"Peak RSS {peak_mb:.1f} MB exceeded declared budget")
    if time.perf_counter()>deadline:
        raise TimeoutError("Declared computation deadline exceeded")
    return peak_mb


def run():
    inputs=["geometry_portfolio_results.json","geometry_portfolio_plans.json","controlled_scan_geometries.npz","execution_cost_envelopes.json"]
    protocol={"question":"When are decisions from zero-jump or constant-speed timing stable under a feasible finite-acceleration laser-off motion reference?",
        "source_sha256":{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in inputs},
        "same_frozen_geometries_and_13_plans":True,"new_plan_optimization_or_training":False,
        "continuous_speed_ratio_interval":[5.,20.],"continuous_certificate_scope":"Exact finite-portfolio relative-regret maximum with constant-speed jumps and NO separation waiting. Finite precision, not a new minimax theorem or global schedule optimum.",
        "no_spacing_grid":{"kappa":[5.,10.,20.],"alpha":[10.,100.,1000.,None]},
        "spacing_grid":{"radius":8.,"kappa":[10.],"alpha":[10.,100.,1000.,None]},
        "None_alpha":"infinite acceleration, original constant-speed jump model",
        "alpha_definition":"a_max * (v_scan * median(scan_duration)) / v_scan**2",
        "parameter_status":"Dimensionless, predeclared sensitivity settings; not a calibrated plausible range for any particular physical scanner.",
        "motion_reference":"Brake outgoing exposure velocity to rest; straight rest-to-rest bounded-speed bounded-acceleration connection; ramp to incoming exposure velocity. Laser off throughout transition. Continuous collinear marked joins have zero transition cost.",
        "optimality_of_motion":"Constructively feasible brake/ramp trajectory; NOT necessarily the minimum-time transition and NOT an OEM controller reproduction.",
        "physics_scope":"Exposure durations/trajectories fixed. Velocity continuous across exposure endpoints. Jerk unbounded at phase changes; no thermal, settling, field-limit, first-positioning, final-parking or recoating model.",
        "settling_delay":0.,
        "sources":[{"url":"https://utw10945.utweb.utexas.edu/sites/default/files/2017/Manuscripts/TimeOptimalScanPathPlanningBasedonAnalysis.pdf","use":"Full text sections 2/4: known jump dynamics, endpoint velocity and acceleration effects; not a calibration source for these layers."},
                   {"url":"https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=921536","use":"NIST scanner dynamics and laser/path control study. Different testbed; no transfer of its parameter values to these data."},
                   {"url":"https://doi.org/10.1016/j.addma.2025.104694","use":"Publisher abstract: measured scanner variability on four machines. Supports need for disclosed timing assumptions, not our numerical parameter ranges."}],
        "tie_rule":"Optimistic full-reference tie breaking among nominal near-equal minima within 1e-8 total scan time; same rule as previous stage.",
        "retiming":"Best of whole-job waiting and continuous release; also retain already audited starts if setup constraints remain feasible. Finite candidates, not optimal retiming.",
        "budget":{"wall_s":900.,"rss_MB":512.,"memory_check":"ru_maxrss guard between schedules","threads":1},
        "frozen_before_first_run":True,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/"motion_fidelity_protocol.json").write_text(json.dumps(protocol,indent=2))
    begin=time.perf_counter();deadline=begin+900
    robust,curves=robust_constant_speed()
    old=json.loads((ROOT/"geometry_portfolio_results.json").read_text())
    allplans=json.loads((ROOT/"geometry_portfolio_plans.json").read_text())
    arrays=np.load(ROOT/"controlled_scan_geometries.npz")
    output={"protocol":protocol,"checks":numerical_checks(),"constant_speed_continuous_certificates":robust,
            "rows":[],"comparisons":[],"complete":False}
    saved={};raw_count=0;max_rss=0.
    try:
        for geo in old["geometry"]:
            name=geo["case"]
            if geo["origin"]=="public_source":
                jobs=read_jobs(ROOT/"inputs"/(name+" - SharpCornerCons - less5mm.xlsx"))
            else:
                fields=[arrays[name+"_"+k] for k in ["a","b","p","dedicated"]]
                jobs=Jobs(np.arange(len(fields[2])),*fields)
            for m in [2,4]:
                plans=allplans[name+f"_m{m}"];labels=list(plans)
                assn={k:assignment_of(v,len(jobs.p)) for k,v in plans.items()}
                rr=next(r for r in robust if r["instance"]==name+f"_m{m}")
                for radius,kappas in [(0.,[5.,10.,20.]),(8.,[10.])]:
                    for kappa in kappas:
                        table={};cache={};best_starts={}
                        for label,q in plans.items():
                            pool=[]
                            for alpha_value in [10.,100.,1000.,None]:
                                max_rss=max(max_rss,resource_guard(deadline,512))
                                alpha=math.inf if alpha_value is None else alpha_value
                                setup,edge_bound=motion_setup(jobs,q,kappa,alpha)
                                tick=time.perf_counter()
                                if radius==0:
                                    options=[("exact_chain",free_starts(jobs,q,setup))]
                                else:
                                    options=[]
                                    for continuous in [False,True]:
                                        st=dispatch(jobs,q,setup,radius,continuous,cache,deadline)
                                        verify_execution(jobs,st,assn[label],q,setup,radius)
                                        options.append(("continuous_release" if continuous else "whole_job_wait",st))
                                        raw_count+=1
                                    for previous in pool:
                                        if all(np.all(previous[np.array(qq[1:],int)]-previous[np.array(qq[:-1],int)]-jobs.p[np.array(qq[:-1],int)]-setup[np.array(qq[1:],int)]>=-1e-10) for qq in q):
                                            options.append(("previous_feasible_trajectory",previous))
                                policy,starts=min(options,key=lambda item:float(max(item[1]+jobs.p)))
                                metrics=verify_execution(jobs,starts,assn[label],q,setup,radius)
                                pool.append(starts.copy())
                                if alpha_value not in best_starts or metrics["makespan"]<best_starts[alpha_value][0]:
                                    best_starts[alpha_value]=(metrics["makespan"],label,starts.copy())
                                ideal,_=motion_setup(jobs,q,kappa,math.inf)
                                upper_extra=max(sum(edge_bound[qq]) for qq in q)
                                if radius==0:
                                    cv=float(max(free_starts(jobs,q,ideal)+jobs.p))
                                    assert metrics["makespan"]<=cv+upper_extra+1e-8
                                row={"case":name,"origin":geo["origin"],"lasers":m,"radius":radius,"kappa":kappa,"alpha":alpha_value,
                                     "plan":label,"retimer":policy,"wall_s":time.perf_counter()-tick,
                                     "sum_setup":float(setup.sum()),"sum_ideal_setup":float(ideal.sum()),
                                     "no_spacing_overhead_bound":float(upper_extra),**metrics}
                                output["rows"].append(row);table[(label,alpha_value)]=row
                        tol=1e-8*float(jobs.p.sum())
                        cv_min=min(table[(label,None)]["makespan"] for label in labels)
                        cv_choices=[label for label in labels if table[(label,None)]["makespan"]<=cv_min+tol]
                        # Preserve the prior zero-jump selector. It uses only
                        # the preceding frozen comparison, not acceleration outcomes.
                        previous=[r for r in old["rows"] if r["case"]==name and r["lasers"]==m and r["radius"]==radius and r["jump_ratio"] is None]
                        zero_min=min(r["makespan"] for r in previous)
                        zero_choices=[r["plan"] for r in previous if r["makespan"]<=zero_min+tol]
                        for av in [10.,100.,1000.]:
                            best_cost,best_label,st=best_starts[av]
                            cv_plan=min(cv_choices,key=lambda label:table[(label,av)]["makespan"])
                            zero_plan=min(zero_choices,key=lambda label:table[(label,av)]["makespan"])
                            robust_cost=table[(rr["plan"],av)]["makespan"]
                            item={"case":name,"origin":geo["origin"],"lasers":m,"radius":radius,"kappa":kappa,"alpha":av,
                                "full_reference_best_plan":best_label,"reference_cost":best_cost,
                                "constant_speed_selected_plan":cv_plan,"constant_speed_selection_regret_pct":max(0.,100*(table[(cv_plan,av)]["makespan"]/best_cost-1)),
                                "zero_jump_selected_plan":zero_plan,"zero_jump_selection_regret_pct":max(0.,100*(table[(zero_plan,av)]["makespan"]/best_cost-1)),
                                "constant_speed_minimax_plan":rr["plan"],"minimax_plan_reference_regret_pct":max(0.,100*(robust_cost/best_cost-1)),
                                "minimax_plan_applicability":"Guarantee is only for no-spacing constant-speed model; finite-acceleration or spacing rows are an out-of-model diagnostic.",
                                "constant_speed_cost_underestimate_pct":100*(table[(cv_plan,av)]["makespan"]/table[(cv_plan,None)]["makespan"]-1)}
                            output["comparisons"].append(item)
                            if radius==8 and av==100.:
                                key=name.replace(" ","_")+f"_m{m}"
                                saved[key+"_starts"]=st;saved[key+"_assignment"]=assn[best_label]
                print(json.dumps({"case":name,"lasers":m,"rows":len(output["rows"]),"elapsed_s":time.perf_counter()-begin,"peak_rss_MB":max_rss}),flush=True)
                (ROOT/"motion_fidelity_results.json").write_text(json.dumps(output,separators=(",",":")))
        output["complete"]=len(output["rows"])==14*2*4*4*13
    except Exception as exc:
        output["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        output["total_wall_s"]=time.perf_counter()-begin;output["peak_rss_MB"]=max_rss
        output["raw_separation_schedules_audited"]=raw_count
        (ROOT/"motion_fidelity_results.json").write_text(json.dumps(output,separators=(",",":")))
        (ROOT/"constant_speed_regret_curves.json").write_text(json.dumps(curves,separators=(",",":")))
        np.savez_compressed(ROOT/"motion_fidelity_selected_schedules.npz",**saved)
        if output["comparisons"]:
            with (ROOT/"motion_fidelity_comparisons.csv").open("w") as f:
                writer=csv.DictWriter(f,fieldnames=list(output["comparisons"][0]));writer.writeheader();writer.writerows(output["comparisons"])
    print("COMPLETE",output["complete"],len(output["rows"]),output["total_wall_s"],flush=True)


if __name__=="__main__":
    run()
