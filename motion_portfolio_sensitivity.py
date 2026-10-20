"""Same-case sensitivity to transition family and a larger conventional portfolio.

Cubic Hermite interpolation and motion-cost nearest neighbours are standard
constructions, not proposed algorithms. No fitted hardware parameters. The new
motion comparison uses r=0 to remove all separation-retimer uncertainty.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from pilot import ROOT, Jobs, read_jobs
from geometry_portfolio import assignment_of, free_starts
from motion_fidelity import motion_setup, rest_move_time, resource_guard


def motion_parameters(jobs, kappa, alpha):
    velocity = (jobs.b - jobs.a) / jobs.p[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    vs = float(np.median(speed))
    acceleration = alpha * vs / float(np.median(jobs.p))
    return velocity, speed, vs * kappa, acceleration


def brake_edges(jobs, i, j, kappa, alpha):
    """Vectorised form of the unchanged brake/rest/ramp kernel."""
    velocity, speed, cap, acceleration = motion_parameters(jobs, kappa, alpha)
    vi, vj = velocity[i], velocity[j]
    si, sj = speed[i], speed[j]
    displacement = jobs.a[j] - jobs.b[i]
    gap = displacement - (vi * si[..., None] + vj * sj[..., None]) / (2 * acceleration)
    duration = (si + sj) / acceleration + rest_move_time(np.linalg.norm(gap, axis=-1), cap, acceleration)
    continuous = ((np.linalg.norm(displacement, axis=-1) < 1e-9)
                  & (np.linalg.norm(vi - vj, axis=-1) < 1e-8 * np.median(speed)))
    return np.where(continuous, 0., duration)


def hermite_limits(displacement, v0, v1, duration):
    """Sufficient continuous speed bound and exact acceleration maximum.

    Velocity is a quadratic Bezier curve with control vectors v0,
    3*d/T-v0-v1, v1. Its norm is bounded by the largest control norm.
    Acceleration is affine, hence its norm maximum is at an endpoint.
    """
    tt = np.asarray(duration)[..., None]
    middle = 3 * displacement / tt - v0 - v1
    speed_bound = np.maximum.reduce([np.linalg.norm(v0, axis=-1), np.linalg.norm(v1, axis=-1),
                                    np.linalg.norm(middle, axis=-1)])
    a0 = 6 * displacement / tt**2 - (4 * v0 + 2 * v1) / tt
    a1 = -6 * displacement / tt**2 + (2 * v0 + 4 * v1) / tt
    acceleration_max = np.maximum(np.linalg.norm(a0, axis=-1), np.linalg.norm(a1, axis=-1))
    return speed_bound, acceleration_max


def hermite_duration(displacement, v0, v1, cap, acceleration):
    """Return a feasible cubic duration, NOT a minimum-time motion solution.

    The upper endpoint initially satisfies triangle-inequality bounds. Every
    bisection retains a verified feasible upper endpoint; no monotonicity or
    globally shortest duration is assumed. A small margin is added at the end.
    """
    d = np.linalg.norm(displacement, axis=-1)
    speed = np.maximum(np.linalg.norm(v0, axis=-1), np.linalg.norm(v1, axis=-1))
    assert np.all(cap > 2 * speed)
    speed_upper = 3 * d / (cap - 2 * speed)
    accel_upper = (6 * speed + np.sqrt(36 * speed**2 + 24 * acceleration * d)) / (2 * acceleration)
    high = np.maximum(speed_upper, accel_upper) * (1 + 1e-12)
    low = np.zeros_like(high)
    for _ in range(40):
        mid = (low + high) / 2
        vmax, amax = hermite_limits(displacement, v0, v1, mid)
        good = (vmax <= cap) & (amax <= acceleration)
        high = np.where(good, mid, high)
        low = np.where(good, low, mid)
    high *= 1 + 1e-10
    vmax, amax = hermite_limits(displacement, v0, v1, high)
    assert np.all(vmax <= cap * (1 + 1e-9)) and np.all(amax <= acceleration * (1 + 1e-9))
    assert np.all(high >= d / cap - 1e-12)
    return high


def endpoint_sample(displacement, v0, v1, duration, tau):
    """Independent power-coefficient evaluation for trajectory checks."""
    t = np.asarray(tau) * duration
    c2 = 3 * displacement / duration**2 - (2 * v0 + v1) / duration
    c3 = -2 * displacement / duration**3 + (v0 + v1) / duration**2
    x = t[..., None] * v0 + t[..., None]**2 * c2 + t[..., None]**3 * c3
    v = v0 + 2 * t[..., None] * c2 + 3 * t[..., None]**2 * c3
    a = 2 * c2 + 6 * t[..., None] * c3
    return x, v, a


def numerical_checks():
    rng = np.random.default_rng(20261017)
    max_x = max_v = 0.
    for _ in range(300):
        d = rng.normal(size=2) * 20
        v = rng.normal(size=(2, 2))
        v /= np.linalg.norm(v, axis=1)[:, None]
        cap = float(rng.choice([5, 10, 20]))
        accel = float(rng.choice([2, 20, 200]))
        T = float(hermite_duration(d[None], v[0:1], v[1:2], cap, accel)[0])
        x, vv, aa = endpoint_sample(d, v[0], v[1], T, np.linspace(0., 1., 1001))
        max_x = max(max_x, float(np.linalg.norm(x[-1] - d)))
        max_v = max(max_v, float(np.linalg.norm(vv[-1] - v[1])))
        assert np.linalg.norm(vv, axis=1).max() <= cap * (1 + 1e-8)
        assert np.linalg.norm(aa, axis=1).max() <= accel * (1 + 1e-8)
    assert max_x < 1e-10 and max_v < 1e-10
    # Rest-to-rest cubic with d=1: sufficient speed bound=3/T;
    # exact acceleration maximum=6/T^2. With a=1 the latter binds.
    duration = hermite_duration(np.array([[1., 0.]]), np.zeros((1, 2)), np.zeros((1, 2)), 5., 1.)[0]
    assert abs(duration - math.sqrt(6)) < 1e-8
    return {"independent_sampled_trajectories": 300, "samples_each": 1001,
            "analytic_continuous_limit_check": "PASS", "rest_cubic_closed_form": "PASS",
            "maximum_endpoint_position_error": max_x, "maximum_endpoint_velocity_error": max_v}


def motion_nearest_neighbor(jobs, queues, alpha, deadline):
    """Conventional directed nearest-neighbour using finite motion time."""
    result = []
    scale = float(np.median(jobs.p))
    for q in queues:
        if not q:
            result.append([])
            continue
        path = [q[0]]
        remaining = np.zeros(len(jobs.p), bool)
        remaining[q[1:]] = True
        while remaining.any():
            if len(path) % 128 == 0:
                resource_guard(deadline, 512)
            candidates = np.flatnonzero(remaining)
            dt = brake_edges(jobs, path[-1], candidates, 10., alpha)
            # Same tolerance convention used in existing plan constructors.
            pick = candidates[np.lexsort((jobs.ids[candidates], np.round(dt / scale, 10)))[0]]
            path.append(int(pick))
            remaining[pick] = False
        result.append(path)
    original_assignment = assignment_of(queues, len(jobs.p))
    np.testing.assert_array_equal(original_assignment, assignment_of(result, len(jobs.p)))
    return result


def cases():
    for stage, result_name, array_name, plan_name in [
        ("earlier", "geometry_portfolio_results.json", "controlled_scan_geometries.npz", "geometry_portfolio_plans.json"),
        ("prospective", "prospective_geometry_results.json", "prospective_geometry_inputs.npz", "prospective_geometry_plans.json")]:
        data = json.loads((ROOT / result_name).read_text())
        arrays = np.load(ROOT / array_name)
        allplans = json.loads((ROOT / plan_name).read_text())
        for geo in data["geometry"]:
            name = geo["case"]
            if geo.get("origin") == "public_source":
                jobs = read_jobs(ROOT / "inputs" / (name + " - SharpCornerCons - less5mm.xlsx"))
            else:
                fields = [arrays[name + "_" + k] for k in ["a", "b", "p", "dedicated"]]
                jobs = Jobs(np.arange(len(fields[2])), *fields)
            for m in [2, 4]:
                yield stage, name, m, jobs, allplans[name + f"_m{m}"]


def groups(labels):
    original = [k for k in labels if not k.startswith("motion_NN_")]
    return {"original_13": original, "expanded_19": list(labels),
            "original_without_EFT": [k for k in original if not k.startswith("append_EFT")],
            "original_without_x_partition": [k for k in original if not k.startswith("region_x")],
            "original_without_y_partition": [k for k in original if not k.startswith("region_y")],
            "original_without_list_family": [k for k in original if k.startswith(("region_", "append_EFT"))]}


def main():
    if "--checks" in sys.argv:
        print(json.dumps(numerical_checks(), indent=2)); return
    inputs = ["motion_fidelity.py", "geometry_portfolio.py", "prospective_geometry.py",
              "geometry_portfolio_results.json", "controlled_scan_geometries.npz", "geometry_portfolio_plans.json",
              "prospective_geometry_results.json", "prospective_geometry_inputs.npz", "prospective_geometry_plans.json"]
    protocol = {"question": "Do same-portfolio decisions remain sensitive when full stops are optional and conventional motion-aware plans are added?",
                "cases": "All 26 already specified geometry-angle cases, both 2 and 4 lasers; no exclusions or new geometry selection.",
                "radius": 0, "reason_for_zero_radius": "Exact per-laser chain costs remove any separation-retimer effect. No robustness claim for r>0 follows.",
                "kappa": [5, 10, 20], "alpha": [10, 100, 1000],
                "motion_models": ["original brake/rest/ramp", "edgewise faster of original and feasible cubic Hermite"],
                "cubic": "Endpoint position/velocity preserved; affine acceleration certified at endpoints; quadratic Bezier velocity norm bounded by control norms. Feasible duration via 40-step bisection retaining a certified upper endpoint, not global time optimization.",
                "cubic_scope": "Standard polynomial interpolation, unbounded planar field, no jerk or settling; model sensitivity rather than machine calibration. Laser off throughout.",
                "continuous_joins": "Zero transition only for coincident endpoints and equal velocities, inherited original tolerances. No forced waits at r=0.",
                "new_plans": "Six standard nearest-neighbour reorderings: existing stable LPT/x/y assignments, each using original finite-motion edge times at kappa=10 and alpha=10 or 100. Same first vector and fixed assignments; all six supplied to both model selectors at every evaluation setting.",
                "subsets": "Original13, expanded19, original without EFT/x-partition/y-partition/list family. Reduced subsets are sensitivity checks, never evidence of superiority over stronger plans.",
                "tie_rule": "Same optimistic reference-best tie break within 1e-8 sum of exposure times; diagnostic, not deployable.",
                "comparisons": "Every case, parameter, motion reference and predefined portfolio. Counts are deterministic scenarios, not independent manufacturing repetitions.",
                "budget": {"wall_s": 900, "rss_MB": 512, "threads": 1},
                "sources": ["https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-3-polynomial-via-point-trajectories/",
                            "https://utw10945.utweb.utexas.edu/sites/default/files/2017/Manuscripts/TimeOptimalScanPathPlanningBasedonAnalysis.pdf"],
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fixed_before_evaluation": True}
    protocol_path = ROOT / "motion_portfolio_sensitivity_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Do not change a frozen protocol"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    begin = time.perf_counter(); deadline = begin + protocol["budget"]["wall_s"]
    output = {"protocol": protocol, "checks": numerical_checks(), "rows": [], "comparisons": [],
              "configurations": [], "complete": False}
    added_plans = {}; peak = 0.
    try:
        for stage, name, m, jobs, originals in cases():
            plans = dict(originals)
            added = {}
            for base, short in [("LPT_tie_stable", "LPT"), ("region_x_file", "x"), ("region_y_file", "y")]:
                for alpha in [10., 100.]:
                    label = f"motion_NN_{short}_a{int(alpha)}"
                    plans[label] = motion_nearest_neighbor(jobs, originals[base], alpha, deadline)
                    added[label] = plans[label]
            added_plans[name + f"_m{m}"] = added
            labels = list(plans)
            subsets = groups(labels)
            assert len(labels) == 19 and len(subsets["original_13"]) == 13
            signatures = {tuple(tuple(q) for q in plan) for plan in plans.values()}
            velocity, speeds, _, _ = motion_parameters(jobs, 10., 10.)
            edges = sorted({(i, j) for queues in plans.values() for q in queues for i, j in zip(q[:-1], q[1:])})
            ei, ej = np.array(edges, int).T
            edge_index = {edge: idx for idx, edge in enumerate(edges)}
            plan_edge_indices = {label: [np.array([edge_index[(i, j)] for i, j in zip(q[:-1], q[1:])], int) for q in queues]
                                 for label, queues in plans.items()}
            displacement = jobs.a[ej] - jobs.b[ei]
            continuous = ((np.linalg.norm(displacement, axis=1) < 1e-9)
                          & (np.linalg.norm(velocity[ei] - velocity[ej], axis=1) < 1e-8 * np.median(speeds)))
            tolerance = 1e-8 * float(jobs.p.sum())
            replacement_count = 0; max_original_reproduction_error = 0.
            for kappa in [5., 10., 20.]:
                peak = max(peak, resource_guard(deadline, 512))
                cv = np.linalg.norm(displacement, axis=1) / (kappa * np.median(speeds))
                table = {}
                def evaluate(edge_times, model, alpha):
                    for label, queues in plans.items():
                        loads = [float(sum(jobs.p[q]) + sum(edge_times[ix]))
                                 for q, ix in zip(queues, plan_edge_indices[label])]
                        value = max(loads)
                        table[(label, model, alpha)] = value
                        output["rows"].append({"stage": stage, "case": name, "lasers": m, "kappa": kappa,
                                               "alpha": alpha, "model": model, "plan": label, "makespan": value,
                                               "chain_loads": loads})
                evaluate(cv, "constant_speed", None)
                for alpha in [10., 100., 1000.]:
                    peak = max(peak, resource_guard(deadline, 512))
                    _, _, cap, acceleration = motion_parameters(jobs, kappa, alpha)
                    brake = brake_edges(jobs, ei, ej, kappa, alpha)
                    cubic = hermite_duration(displacement, velocity[ei], velocity[ej], cap, acceleration)
                    cubic[continuous] = 0.
                    optional = np.minimum(brake, cubic)
                    assert np.min(optional - cv) > -1e-10
                    replacement_count += int(np.sum(cubic < brake - 1e-12))
                    # Every candidate's original cost is checked by the old,
                    # separately implemented kernel and cumulative start times.
                    for label, queues in plans.items():
                        setup, _ = motion_setup(jobs, queues, kappa, alpha)
                        expected = float(max(free_starts(jobs, queues, setup) + jobs.p))
                        actual = max(float(sum(jobs.p[q]) + sum(brake[ix]))
                                     for q, ix in zip(queues, plan_edge_indices[label]))
                        max_original_reproduction_error = max(max_original_reproduction_error, abs(expected - actual))
                        assert abs(expected - actual) < 1e-9
                    for model, edge_times in [("brake_rest_ramp", brake), ("optional_cubic", optional)]:
                        evaluate(edge_times, model, alpha)
                        for subset_name, subset in subsets.items():
                            low = min(table[(k, "constant_speed", None)] for k in subset)
                            ties = [k for k in subset if table[(k, "constant_speed", None)] <= low + tolerance]
                            selected = min(ties, key=lambda k: (table[(k, model, alpha)], k))
                            best = min(subset, key=lambda k: (table[(k, model, alpha)], k))
                            selected_cost = table[(selected, model, alpha)]
                            best_cost = table[(best, model, alpha)]
                            estimate = table[(selected, "constant_speed", None)]
                            output["comparisons"].append({"stage": stage, "case": name, "lasers": m, "kappa": kappa,
                                "alpha": alpha, "model": model, "portfolio": subset_name, "candidate_count": len(subset),
                                "selected_plan": selected, "alternative_plan": best, "selected_cost": selected_cost,
                                "best_cost": best_cost, "simplified_estimate": estimate,
                                "time_bias_pct": 100 * (selected_cost / estimate - 1),
                                "selection_loss_pct": 100 * (selected_cost / best_cost - 1)})
            output["configurations"].append({"case": name, "stage": stage, "lasers": m, "vectors": len(jobs.p),
                "unique_plans_of_19": len(signatures), "unique_edges": len(edges),
                "faster_cubic_edge_parameter_evaluations": replacement_count,
                "max_old_kernel_reproduction_error": max_original_reproduction_error})
            print(json.dumps({"case": name, "lasers": m, "elapsed_s": round(time.perf_counter()-begin, 2),
                              "unique_plans": len(signatures)}), flush=True)
        assert len(output["configurations"]) == 52 and len(output["rows"]) == 20748
        assert len(output["comparisons"]) == 5616
        output["complete"] = True
    except Exception as exc:
        output["error"] = repr(exc)
        raise
    finally:
        output["wall_s"] = time.perf_counter() - begin
        output["peak_rss_MB"] = peak
        (ROOT / "motion_portfolio_sensitivity_results.json").write_text(json.dumps(output, indent=2))
        (ROOT / "motion_portfolio_added_plans.json").write_text(json.dumps(added_plans))
        if output["comparisons"]:
            with (ROOT / "motion_portfolio_sensitivity_comparisons.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(output["comparisons"][0]))
                writer.writeheader(); writer.writerows(output["comparisons"])


if __name__ == "__main__":
    main()
