"""Execution-cost sensitivity on the two already acquired public scan layers.

No fitting or training. The jump/scan-speed ratios and delays are illustrative
scenario parameters, not measured hardware values. Instantaneous laser-on
separation is the same explicit mathematical constraint as the earlier pilot.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

from pilot import ROOT, Jobs, audit, closest_distance, forbidden_interval, lpt, np, read_jobs, wait_repair


def ordered_queues(jobs, assignment, original, order):
    if order == "LPT":
        return original
    result = []
    for ell, q in enumerate(original):
        if order == "file_order":
            result.append(sorted(q))
            continue
        if not q:
            result.append([])
            continue
        # Common starting vector for LPT and nearest-neighbour; orientations
        # and laser assignments are never changed.
        remaining = set(q[1:])
        path = [q[0]]
        while remaining:
            candidates = np.array(sorted(remaining))
            squared = np.sum((jobs.a[candidates] - jobs.b[path[-1]]) ** 2, axis=1)
            pick = int(candidates[np.lexsort((jobs.ids[candidates], squared))[0]])
            path.append(pick)
            remaining.remove(pick)
        result.append(path)
    assert sorted(i for q in result for i in q) == list(range(len(jobs.p)))
    return result


def setup_times(jobs, queues, jump_ratio, delay_fraction):
    scan_speed = float(np.median(np.linalg.norm(jobs.b - jobs.a, axis=1) / jobs.p))
    delay = delay_fraction * float(np.median(jobs.p))
    setup, travel = np.zeros(len(jobs.p)), np.zeros(len(jobs.p))
    for q in queues:
        if len(q) > 1:
            length = np.linalg.norm(jobs.a[q[1:]] - jobs.b[q[:-1]], axis=1)
            travel[q[1:]] = length
            setup[q[1:]] = length / (scan_speed * jump_ratio) + delay
    return setup, travel, scan_speed


def dispatch(jobs, queues, setup, radius, continuous_release, cache, deadline):
    """Standard event-based dispatch with optional within-vector release.

    Setup is laser-off; it may overlap other lasers' scans. A laser is assumed
    pre-positioned at the first vector, so initial and final travel are excluded.
    """
    starts = np.zeros(len(jobs.p))
    active = [None] * len(queues)
    cursor = np.zeros(len(queues), int)
    ready = np.array([setup[q[0]] if q else math.inf for q in queues])
    t, assigned, steps = 0., 0, 0
    while assigned < len(jobs.p):
        steps += 1
        if steps % 128 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError("Declared computation budget exhausted")
        for ell, job in enumerate(active):
            if job is not None and starts[job] + jobs.p[job] <= t + 1e-12:
                active[ell] = None
        releases = []
        for ell, q in enumerate(queues):
            if active[ell] is not None or cursor[ell] == len(q) or ready[ell] > t + 1e-12:
                continue
            j = q[cursor[ell]]
            proposed = t
            while True:
                blocked = False
                for i in active:
                    if i is None or closest_distance(jobs, j, i, proposed, starts[i]) >= radius - 1e-9:
                        continue
                    blocked = True
                    if continuous_release:
                        key = (i, j)
                        if key not in cache:
                            cache[key] = forbidden_interval(jobs.a[i], jobs.b[i], jobs.p[i],
                                                            jobs.a[j], jobs.b[j], jobs.p[j], radius)
                        interval = cache[key]
                        assert interval is not None
                        new = starts[i] + interval[1]
                        if new <= proposed + 1e-13:
                            raise ArithmeticError("Non-forward conflict release")
                        proposed = new
                    break
                if not blocked or not continuous_release:
                    break
            if proposed <= t + 1e-12 and not blocked:
                starts[j], active[ell] = t, j
                cursor[ell] += 1
                assigned += 1
                ready[ell] = t + jobs.p[j] + setup[q[cursor[ell]]] if cursor[ell] < len(q) else math.inf
            elif continuous_release:
                releases.append(proposed)
        if assigned < len(jobs.p):
            events = [starts[j] + jobs.p[j] for j in active if j is not None]
            events.extend(ready[ell] for ell, q in enumerate(queues)
                          if active[ell] is None and cursor[ell] < len(q) and ready[ell] > t + 1e-12)
            events.extend(releases)
            events = [v for v in events if v > t + 1e-12 and math.isfinite(v)]
            if not events:
                raise RuntimeError("No forward event for unfinished layer")
            t = min(events)
    return starts


def verify_execution(jobs, starts, assignment, queues, setup, radius):
    result = audit(jobs, starts, assignment, queues, radius)
    assert result["violating_pairs"] == 0
    slack = []
    chain_loads = []
    for q in queues:
        if not q:
            chain_loads.append(0.)
            continue
        slack.append(starts[q[0]] - setup[q[0]])
        slack.extend(starts[j] - starts[i] - jobs.p[i] - setup[j] for i, j in zip(q[:-1], q[1:]))
        chain_loads.append(float(np.sum(jobs.p[q] + setup[q])))
    assert min(slack) >= -1e-10, min(slack)
    lower = max(chain_loads)
    result.pop("pairs")
    result.update({"minimum_setup_slack": float(min(slack)), "fixed_order_lower_bound": lower,
                   "upper_bound_gap_pct": max(0., 100. * (result["makespan"] / lower - 1.)),
                   "laser_workload": [float(np.sum(jobs.p[q])) for q in queues],
                   "laser_setup_time": [float(np.sum(setup[q])) for q in queues],
                   "laser_unoccupied_before_completion": [float(starts[q[-1]] + jobs.p[q[-1]] - chain_loads[k]) if q else 0.
                                                         for k, q in enumerate(queues)]})
    return result


def numeric_checks():
    # Closed-form single-laser timing and a forced overlap, independently known.
    j = Jobs(np.arange(3), np.array([[0., 0.], [3., 0.], [8., 0.]]),
             np.array([[1., 0.], [4., 0.], [10., 0.]]), np.array([1., 1., 2.]), np.zeros(3, bool))
    setup, _, _ = setup_times(j, [[0, 1, 2]], 2., .1)
    np.testing.assert_allclose(setup, [0., 1.1, 2.1])
    for continuous in [False, True]:
        starts = dispatch(j, [[0, 1, 2]], setup, 1., continuous, {}, time.perf_counter() + 5.)
        np.testing.assert_allclose(starts, [0., 2.1, 5.2])
        assert abs(max(starts + j.p) - 7.2) < 1e-12
    two = Jobs(np.arange(2), np.array([[0., 0.], [0., .1]]),
               np.array([[1., 0.], [1., .1]]), np.ones(2), np.zeros(2, bool))
    starts = dispatch(two, [[0], [1]], np.zeros(2), .5, True, {}, time.perf_counter() + 5.)
    np.testing.assert_allclose(starts, [0., math.sqrt(.25 - .01)], atol=1e-10)
    verify_execution(two, starts, np.arange(2), [[0], [1]], np.zeros(2), .5)
    previous = json.loads((ROOT / "results.json").read_text())
    zero_errors = []
    for item in previous["exact_pilots"]:
        alljobs = read_jobs(ROOT / "inputs" / (item["case"] + " - SharpCornerCons - less5mm.xlsx"))
        small = alljobs.take([int(np.where(alljobs.ids == i)[0][0]) for i in item["ids"]])
        q = item["queues"]
        ref, _ = wait_repair(small, q, 8.)
        got = dispatch(small, q, np.zeros(len(small.p)), 8., False, {}, time.perf_counter() + 5.)
        zero_errors.append(float(np.max(np.abs(ref - got))))
    assert max(zero_errors) < 1e-10
    return {"single_laser_closed_form": "PASS", "two_laser_exact_release": "PASS",
            "zero_setup_previous_scheduler_max_difference": max(zero_errors)}


def run():
    protocol = {"datasets": ["Layer 10", "Nozzle"], "lasers": [2, 3, 4],
                "orders": ["LPT", "file_order", "nearest_neighbor"],
                "assignment": "same prior pilot eligibility-constrained LPT assignment for all three orders",
                "jump_to_scan_speed_ratios": [None, 20., 10., 5.],
                "None_ratio": "infinite jump speed: original zero-travel limit",
                "delay_fractions_of_median_scan_duration": [0., .02],
                "radius": 8., "retimers": ["whole_job_wait", "continuous_release"],
                "global_wall_budget_s": 300., "parameters_frozen_before_first_execution": True,
                "physics_status": "Illustrative execution-cost sensitivity; no measured machine timing or material-quality validation.",
                "setup_model": "distance(end_i,start_j)/jump_speed + fixed delay for each transition; laser off during setup",
                "scope": "Single-layer makespan, no first positioning, final parking, recoating, acceleration or thermal memory.",
                "causal_scope": "Retimer comparisons preserve each queue. Ordering comparisons deliberately change queue order but preserve laser assignment and vector direction.",
                "prior_art": "Jump times, event dispatch, nearest neighbour and continuous collision intervals are established; no originality claim.",
                "sources": ["https://pyslm.readthedocs.io/en/latest/examples/basic_analysis.html",
                            "https://zenodo.org/records/12747089"],
                "code_sha256": hashlib.sha256(__file__.encode()).hexdigest()}
    # Hash source contents (rather than a path) for provenance.
    from pathlib import Path
    protocol["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT / "execution_cost_protocol.json").write_text(json.dumps(protocol, indent=2))
    begin = time.perf_counter()
    result = {"protocol": protocol, "checks": numeric_checks(), "rows": [], "complete": False}
    selected_schedules = {}
    for name in protocol["datasets"]:
        jobs = read_jobs(ROOT / "inputs" / (name + " - SharpCornerCons - less5mm.xlsx"))
        for lasers in protocol["lasers"]:
            _, assignment, original = lpt(jobs, lasers)
            cache = {}
            for order in protocol["orders"]:
                queue = ordered_queues(jobs, assignment, original, order)
                for ratio_value in protocol["jump_to_scan_speed_ratios"]:
                    for delay in protocol["delay_fractions_of_median_scan_duration"]:
                        ratio = math.inf if ratio_value is None else ratio_value
                        setup, jump_distances, scan_speed = setup_times(jobs, queue, ratio, delay)
                        for continuous in [False, True]:
                            tick = time.perf_counter()
                            starts = dispatch(jobs, queue, setup, 8., continuous, cache,
                                              begin + protocol["global_wall_budget_s"])
                            metrics = verify_execution(jobs, starts, assignment, queue, setup, 8.)
                            row = {"case": name, "n": len(jobs.p), "lasers": lasers, "order": order,
                                   "jump_ratio": ratio_value, "delay_fraction": delay,
                                   "retimer": "continuous_release" if continuous else "whole_job_wait",
                                   "jump_distance": float(np.sum(jump_distances)), "scan_speed_stored_units": scan_speed,
                                   "wall_s": time.perf_counter() - tick, **metrics}
                            result["rows"].append(row)
                            if lasers == 2 and ratio_value == 10. and delay == 0.:
                                key = name.replace(" ", "_") + "_" + order + "_" + row["retimer"]
                                selected_schedules[key + "_starts"] = starts
                                selected_schedules[key + "_assignment"] = assignment
                        (ROOT / "execution_cost_results.json").write_text(json.dumps(result, indent=2))
                print(json.dumps({"case": name, "lasers": lasers, "order": order,
                                  "rows_completed": len(result["rows"]), "elapsed_s": time.perf_counter() - begin}), flush=True)
    result["complete"] = len(result["rows"]) == 288
    result["total_wall_s"] = time.perf_counter() - begin
    (ROOT / "execution_cost_results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(ROOT / "execution_cost_selected_schedules.npz", **selected_schedules)
    print("COMPLETED", len(result["rows"]), result["total_wall_s"], flush=True)


if __name__ == "__main__":
    run()
