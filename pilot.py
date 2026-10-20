"""Bounded computational qualification; no thermal model or AI training.

Replay: research/real_geometry/.venv/bin/python research/scan_planning/pilot.py
All guarantees concern the explicit constant-speed, laser-on distance model.
The source paper's spatial-overlap penalty is a DIFFERENT constraint model.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class Jobs:
    ids: np.ndarray
    a: np.ndarray
    b: np.ndarray
    p: np.ndarray
    dedicated: np.ndarray

    def take(self, idx):
        return Jobs(*(getattr(self, k)[idx] for k in ("ids", "a", "b", "p", "dedicated")))


def read_jobs(path):
    with zipfile.ZipFile(path) as z:
        rows = ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("s:sheetData/s:row", NS)
        values = []
        for row in rows[1:]:
            cells = {}
            for c in row.findall("s:c", NS):
                v = c.find("s:v", NS)
                if v is not None:
                    col = "".join(x for x in c.attrib["r"] if x.isalpha())
                    cells[col] = float(v.text)
            values.append([cells[c] for c in "ABCDEFGH"])
    q = np.asarray(values)
    assert np.isfinite(q).all() and np.all(q[:, 6] > 0)
    assert len(set(q[:, 0])) == len(q)
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:5] - q[:, 1:3], axis=1), q[:, 7], rtol=1e-8)
    return Jobs(q[:, 0].astype(int), q[:, 1:3], q[:, 3:5], q[:, 6], q[:, 5].astype(bool))


def lpt(jobs, m):
    order = np.lexsort((jobs.ids, -jobs.p))
    starts, assignment = np.zeros(len(order)), np.zeros(len(order), dtype=int)
    load = np.zeros(m)
    queues = [[] for _ in range(m)]
    for j in order:
        ell = 0 if jobs.dedicated[j] else int(np.argmin(load))
        assignment[j], starts[j] = ell, load[ell]
        load[ell] += jobs.p[j]
        queues[ell].append(int(j))
    return starts, assignment, queues


def closest_distance(jobs, i, j, si, sj):
    lo = max(si, sj)
    hi = min(si + jobs.p[i], sj + jobs.p[j])
    if hi - lo <= 1e-12:
        return math.inf
    vi, vj = (jobs.b[i] - jobs.a[i]) / jobs.p[i], (jobs.b[j] - jobs.a[j]) / jobs.p[j]
    r = jobs.a[i] + vi * (lo - si) - jobs.a[j] - vj * (lo - sj)
    v = vi - vj
    vv = float(v @ v)
    tau = np.clip(-float(r @ v) / vv, 0, hi - lo) if vv > 1e-25 else 0
    return float(np.linalg.norm(r + v * tau))


def audit(jobs, starts, assignment, queues, distance):
    """Sweep every overlapping interval; also measure duration below the limit."""
    conflicts, minimum, unsafe_sum = [], math.inf, 0.0
    seen = sorted(j for q in queues for j in q)
    assert seen == list(range(len(jobs.p)))
    assert np.all(assignment[jobs.dedicated] == 0)
    assert np.min(starts) >= -1e-8
    for ell, q in enumerate(queues):
        for k, j in enumerate(q):
            assert assignment[j] == ell
            if k:
                assert starts[j] >= starts[q[k - 1]] + jobs.p[q[k - 1]] - 1e-8
    for x in range(len(queues)):
        for y in range(x + 1, len(queues)):
            qx, qy = queues[x], queues[y]
            b0 = 0
            for i in qx:
                while b0 < len(qy) and starts[qy[b0]] + jobs.p[qy[b0]] <= starts[i] + 1e-12:
                    b0 += 1
                for j in qy[b0:]:
                    if starts[j] >= starts[i] + jobs.p[i] - 1e-12:
                        break
                    lo, hi = max(starts[i], starts[j]), min(starts[i] + jobs.p[i], starts[j] + jobs.p[j])
                    if hi <= lo + 1e-12:
                        continue
                    v1 = (jobs.b[i] - jobs.a[i]) / jobs.p[i]
                    v2 = (jobs.b[j] - jobs.a[j]) / jobs.p[j]
                    r = jobs.a[i] + v1 * (lo - starts[i]) - jobs.a[j] - v2 * (lo - starts[j])
                    v = v1 - v2
                    aa, bb, cc = float(v @ v), 2 * float(r @ v), float(r @ r) - distance**2
                    dmin = closest_distance(jobs, i, j, starts[i], starts[j])
                    minimum = min(minimum, dmin)
                    if dmin < distance - 1e-7:
                        conflicts.append((i, j))
                        if aa < 1e-25:
                            unsafe_sum += hi - lo
                        else:
                            disc = max(0, bb * bb - 4 * aa * cc)
                            t1, t2 = (-bb - math.sqrt(disc)) / (2 * aa), (-bb + math.sqrt(disc)) / (2 * aa)
                            unsafe_sum += max(0, min(hi - lo, t2) - max(0, t1))
    return {"violating_pairs": len(conflicts), "minimum_distance": minimum if np.isfinite(minimum) else None,
            "sum_pairwise_violation_duration": unsafe_sum, "makespan": float(np.max(starts + jobs.p)),
            "pairs": conflicts}


def wait_repair(jobs, queues, distance):
    """Standard conservative event-driven baseline, preserving assignment/order.

    A blocked laser waits until a currently executing scan segment finishes.
    Continuous within-segment release times are intentionally not optimized.
    """
    m = len(queues)
    starts = np.zeros(len(jobs.p))
    assignment = np.empty(len(jobs.p), int)
    cursor = np.zeros(m, int)
    active = [None] * m
    t, done = 0., 0
    while done < len(jobs.p):
        for ell in range(m):
            if active[ell] is not None and starts[active[ell]] + jobs.p[active[ell]] <= t + 1e-12:
                active[ell] = None
        for ell in range(m):
            if active[ell] is not None or cursor[ell] == len(queues[ell]):
                continue
            j = queues[ell][cursor[ell]]
            if all(i is None or closest_distance(jobs, j, i, t, starts[i]) >= distance - 1e-9 for i in active):
                starts[j], assignment[j], active[ell] = t, ell, j
                cursor[ell] += 1
                done += 1
        if done < len(jobs.p):
            ends = [starts[j] + jobs.p[j] for j in active if j is not None]
            assert ends and min(ends) > t
            t = min(ends)
    return starts, assignment


def forbidden_interval(a, b, pi, c, d, pj, radius):
    """Project a convex disk/box intersection onto start difference s_j-s_i.

    Local scan times z=(t_i,t_j); relative start delta=t_i-t_j.
    Extremizers lie at box corners, edge/disk intersections, or unconstrained
    ellipse support points. Parallel/degenerate paths are covered by edges.
    Endpoint-only contact is permitted; interference concerns positive overlap.
    """
    B = np.column_stack(((b - a) / pi, -(d - c) / pj))
    r = a - c
    hi = np.array([pi, pj])
    h = np.array([1., -1.])
    candidates = []

    def add(z):
        if np.all(z >= -1e-9) and np.all(z <= hi + 1e-9) and np.linalg.norm(r + B @ z) <= radius + 1e-7:
            candidates.append(float(h @ np.clip(z, 0, hi)))

    for fixed in range(2):
        free = 1 - fixed
        for value in (0., hi[fixed]):
            z = np.zeros(2)
            z[fixed] = value
            q, v = r + B @ z, B[:, free]
            aa, bb, cc = float(v @ v), 2 * float(q @ v), float(q @ q) - radius**2
            if aa < 1e-25:
                limits = (0., hi[free]) if cc <= 0 else ()
            else:
                disc = bb * bb - 4 * aa * cc
                if disc < -1e-9:
                    limits = ()
                else:
                    root = math.sqrt(max(0, disc))
                    low, high = max(0., (-bb - root) / (2 * aa)), min(hi[free], (-bb + root) / (2 * aa))
                    limits = (low, high) if high >= low - 1e-10 else ()
            for x in limits:
                z[free] = x
                add(z.copy())
    if abs(np.linalg.det(B)) > 1e-10 * max(1., np.linalg.norm(B)**2):
        # Work in physical displacement coordinates instead of forming B.T@B;
        # normal equations square the condition number for almost-parallel scans.
        support = np.linalg.solve(B.T, h)
        support = radius * support / np.linalg.norm(support)
        add(np.linalg.solve(B, -support - r))
        add(np.linalg.solve(B, support - r))
    if not candidates or max(candidates) - min(candidates) <= 1e-10:
        return None
    return min(candidates), max(candidates)


def exact_retime(jobs, assignment, queues, radius, upper, limit=35.):
    """MILP with valid pair separation; certificate applies to fixed chains only."""
    ts = time.perf_counter()
    scale = float(np.median(jobs.p))
    p, n = jobs.p / scale, len(jobs.p)
    horizon = float(p.sum())
    cuts, records = {}, []
    best = None
    for iteration in range(100):
        if time.perf_counter() - ts > limit:
            break
        nv = n + 1 + len(cuts)
        rr, cc, vv, lows, highs = [], [], [], [], []

        def row(vals, low=-np.inf, high=np.inf):
            k = len(lows)
            for col, value in vals.items():
                rr.append(k); cc.append(col); vv.append(value)
            lows.append(low); highs.append(high)

        for q in queues:
            for i, j in zip(q[:-1], q[1:]):
                row({j: 1., i: -1.}, low=p[i])
            if q:
                row({n: 1., q[-1]: -1.}, low=p[q[-1]])
        big = 3 * horizon
        for k, ((i, j), (low, high)) in enumerate(cuts.items()):
            vals = {j: 1., i: -1., n + 1 + k: -big}
            row(vals, high=low)
            row(vals, low=high - big)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lows), nv)).tocsc()
        lower, upp = np.zeros(nv), np.full(nv, horizon)
        upp[n] = upper / scale + 1e-7
        upp[n + 1:] = 1.
        cobj = np.zeros(nv); cobj[n] = 1.
        integrality = np.zeros(nv, int); integrality[n + 1:] = 1
        # HiGHS' default MIP feasibility tolerance is too loose for adjacent
        # segment start/end events. scipy forwards these supported HiGHS options.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            out = milp(cobj, integrality=integrality, bounds=Bounds(lower, upp),
                       constraints=LinearConstraint(matrix, lows, highs),
                       options={"time_limit": min(5., max(.1, limit - (time.perf_counter() - ts))),
                                "mip_rel_gap": 1e-8, "mip_feasibility_tolerance": 1e-9,
                                "primal_feasibility_tolerance": 1e-9})
        log = {"iteration": iteration, "cuts": len(cuts), "status": int(out.status), "message": out.message}
        if out.x is None:
            records.append(log)
            break
        # Independent difference-constraint replay of the selected disjuncts.
        # This removes tiny Big-M feasibility violations without weakening the
        # continuous collision checker or changing the selected discrete order.
        edges = [(i, j, p[i]) for q in queues for i, j in zip(q[:-1], q[1:])]
        for k, ((i, j), (low, high)) in enumerate(cuts.items()):
            edges.append((i, j, high) if out.x[n + 1 + k] > .5 else (j, i, -low))
        polished = np.zeros(n)
        for replay_iteration in range(n + 1):
            changed = False
            for i, j, w in edges:
                if polished[i] + w > polished[j] + 1e-12:
                    polished[j] = polished[i] + w
                    changed = True
            if not changed:
                break
        else:
            raise RuntimeError("Selected disjunctions have an infeasible positive cycle")
        starts = polished * scale
        check = audit(jobs, starts, assignment, queues, radius)
        bound = getattr(out, "mip_dual_bound", None)
        if bound is None and out.status == 0:
            bound = out.fun
        log.update({"makespan": check["makespan"], "violations": check["violating_pairs"],
                    "solver_makespan": float(out.fun) * scale,
                    "lower_bound": float(bound) * scale if bound is not None else None})
        records.append(log)
        if not check["pairs"]:
            gap = max(0., check["makespan"] - log["lower_bound"]) / check["makespan"] if log["lower_bound"] is not None else None
            best = {"starts": starts.tolist(), "audit": {k: v for k, v in check.items() if k != "pairs"},
                    "certified_optimal_within_tolerance": bool(out.status == 0 and gap is not None and gap <= 1e-7),
                    "relative_gap": gap, "lower_bound": log["lower_bound"]}
            break
        added = 0
        for i, j in check["pairs"]:
            if (i, j) in cuts:
                raise RuntimeError({"kind": "Previously separated constraint still violated", "pair": [i, j],
                                    "interval": cuts[(i, j)], "delta": float((starts[j]-starts[i])/scale),
                                    "distance": closest_distance(jobs, i, j, starts[i], starts[j]),
                                    "a": jobs.a[[i, j]].tolist(), "b": jobs.b[[i, j]].tolist(),
                                    "p": p[[i, j]].tolist()})
            interval = forbidden_interval(jobs.a[i], jobs.b[i], p[i], jobs.a[j], jobs.b[j], p[j], radius)
            assert interval is not None
            cuts[(i, j)] = interval
            added += 1
        assert added
    return {"result": best, "iterations": records, "wall_s": time.perf_counter() - ts}


def numerical_checks():
    rng = np.random.default_rng(83217)
    disagreements = 0
    count = 0
    for k in range(200):
        points = rng.normal(size=(4, 2)) * 3
        if k % 3 == 0:
            points[3] = points[2] + points[1] - points[0]
        if k >= 100:
            points[3] = points[2] + points[1] - points[0] + rng.normal(size=2) * 10.**(-int(rng.integers(5, 12)))
        p = rng.uniform(.5, 2, 2)
        rad = 1.7
        interval = forbidden_interval(*points[:2], p[0], *points[2:], p[1], rad)
        jobs = Jobs(np.arange(2), points[[0, 2]], points[[1, 3]], p, np.zeros(2, bool))
        for delta in rng.uniform(-p[1] - .2, p[0] + .2, 50):
            measured = closest_distance(jobs, 0, 1, max(0, -delta), max(0, delta)) < rad - 1e-8
            predicted = interval is not None and interval[0] + 1e-8 < delta < interval[1] - 1e-8
            disagreements += bool(measured != predicted)
            count += 1
    assert disagreements == 0, disagreements
    # Same trajectories split in two must give the same closest distance.
    split_errors = []
    for k in range(100):
        a, b = rng.normal(size=(2, 2, 2)) * 4
        p = rng.uniform(.5, 2, 2)
        si = rng.uniform(0, 1, 2)
        jobs = Jobs(np.arange(2), a, b, p, np.zeros(2, bool))
        whole = closest_distance(jobs, 0, 1, *si)
        mids = (a + b) / 2
        part = Jobs(np.arange(4), np.vstack([a, mids]), np.vstack([mids, b]), np.tile(p / 2, 2), np.zeros(4, bool))
        st = np.r_[si, si + p / 2]
        pieces = min(closest_distance(part, i, j, st[i], st[j]) for i in [0, 2] for j in [1, 3])
        if np.isfinite(whole):
            split_errors.append(abs(whole - pieces))
        else:
            assert not np.isfinite(pieces)
    assert max(split_errors) < 1e-9
    return {"random_interval_checks": count, "disagreements": disagreements,
            "fixed_trajectory_subdivision_max_distance_error": max(split_errors)}


def main():
    t0 = time.perf_counter()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    radius = protocol["minimum_distance"]
    output = {"checks": numerical_checks(), "full_cases": [], "exact_pilots": []}
    schedules = {}
    for name in ["Layer 10 - SharpCornerCons - less5mm.xlsx", "Nozzle - SharpCornerCons - less5mm.xlsx"]:
        jobs = read_jobs(ROOT / "inputs" / name)
        label = name.split(" -")[0]
        for m in [2, 3, 4]:
            st, assignment, qs = lpt(jobs, m)
            t = time.perf_counter()
            repaired, assignment2 = wait_repair(jobs, qs, radius)
            elapsed = time.perf_counter() - t
            assert np.array_equal(assignment, assignment2)
            raw_check, check = audit(jobs, st, assignment, qs, radius), audit(jobs, repaired, assignment, qs, radius)
            assert not check["violating_pairs"]
            record = {"case": label, "n": len(jobs.p), "lasers": m,
                      "dedicated_jobs": int(jobs.dedicated.sum()), "raw": {k: v for k, v in raw_check.items() if k != "pairs"},
                      "repaired": {k: v for k, v in check.items() if k != "pairs"}, "repair_wall_s": elapsed,
                      "workload_lower_bound": max(float(jobs.p.sum()) / m, float(jobs.p[jobs.dedicated].sum()), float(jobs.p.max()))}
            output["full_cases"].append(record)
            schedules[f"{label}_{m}_raw"] = st
            schedules[f"{label}_{m}_wait"] = repaired
            schedules[f"{label}_{m}_laser"] = assignment
            print(json.dumps(record), flush=True)
        order = np.lexsort((jobs.ids, -jobs.p))
        for fraction in [0., 1/3, 2/3]:
            start = int(fraction * (len(order) - 16))
            sub = jobs.take(order[start:start + 16])
            st, assn, qs = lpt(sub, 2)
            rep, _ = wait_repair(sub, qs, radius)
            base = audit(sub, rep, assn, qs, radius)
            solved = exact_retime(sub, assn, qs, radius, base["makespan"])
            record = {"case": label, "selection_fraction": fraction, "ids": sub.ids.tolist(),
                      "assignment": assn.tolist(), "queues": qs, "wait_makespan": base["makespan"], **solved}
            output["exact_pilots"].append(record)
            print(json.dumps(record), flush=True)
            (ROOT / "results.json").write_text(json.dumps(output, indent=2))
    output["total_wall_s"] = time.perf_counter() - t0
    output["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT / "results.json").write_text(json.dumps(output, indent=2))
    np.savez_compressed(ROOT / "schedules.npz", **schedules)
    print("TOTAL_WALL_S", output["total_wall_s"], flush=True)


if __name__ == "__main__":
    main()
