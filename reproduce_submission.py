"""Replay the submitted study in a separate directory, preserving reference evidence.

Usage: python reproduce_submission.py --mode full --work-dir /tmp/lpbf-replay
The installed interpreter must satisfy requirements.lock.txt. Numerical replay
uses no network; a TeX compiler may separately fetch its TeX/font dependencies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
IGNORE = {'wall_s', 'peak_MB', 'dp_wall_s', 'runtimes'}
RESULTS = ['primary_results.json', 'exact_results.json', 'mechanism_results.json',
           'search_results.json', 'search_plans.json', 'continuous_results.json',
           'summary.json', 'verification.json']


def compare(a, b, path='', stats=None):
    if stats is None:
        stats = {'numeric_values': 0, 'max_absolute_difference': 0., 'non_numeric_values': 0}
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            if key not in IGNORE:
                compare(a[key], b[key], path+'/'+key, stats)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+'/'+str(i), stats)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)), path
        if isinstance(a, int):
            assert a == b, (path, a, b)
        else:
            assert math.isfinite(a) and math.isfinite(b), path
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (path, a, b)
        stats['numeric_values'] += 1
        stats['max_absolute_difference'] = max(stats['max_absolute_difference'], abs(a-b))
    else:
        assert a == b, (path, a, b)
        stats['non_numeric_values'] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['check', 'full'], default='full')
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--paper', action='store_true', help='Requires tectonic on PATH; implies figures')
    args = ap.parse_args()
    work = args.work_dir.resolve()
    if work.exists():
        raise SystemExit('Choose a new work directory; existing results will not be overwritten.')
    manifest = ROOT/'SHA256SUMS.json'
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    payload = json.loads((ROOT/'reproduction_files.json').read_text())
    work.mkdir(parents=True)
    for name in payload:
        dest = work/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    (work/'submission').mkdir(exist_ok=True)
    (work/'paper').mkdir(exist_ok=True)
    if args.mode == 'check':
        for name in RESULTS+['separation_schedules.npz']:
            shutil.copy2(ROOT/'submission'/name, work/'submission'/name)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    stages = [['submission_revision.py', x] for x in ['primary','exact','mechanism','search']] if args.mode == 'full' else []
    stages += [['verify_submission.py'], ['submission_revision.py','summarize']]
    if args.figures or args.paper:
        stages += [['submission_figures.py'], ['submission_manuscript.py'], ['check_release_artifacts.py']]
    started = time.perf_counter()
    timings = []
    for n, stage in enumerate(stages):
        stamp = time.perf_counter()
        print('Running '+ ' '.join(stage), flush=True)
        with (work/'submission'/f'replay_{n:02d}.log').open('w') as log:
            subprocess.run([sys.executable, *stage], cwd=work, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1500)
        timings.append({'stage':stage,'seconds':time.perf_counter()-stamp})
    comparisons = {}
    for name in RESULTS:
        comparisons[name] = compare(json.loads((ROOT/'submission'/name).read_text()),
                                    json.loads((work/'submission'/name).read_text()))
    if args.figures or args.paper:
        comparisons['artifact_verification.json'] = compare(
            json.loads((ROOT/'submission/artifact_verification.json').read_text()),
            json.loads((work/'submission/artifact_verification.json').read_text()))
    import numpy as np
    original = np.load(ROOT/'submission/separation_schedules.npz')
    replay = np.load(work/'submission/separation_schedules.npz')
    assert set(original.files) == set(replay.files)
    for name in original.files:
        np.testing.assert_allclose(original[name], replay[name], rtol=1e-9, atol=1e-10)
    if args.paper:
        compiler = shutil.which('tectonic')
        if not compiler:
            raise SystemExit('Numerical replay passed. Install Tectonic to compile the paper.')
        subprocess.run([compiler, '--keep-logs', 'main.tex'], cwd=work/'paper', check=True, timeout=300)
    out = {'complete':True, 'mode':args.mode, 'python':sys.version,
           'comparisons':comparisons, 'schedule_arrays':len(original.files),
           'ignored_machine_dependent_fields':sorted(IGNORE),
           'tolerance':{'relative':1e-9,'absolute':1e-10,'integers_and_identifiers':'exact'},
           'stages':timings, 'total_seconds':time.perf_counter()-started}
    (work/'replay_verification.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
