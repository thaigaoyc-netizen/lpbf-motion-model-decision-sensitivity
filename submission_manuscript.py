"""Render manuscript statistics and tables from the audited numerical release."""
import json
import re
from pathlib import Path
from submission_revision import ROOT,OUT,load

def replace_between(text,start,end,new):
    a=text.index(start);b=text.index(end,a);return text[:a]+new+text[b:]

def main():
    s=load('summary.json');v=load('verification.json');assert v['complete']
    t=(ROOT/'paper/before_submission_revision_20261017.tex').read_text()
    t=t.replace(r'\date{Revised working manuscript --- 17 October 2026}',r'\date{}')
    t=t.replace('images/fig','../submission/figures/fig')
    tables=ROOT/'paper/tables';tables.mkdir(exist_ok=True)
    def tablefile(name,body):
        (tables/(name+'.tex')).write_text('% Generated from submission/summary.json; do not edit numbers manually.\n'+body+'\n')
    def replace_table(label,filename,rows):
        nonlocal t
        pattern=r'(\\label\{'+re.escape(label)+r'\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        match=re.search(pattern,t,re.S);assert match,label
        tablefile(filename,rows)
        # Inline generated rows: LaTeX file hooks inside tabular can interfere
        # with booktabs' noalign. Separate generated table files remain supplied.
        t=t[:match.start(2)]+rows+'\n'+t[match.end(2):]
    rows=[]
    for x in s['table2']:
        z=x['loss'];rows.append(f"{x['radius']} & {x['alpha']} & {z['n']} & {x['bias']['median']:.3f} & {z['zero']} ({100*z['zero']/z['n']:.1f}) & {z['p95']:.3f} & {z['maximum']:.3f} & {x['bound']['positive']}"+r'\\')
    replace_table('tab:extension','table2','\n'.join(rows))
    titles={'brake_rest_ramp':'Brake/rest/ramp','optional_cubic':'Optional cubic','constant_speed':'Constant speed'}
    replace_table('tab:sensitivity','table3','\n'.join(f"{titles[z['model']]} & {'Original 13' if z['portfolio']=='original_13' else 'Expanded 19'} & {z['positive']} & {z['p95']:.3f} & {z['maximum']:.3f}"+r'\\' for z in s['table3']))
    ports={'original_13':'Original 13','expanded_19':'Expanded 19','original_without_EFT':'Without EFT family','original_without_x_partition':r'Without $x$ partitions','original_without_y_partition':r'Without $y$ partitions','original_without_list_family':'Without list-based family'}
    deletion='\n'.join(f"{ports[z['portfolio']]} & {z['positive']} & {z['p95']:.3f} & {z['maximum']:.3f}"+r'\\' for z in s['table4'] if z['portfolio'] in ['original_13','original_without_EFT','original_without_x_partition'])
    replace_table('tab:deletion','table4',deletion)
    full='\n'.join(f"{ports[z['portfolio']]} & {z['positive']} & {z['p95']:.3f} & {z['maximum']:.3f}"+r'\\' for z in s['table4'] if z['portfolio']!='expanded_19')
    tablefile('deletion_complete',full)
    a=t.index(r'\section{Complete portfolio-deletion results}');b=t.index(r'\midrule',a)+len(r'\midrule');c=t.index(r'\bottomrule',b)
    t=t[:b]+'\n'+full+'\n'+t[c:]
    abstract=r'''\begin{abstract}
Scan-time prediction and scan-plan selection need not respond equally to motion-model error in multi-laser powder bed fusion.
We assess this distinction on two released layers and 24 constructed geometry--angle cases, retaining common directed exposures and candidate plans.
An outcome-independent tie rule and isolated simplified-model retiming avoid using reference outcomes to break ties or construct simplified schedules.
A piecewise-affine calculation bounds fixed-plan loss over a continuous constant-speed interval; the largest minimax loss within the 13-plan portfolio is 1.3411\%.
Changing the motion law can nevertheless reverse a near-tie: an unequal-island case has an 11.04\% loss under brake/rest/ramp motion and 11.47\% when feasible cubic connections are allowed.
Across 468 zero-separation settings, the expanded 19-plan comparison has 94 positive losses and zero median loss.
Three hundred spatial-window comparisons against exact two-laser scheduling, together with full-layer multistart search, show that the phenomenon persists beyond the original candidate constructions without establishing full-layer optimality.
At a fixed diagnostic setting, all 13 positive-loss cases have more critical-chain jumps, but only three have greater direct travel distance.
The results separate timing bias, model-conditional decision loss, and optimizer limitations; they do not estimate measured production-time savings or validate manufacturing quality.
\end{abstract}'''
    t=replace_between(t,r'\begin{abstract}',r'\section{Introduction}',abstract+'\n\n')
    # Add focused antecedents rather than claiming a new motion or search algorithm.
    t=t.replace('Acceleration-aware scan planning predates this study.',r'''Process planning combines geometric preparation, toolpath generation and manufacturing objectives that need not agree \cite{livesu}.
The present study fixes the exposure geometry and changes only execution-time evaluation; it therefore concerns one restricted part of that pipeline.
Acceleration-aware scan planning predates this study.''')
    t=t.replace('Cubic interpolation, minimax regret, dynamic programming and chain bounds remain established tools',r'''The stronger comparator uses conventional neighborhood search \cite{vns}; it is an optimization control, not a new planning method.
Trajectory generators such as Ruckig solve a different, jerk-limited state-to-state problem with per-axis constraints \cite{ruckig}.
Neither their minimum-time guarantees nor measured scanner qualification results transfer to the Euclidean-bound motion conventions used here.
Cubic interpolation, minimax regret, dynamic programming and chain bounds remain established tools''')
    t=t.replace('Already computed start times from slower-motion settings are also retained when they remain feasible, including the interruption condition above.\nThe smallest verified makespan among these candidates is used.',r'''For the finite-acceleration reference, schedules from previously evaluated lower-acceleration settings are also retained if independently feasible, including the interrupted-join condition.
The constant-speed selector uses only its own two dispatchers and never receives a finite-acceleration schedule.
The smallest verified makespan within each permitted set is used.''')
    t=t.replace('Thus the sensitivity scenarios cannot be interpreted as measured equipment variability.',r'''Holding $\alpha$ across cases need not hold $a$ fixed, because their median vector durations can differ.
A supplementary cross-geometry check therefore fixes $a=720{,}000$ stored coordinate units per squared stored time unit and $V=6{,}000$ stored coordinate units per stored time unit.
These values are a declared scenario, not an identified scanner calibration.
Thus the sensitivity scenarios cannot be interpreted as measured equipment variability.''')
    t=replace_between(t,'To avoid attributing an arbitrary tie',r'\subsection{A bound valid',r'''Costs within $\epsilon_C=10^{-8}\sum_i p_i$ of the minimum are tied.
The primary selector chooses the lexicographically first, case-sensitive plan label from that set:
\begin{equation}
\hat s=\operatorname{lexmin}\{s:C_s^{\rm cv}\leq\min_u C_u^{\rm cv}+\epsilon_C\}.
\label{eq:tie}
\end{equation}
This rule is fixed before reference evaluation and uses no reference cost or schedule.
Reference-best and reference-worst choices within the same tie set are reported only as optimistic and pessimistic sensitivity bounds.
Candidate labels and their complete queues are versioned, so the arbitrary but executable ordering is reproducible.

''')
    t=t.replace('Scan direction reversal, new learned policies, and reference-model-specific assignment optimization are excluded.',r'''The main 13/19-plan comparison keeps these constructions fixed.
Exposure-vector direction reversal and learned policies are excluded.
A separate stronger-search comparison allows reassignment and reordering and supplies its entire candidate union to both evaluators.''')
    small=r'''\subsection{Exact spatial subproblems and stronger search}
\label{sec:smallmethod}
To quantify portfolio suboptimality, we retain one angle per constructed family ($30^\circ$ for the earlier four and $45^\circ$ for the extension) and both released layers.
For each case, vector midpoints are sorted by $x$, then $y$, then identifier.
Seeds at the 25th and 75th percentiles define two spatial neighborhoods.
The nearest 8, 12 or 16 midpoints to each seed are retained, with deterministic identifier ties.
These compact windows preserve whole directed vectors, exposure durations and dedicated-laser flags; they are not sparse, uniformly sampled vector indices or newly fabricated parts.
Windows may overlap and the three sizes are nested, so they are not independent replications.

All windows use two lasers, $r=0$, $\kappa=10$, and the dimensional accelerations implied by the \emph{parent} case at $\alpha=10$ and 100.
The subset's $\alpha$ is adjusted to retain that acceleration.
Constant-speed costs are evaluated once, giving $10\times2\times3\times5=300$ comparisons.
The original 13 and expanded 19 constructions are regenerated on every window.

For an exposure subset $M$ ending at vector $j$, an open-path subset dynamic program \cite{heldkarp} computes
\begin{equation}
D(M,j)=p_j+\min_{i\in M\setminus\{j\}}\{D(M\setminus\{j\},i)+d_{ij}\},\qquad D(\{j\},j)=p_j.
\end{equation}
Let $N$ be the window's full vector set, $g(M)=\min_{j\in M}D(M,j)$ and $g(\varnothing)=0$.
For disjoint mandatory-assignment sets $D_1,D_2$, exhaustive partitioning gives
\begin{equation}
C^*=\min_{\substack{D_1\subseteq M\subseteq N\\M\cap D_2=\varnothing}}
\max\{g(M),g(N\setminus M)\}.
\label{eq:exact}
\end{equation}
Here the input flag defines $D_1$ and $D_2=\varnothing$; the data do not encode second-laser mandatory jobs.
The singleton initialization excludes initial travel, no terminal parking term is added, and zero-cost continuous joins are allowed without special pruning.
The solution is exact over assignments and orders under the specified edge costs, not over alternative motion laws.
Three six-vector permutation-and-partition enumerations and explicit zero-edge/all-dedicated cases verify the recurrence and reconstruction.

The stronger baseline uses conventional multistart neighborhood descent with single-vector relocation, vector exchange, and within-queue order reversal \cite{vns}.
Reversing an order never reverses an exposure vector.
The best three distinct portfolio schedules under the optimized model initialize the search; a fixed seed generates 1,200 proposals per start on each window and 6,000 per start on each full layer.
Accepted moves decrease the lexicographic pair (makespan, sum of chain costs), compared at 12 decimal places.
Dedicated assignments remain enforced; reversal segments are capped at 32 vectors.
This is bounded randomized search, not a certificate of local or global optimality.

On all 52 full geometry--laser configurations at $\kappa=10$, $\alpha=10$ and $r=0$, searches optimize constant-speed and brake/rest/ramp objectives separately.
Their up to six resulting plans are added to the same original 19 candidates.
Both selectors see this common union, including candidates generated under the other objective; neither receives the other's evaluation when breaking ties.
The common union is a controlled-comparison asset, not a reference-free online plan-construction algorithm.
The union is evaluated under both finite-motion families.
The optional-cubic model is not separately optimized on full layers, a limitation of this bounded comparator.

\subsection{Cross-geometry mechanism and representation checks}
For each of the 52 configurations at $(\kappa,\alpha)=(10,10)$, we compare the selected and reference-best plans from the original 13.
Each plan's own critical laser supplies exposure load, nonzero transition count, direct endpoint distance sum, and $\sum\|v_j-v_i\|$.
Finite-motion time is decomposed into braking/acceleration and intermediate rest-to-rest travel.
We also record whether the critical laser changes relative to constant-speed evaluation.
The complete comparison is repeated at the fixed dimensional parameters defined above.
Exposure-speed magnitudes are essentially constant in these inputs, so endpoint-velocity differences primarily measure direction changes, not variation in commanded scan speed.
These descriptive comparisons do not identify independent causal effects of correlated path features.

Finally, the same unequal-island queues are subdivided into 1, 2 and 4 equal contiguous pieces while dimensional acceleration is preserved.
A fixed-$\alpha$ control exposes the acceleration confound.
Outcome-informed tie bounds remain supplementary to Equation~\eqref{eq:tie}; neither changes the primary outcomes.

'''
    t=replace_between(t,r'\subsection{Exact small problems',r'\subsection{Fixed settings',small)
    t=replace_between(t,'This yields 5,824 plan evaluations','The 624 comparison settings',f'''The revised calculation regenerates 20,748 zero-separation plan--model costs and 2,704 separation-constrained costs, including 5,408 raw separation schedule audits.
It takes {s['runtimes']['primary']:.2f} seconds on the local Mac.
The 300 exact comparisons take {s['runtimes']['exact']:.2f} seconds, full-layer neighborhood search {s['runtimes']['search']:.2f} seconds, and cross-geometry mechanism calculation {s['runtimes']['mechanism']:.2f} seconds; these are separate process wall times, not additive elapsed project time.
Historical results are preserved but excluded wherever superseded.
No GPU or physical printing is used.

''')
    t=replace_between(t,'An earlier four-island example with',r'\begin{figure}[p]',r'''All separation results use the interrupted-join restriction and the information-isolated selector.
The accompanying audit records the corrected historical calculations rather than mixing them into the current statistics.

''')
    t=t.replace(r'''(a,b) Identical planar geometry and camera, colored by actual laser assignment. Laser-colored dots mark each laser's first exposure; thin dark paths are the critical laser's nonzero brake/rest/ramp transitions, with its longest-duration transition highlighted in red. Rendered thickness and powder texture only aid depth perception. (c,d) Every exposure and off-motion interval is retained; outlined rows are critical. (e) Exact critical-chain costs.''',r'''(a,b) Orthographic views of identical planar exposure geometry, colored by laser assignment. Dots mark first exposures; dark curves are critical-laser off paths. (c,d) Actual longest-duration transitions from the outlined regions, enlarged independently without deforming their geometry. (e,f) Complete four-laser time axes; outlined rows are critical. (g) Exact critical-chain costs.''')
    t=t.replace('Three hundred independently sampled trajectories and a rest-to-rest closed form check the polynomial implementation; every evaluated cubic also passes analytic continuous-limit checks. Figure~\\ref{fig:motion} illustrates the endpoint and duration differences on a retained transition.',r'''Three hundred independently sampled trajectories and a rest-to-rest closed form check the polynomial implementation; every evaluated cubic also passes analytic continuous-limit checks.
Figure~\ref{fig:motion} shows the largest absolute cubic saving on the already specified critical chain of the unequal-island example.
The previous longest-duration edge, where cubic motion has no advantage, is retained in Supplementary Figure~\ref{fig:noadvantage}.
This descriptive figure selection does not alter any aggregated experiment.''')
    t=t.replace('using the brake/rest/ramp duration as a common time scale.', 'using a common axis in stored time units.')
    t=t.replace('gray segments are the adjacent exposure vectors.', 'gray segments show only the adjacent portions of the unchanged exposure vectors.')
    # Numerical narrative kept in sync with the generated table.
    t=t.replace('eight observed losses are positive, but only three', 'seven observed losses are positive, but only three')
    t=replace_between(t,'Across the extension, all 59 positive lower-bound witnesses',r'Figure~\ref{fig:map}',f'''All {v['selected_separation_schedule_replays']} selected and alternative separation schedules were replayed from stored start times without rerunning the dispatcher; assignment, precedence, continuous beam separation and interrupted-join checks passed.
At $\\alpha=1000$, one extension setting has a 0.507\\% observed gap but no positive chain-bound witness, illustrating the remaining retimer dependence.

''')
    t=t.replace('Every cell is one evaluated setting; unnumbered cells have zero loss within tolerance. Positive values are rounded to one decimal, with very small positive values marked $<.1$. The left nine columns in each panel have $r=0$ and the right three have $r=8$; only $\\kappa=10$ is evaluated at $r=8$. Separation entries use the corrected interrupted-join calculations.',r'''The two left panels contain $r=0$ settings; the two right panels contain $r=8$ at $\kappa=10$ only. Pale cells with dots mean zero loss; colors encode positive loss and values at least 5\% are annotated. Hatched white denotes not evaluated, not zero; the untested separation speed ratios are stated explicitly. Family separators preserve geometry membership. Every value uses the executable tie rule and information-isolated retiming.''')
    t=t.replace('Positive selection loss rises from 85 to 91','Positive selection loss rises from 85 to 94')
    t=t.replace('14 become positive after portfolio expansion; eight of the 85 positive settings become zero, giving a net increase of six.', '17 become positive after portfolio expansion; eight of the 85 positive settings become zero, giving a net increase of nine.')
    t=t.replace('Only 36 settings change loss by more than $10^{-6}$ percentage points; the other 432 are unchanged.', 'Only 39 settings change loss by more than $10^{-6}$ percentage points; the other 429 are unchanged.')
    # Recompute family statements rather than preserving historical optimistic counts.
    comps=load('primary_results.json')['comparisons'];rr=[r for r in comps if r['portfolio']=='expanded_19' and r['model']=='optional_cubic' and r['radius']==0]
    for family,oldcount in [('unequal_islands_',28)]:
        count=sum(r['selection_loss_pct']>1e-6 for r in rr if r['case'].startswith(family))
        t=t.replace(f'positive loss in {oldcount} of 54',f'positive loss in {count} of 54')
    t=t.replace('the L section and equal-island family have zero selection loss in all 54 settings each, whereas the unequal-island family has positive loss in 28 of 54.', 'the equal-island family has zero selection loss in all 54 settings, the L section has three positive settings, and unequal islands have positive loss in 28 of 54.')
    exact16=next(r for r in s['exact'] if r['size']==16 and r['model']=='brake_rest_ramp')
    search=s['search'][0];searchc=s['search'][1]
    newresults=r'''\subsection{Exact comparisons and stronger full-layer candidates}
Figure~\ref{fig:checks} reports all spatial-window sizes and all three motion models.
The original and expanded portfolios are often suboptimal even on compact windows.
For the 40 brake/rest/ramp comparisons at 16 vectors, the 19-plan median gap to exact scheduling is @MED19@\% and its maximum is @MAX19@\%.
Neighborhood search reduces these values to @MEDSEARCH@\% and @MAXSEARCH@\%, respectively.
The corresponding optional-cubic median and maximum search gaps remain 1.697\% and 8.589\%.
Thus stronger search materially improves the baseline but does not close every gap.
The three exhaustive six-vector checks agree with the dynamic program within $4.45\times10^{-16}$ stored time units.
All exact and searched queues satisfy coverage and dedicated-assignment restrictions.
Neither these windows nor their gap distribution certify any full-layer optimum.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../submission/figures/fig07_exact_comparison.pdf}
\caption{Exact spatial-window comparison. Each size includes two deterministic spatial windows in each of ten retained geometries. Symbols show median gaps to the two-laser directed-edge optimum; top caps show maxima. There are 20 constant-speed and 40 finite-motion evaluations per size. All individual results and exact queues are supplied, including failed search improvements.}
\label{fig:checks}
\end{figure}

Full-layer search strengthens the common candidate set without making it globally optimal.
At the 52 fixed settings, reference-best brake/rest/ramp cost improves in 39 configurations, by at most @GAIN@\%.
After the searched plans are given to both selectors, positive choice losses increase from 13 to 42, with median 1.339\% and maximum @SEARCHMAX@\%.
Under optional cubic motion the count changes from 15 to 39, with median 1.278\% and maximum 11.249\%.
These increases concern relative decision loss: individual reference costs never increase when candidates are added.
The unequal-island configuration remains sensitive, with searched-plan losses of 11.236\% and 11.249\% under the two reference models.
Its new selected and alternative plans differ from the original S/B pair, so the old pair is retained only as the transparent mechanism illustration.
This experiment refutes the narrow explanation that the reported sensitivity exists only because the original portfolio lacks any assignment-and-order improvement; it does not show invariance to all stronger planners or controllers.

\subsection{Cross-geometry kinematic evidence}
Figure~\ref{fig:crossgeometry} includes every configuration at $(\kappa,\alpha)=(10,10)$ under the original 13-plan design.
There are 13 positive-loss cases out of 52.
In all 13, the selected plan's critical chain has more nonzero transitions and greater off-motion time than the alternative's critical chain.
Only three have a greater direct-distance sum, and twelve have a greater sum of endpoint-velocity changes.
The critical laser itself changes between constant-speed and finite-acceleration evaluation in 18 of the 52 selected plans.
Consequently, shortest travel distance and exposure balancing alone do not characterize the relevant critical chain in these settings.
This association is descriptive: jump count, direction changes and rest-point displacement are correlated, and different plans may be compared on different critical lasers.

Repeating all 52 comparisons at fixed $a=720{,}000$ and $V=6{,}000$ gives 13 positive losses, the same 11.041\% maximum, and a 6.604\% rather than 5.317\% P95 loss.
All 13 positive cases again have more critical-chain jumps; only three have greater direct travel.
The endpoint-change sum is larger in all 13 under this fixed-acceleration comparison.
The shared pattern therefore survives this particular dimensional-parameter control, without establishing a universal geometry-based decision rule.
All dimensional values remain in stored units.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../submission/figures/fig08_cross_geometry.pdf}
\caption{Cross-geometry mechanism diagnostics at $\kappa=10$, $\alpha=10$, $r=0$ with the original 13 plans. Each point compares the selected plan's own critical chain with the reference-best plan's own critical chain. Circles and triangles denote two and four lasers. Orange denotes positive loss; gray retains zero-loss cases, including coincident points. The descriptive associations are conditional on the motion convention and are not independent causal estimates.}
\label{fig:crossgeometry}
\end{figure}

Subdivision and tie sensitivity are separated from the main optimization comparison in Supplementary Figure~\ref{fig:representation}.
At fixed acceleration, splitting each S/B vector into two or four pieces changes cost by at most $1.12\times10^{-15}$ stored time units.
Holding $\alpha$ fixed instead changes dimensional acceleration and is not the same intervention.
For the expanded portfolio and optional cubic motion, reference-best/fixed-name/reference-worst tie rules give 91/94/97 positive settings, with the same 11.467\% maximum.
The primary result is the middle, executable rule.
\FloatBarrier

'''
    for a,b in {'@MED19@':f"{exact16['portfolio19']['median']:.3f}",'@MAX19@':f"{exact16['portfolio19']['maximum']:.3f}",'@MEDSEARCH@':f"{exact16['search']['median']:.3f}",'@MAXSEARCH@':f"{exact16['search']['maximum']:.3f}",'@GAIN@':f"{search['reference_improvement']['maximum']:.3f}",'@SEARCHMAX@':f"{search['after']['maximum']:.3f}"}.items():newresults=newresults.replace(a,b)
    t=replace_between(t,r'\subsection{What exact and representation checks change}',r'\section{Discussion}',newresults)
    t=t.replace('The exact comparisons use sparse 12-vector subsets and cannot certify full-layer global optimality.', 'The exact comparisons use nested spatial windows with at most 16 directed vectors and cannot certify full-layer global optimality. The bounded full-layer search is stronger than the initial portfolio but remains incomplete and uses one fixed computational seed.')
    t=t.replace('The headline result retains all stronger candidates.', 'The main sensitivity table retains all 19 candidates; the additional search experiment further tests candidate strength at a separately declared parameter setting.')
    t=t.replace('The resulting 20,748 plan--model evaluations and 5,616 derived comparisons took 40.78 seconds, with peak resident memory of 156.2 MB.\nOf these evaluations, 8,112 reproduce the historical zero-separation costs; the maximum discrepancy is $2.42\\times10^{-13}$ stored time units.\nThe repeated evaluations are consistency checks, not new independent inputs.', 'The revised run recomputes all 20,748 zero-separation plan--model evaluations and 5,616 derived comparisons.\nThe largest cost difference from the frozen historical zero-separation table is $1.43\\times10^{-14}$ stored time units.\nThese repeated evaluations are consistency checks, not new independent inputs; revised runtimes are reported below.')
    t=t.replace('The prospective extension exhibits an 11.04\\% order-bound counterexample, while the median selection loss remains zero in every tested group.', 'The 13-plan prospective extension exhibits an 11.04\\% order-bound counterexample, while its grouped median selection loss remains zero.')
    t=t.replace('Fixed-acceleration subdivision preserves the key reversal, while exact subset solutions expose nonzero portfolio gaps and prevent a global-optimality interpretation.', 'Fixed-acceleration controls preserve the key mechanism, while spatial-window exact solutions and stronger full-layer search expose portfolio gaps and retain decision sensitivity under a less restrictive candidate set.')
    repro=r'''\section*{Data and reproducibility}
The original vector inputs are from Zenodo record 12747089, version 2 \cite{data}; the record's API metadata identifies a CC BY 4.0 license.
The accompanying reproduction package freezes the two input workbooks, generated geometry arrays, candidate queues, configurations, seeds, Python dependency versions, complete per-setting tables, selected separation schedules and exact/search solutions.
SHA-256 hashes identify every distributed source and result file; upstream MD5 values additionally check the original workbooks.
The numerical entry point is \texttt{submission\_\allowbreak{}revision.py}; independent checks are in \texttt{verify\_\allowbreak{}submission.py}.
The figure and manuscript generators produce the PDF/SVG graphics and the main statistical tables directly from the frozen results.
The package includes reproduction instructions, an experiment audit and a dependency lock; historical results are retained separately and explicitly excluded where superseded.
Public deposition of the derived package has not yet been completed; no archive DOI or public repository is claimed.

'''
    t=replace_between(t,r'\section*{Data and reproducibility}',r'\appendix',repro)
    supplements=r'''\FloatBarrier
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theHfigure}{S\arabic{figure}}
\section{Representation checks and a non-beneficial cubic connection}
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../submission/figures/figS01_representation_checks.pdf}
\caption{Supplementary implementation sensitivities. (a) Fixed-queue subdivision preserves time only when dimensional acceleration is retained. (b) Reference-best and reference-worst ties bound the outcome-independent primary rule; they are not deployable selectors.}
\label{fig:representation}
\end{figure}
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../submission/figures/figS02_no_cubic_advantage.pdf}
\caption{Non-beneficial cubic control: the previous longest-duration transition in plan S. The available cubic is slightly slower than brake/rest/ramp, so the optional evaluator retains the latter. This edge is reported alongside the beneficial transition in Figure~\ref{fig:motion}; aggregate results include both kinds of edge.}
\label{fig:noadvantage}
\end{figure}
\FloatBarrier

'''
    t=t.replace(r'\begin{thebibliography}{99}',supplements+r'\begin{thebibliography}{99}')
    t=t.replace(r'\begin{thebibliography}{99}',r'\begin{thebibliography}{99}'+ '\n'+r'\raggedright')
    refs=r'''
\bibitem{livesu}
M. Livesu, S. Ellero, J. Mart\'inez, S. Lefebvre, and M. Attene.
From 3D models to 3D prints: an overview of the processing pipeline.
\emph{Computer Graphics Forum}, 36(2):537--564, 2017.
\href{https://doi.org/10.1111/cgf.13147}{doi:10.1111/cgf.13147}.

\bibitem{vns}
P. Hansen and N. Mladenovi\'c.
Variable neighborhood search: principles and applications.
\emph{European Journal of Operational Research}, 130(3):449--467, 2001.
\href{https://doi.org/10.1016/S0377-2217(00)00100-4}{doi:10.1016/S0377-2217(00)00100-4}.

\bibitem{ruckig}
L. Berscheid and T. Kr\"oger.
Jerk-limited real-time trajectory generation with arbitrary target states.
\emph{Robotics: Science and Systems XVII}, 2021.
\href{https://doi.org/10.15607/RSS.2021.XVII.015}{doi:10.15607/RSS.2021.XVII.015}.
'''
    t=t.replace(r'\end{thebibliography}',refs+'\n'+r'\end{thebibliography}')
    # Bibliography ordered by first citation, not by editorial insertion order.
    start=t.index(r'\begin{thebibliography}{99}')+len(r'\begin{thebibliography}{99}');end=t.index(r'\end{thebibliography}')
    entries={m.group(1):m.group(0).strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}.*?(?=\\bibitem|\Z)',t[start:end],re.S)}
    order=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',t[:start]):
        for key in group.split(','):
            if key not in order:order.append(key)
    assert set(order)==set(entries),(set(entries)-set(order),set(order)-set(entries))
    t=t[:start]+'\n'+r'\raggedright'+'\n\n'+'\n\n'.join(entries[k] for k in order)+'\n\n'+t[end:]
    # Cross-reference integrity is also checked after compilation.
    labels=re.findall(r'\\label\{([^}]+)\}',t);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',t)
    assert len(labels)==len(set(labels));assert not set(refs)-set(labels)
    assert 'Revised working manuscript' not in t
    (ROOT/'paper/main.tex').write_text(t)
    print(json.dumps({'references':len(order),'labels':len(labels),'source_chars':len(t),'generated_tables':4}))

if __name__=='__main__':main()
