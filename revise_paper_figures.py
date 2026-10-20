"""Data-driven manuscript figures. No inferred thermal field or printed geometry.

Rendered thickness, powder texture and illumination are display conventions only.
Planar polygons, assignments, transition paths and all plotted costs come from
the retained computational cases and fixed execution models.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from pilot import ROOT, Jobs
from geometry_portfolio import assignment_of, free_starts, sections
from execution_cost import setup_times
from motion_fidelity import motion_setup, motion_phases
from motion_portfolio_sensitivity import motion_parameters, endpoint_sample, hermite_duration

OUT = ROOT / "figures" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / "paper" / "images"
PAPER.mkdir(exist_ok=True)
INK = "#243746"
MUTED = "#586977"
SCAN = "#a7bbc9"
OFF = "#e87824"
BLUE = "#17668e"
TEAL = "#128579"
LASERS = ["#0072B2", "#D55E00", "#009E73", "#8D63A6"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "axes.titleweight": "bold", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "grid.linewidth": .5, "grid.alpha": .2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", dpi=300)
    (PAPER / f"{name}.pdf").write_bytes((OUT / f"{name}.pdf").read_bytes())
    plt.close(fig)


def witness():
    name = "unequal_islands_75"
    z = np.load(ROOT / "prospective_geometry_inputs.npz")
    a, b, p, d = [z[name + "_" + k] for k in ["a", "b", "p", "dedicated"]]
    jobs = Jobs(np.arange(len(p)), a, b, p, d)
    polygons = [z[k] for k in z.files if k.startswith(name + "_polygon_")]
    plans = json.loads((ROOT / "prospective_geometry_plans.json").read_text())[name + "_m4"]
    result = []
    for role, label in [("S", "region_y_file"), ("B", "region_x_nearest_neighbor")]:
        queues = plans[label]
        cv = setup_times(jobs, queues, 10., 0.)[0]
        fa = motion_setup(jobs, queues, 10., 10.)[0]
        work = np.array([jobs.p[q].sum() for q in queues])
        csum = np.array([cv[q].sum() for q in queues])
        fsum = np.array([fa[q].sum() for q in queues])
        result.append(dict(role=role, label=label, queues=queues, work=work,
                           cv=cv, fa=fa, cv_sum=csum, fa_sum=fsum,
                           critical=int(np.argmax(work + fsum)),
                           assignment=assignment_of(queues, len(p))))
    np.testing.assert_allclose([max(r["work"] + r["fa_sum"]) for r in result],
                               [.8263108905613437, .7441515570949414], atol=1e-12)
    return jobs, polygons, result


def phase_samples(jobs, i, j, kappa=10., alpha=10.):
    v, _, cap, acc = motion_parameters(jobs, kappa, alpha)
    phases = motion_phases(jobs.b[i], v[i], jobs.a[j], v[j], cap, acc)
    x = jobs.b[i].copy(); vel = v[i].copy(); t0 = 0.
    xx, tt, vv, aa = [], [], [], []
    for dt, acceleration in phases:
        if dt <= 1e-14:
            continue
        t = np.linspace(0, dt, 24)
        xx.extend(x + t[:, None] * vel + .5 * t[:, None]**2 * acceleration)
        vv.extend(vel + t[:, None] * acceleration)
        aa.extend(np.repeat(acceleration[None], len(t), axis=0))
        tt.extend(t + t0)
        x += vel * dt + .5 * acceleration * dt**2
        vel += acceleration * dt
        t0 += dt
    if len(xx):
        assert np.linalg.norm(x - jobs.a[j]) < 1e-8
        assert np.linalg.norm(vel - v[j]) < 1e-8
    return np.asarray(tt), np.asarray(xx), np.asarray(vv), np.asarray(aa)


def extrude(ax, poly, height=3.6, base=.05, color="#c9cfd3", zorder=3):
    """Convex polygons only here; heights are illustrative, not measured."""
    n = len(poly)
    top = np.column_stack([poly, np.full(n, height)])
    bottom = np.column_stack([poly, np.full(n, base)])
    sides = [[bottom[i], bottom[(i+1) % n], top[(i+1) % n], top[i]] for i in range(n)]
    side = Poly3DCollection(sides, facecolors="#89949b", edgecolors="#73818a",
                            linewidths=.18, shade=True,
                            lightsource=LightSource(azdeg=315, altdeg=40), zorder=zorder)
    ax.add_collection3d(side)
    face = Poly3DCollection([top], facecolors=color, edgecolors="#71808b",
                            linewidths=.32, shade=True,
                            lightsource=LightSource(azdeg=315, altdeg=55), zorder=zorder+.1)
    ax.add_collection3d(face)
    # Small lip provides shape perception without changing planar footprints.
    ax.plot(*np.vstack([top, top[0]]).T, color="#f2f5f7", linewidth=.42, zorder=zorder+.2)


def scene(ax, jobs, polygons, rec, critical=True, bed=True):
    """Same camera, polygons and display height for both candidate plans."""
    height = 3.6
    bounds = np.concatenate(polygons)
    lo = bounds.min(0) - 8; hi = bounds.max(0) + 8
    plate = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    if bed:
        extrude(ax, plate, height=-.15, base=-2.4, color="#d6dadd", zorder=0)
        rng = np.random.default_rng(7301)
        x = np.linspace(lo[0], hi[0], 90); y = np.linspace(lo[1], hi[1], 80)
        X, Y = np.meshgrid(x, y)
        tone = np.clip(.83 + rng.normal(0, .014, X.shape), .77, .88)
        fc = np.stack([tone*.96, tone*.98, tone, np.ones_like(tone)], axis=-1)
        ax.plot_surface(X, Y, np.full_like(X, -.10), facecolors=fc,
                        rstride=1, cstride=1, shade=False, linewidth=0,
                        antialiased=False, zorder=.3)
    for poly in polygons:
        shadow = np.column_stack([poly + [1.3, 1.8], np.full(len(poly), -.04)])
        ax.add_collection3d(Poly3DCollection([shadow], facecolors=(.16,.20,.23,.14),
                                            edgecolors="none", zorder=1))
        extrude(ax, poly, height=height)
    segments = np.stack([jobs.a, jobs.b], axis=1)
    xyz = np.concatenate([segments, np.full((*segments.shape[:2], 1), height+.04)], axis=2)
    colors = [to_rgba(LASERS[k], .97) for k in rec["assignment"]]
    ax.add_collection3d(Line3DCollection(xyz, colors=colors, linewidths=.64, zorder=7))
    if critical:
        q = rec["queues"][rec["critical"]]
        for i, j in zip(q[:-1], q[1:]):
            if rec["fa"][j] <= 1e-10:
                continue
            _, xy, _, _ = phase_samples(jobs, i, j)
            ax.plot(xy[:,0], xy[:,1], np.full(len(xy), height+.14),
                    color="#483d4f", lw=.48, alpha=.47, zorder=8)
        # The highlighted curve is one actual, maximum-duration transition.
        j = max(q[1:], key=lambda u: rec["fa"][u]); i = q[q.index(j)-1]
        _, xy, _, _ = phase_samples(jobs, i, j)
        ax.plot(xy[:,0], xy[:,1], np.full(len(xy), height+.2),
                color="#9b2346", lw=1.65, zorder=9)
        ax.scatter([jobs.b[i,0],jobs.a[j,0]], [jobs.b[i,1],jobs.a[j,1]],
                   [height+.24]*2, s=9, c=["#9b2346"]*2, edgecolor="white", linewidth=.3, zorder=10)
    # First exposure of each laser; annotations show actual sequence origins.
    for ell, q in enumerate(rec["queues"]):
        if q:
            x, y = jobs.a[q[0]]
            ax.scatter([x], [y], [height+.25], s=13, c=LASERS[ell],
                       edgecolors="white", linewidths=.5, zorder=10)
    # No mm label: these are the source coordinate units.
    sx, sy = lo[0]+11, lo[1]+3
    ax.plot([sx,sx+10], [sy,sy], [.04,.04], color=INK, lw=1.4, zorder=5)
    ax.text(sx+5, sy-1, .05, "10 units", fontsize=5.5, ha="center", va="top", zorder=11)
    ax.set_xlim(lo[0]-2,hi[0]+2); ax.set_ylim(lo[1]-2,hi[1]+2); ax.set_zlim(-3,9)
    ax.set_box_aspect((hi[0]-lo[0],hi[1]-lo[1],12), zoom=1.23)
    ax.view_init(elev=44, azim=-62)
    ax.set_proj_type("ortho"); ax.set_axis_off()
    ax.set_rasterization_zorder(30)


def main_mechanism():
    jobs, polygons, rr = witness()
    fig = plt.figure(figsize=(7.25, 8.0))
    for k, r in enumerate(rr):
        x = .035 + .5*k
        title = "Plan S: y-partition / input order" if k == 0 else "Plan B: x-partition / nearest neighbor"
        fig.text(x, .966, chr(97+k) + "  " + title, fontsize=9, weight="bold")
        ax = fig.add_axes([x-.035, .598, .49, .365], projection="3d", computed_zorder=False)
        scene(ax, jobs, polygons, r)
        critical_q = r["queues"][r["critical"]]
        nonzero = int(sum(r["fa"][critical_q] > 1e-10))
        fig.text(x+.22, .608, f"Critical laser: L{r['critical']+1}  |  {nonzero} off-transitions",
                 ha="center", fontsize=8, color=MUTED)
    fig.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f"L{i+1}") for i,c in enumerate(LASERS)] +
               [Line2D([0],[0],color="#483d4f",lw=1,label="Critical-laser off path")],
               loc="center", bbox_to_anchor=(.5,.578), ncol=5, fontsize=7, columnspacing=1.1,
               handlelength=1.5)
    # All exposure and off-time intervals are retained, including thin intervals.
    for k, r in enumerate(rr):
        ax = fig.add_axes([.075+.5*k,.372,.395,.165])
        starts = free_starts(jobs,r["queues"],r["fa"])
        for ell, q in enumerate(r["queues"]):
            y=3-ell
            ax.broken_barh([(float(starts[j]),float(jobs.p[j])) for j in q],
                          (y-.29,.58),facecolors=SCAN,edgecolors="none")
            ax.broken_barh([(float(starts[j]-r["fa"][j]),float(r["fa"][j])) for j in q if r["fa"][j]>0],
                          (y-.29,.58),facecolors=OFF,edgecolors="none")
            if ell==r["critical"]:
                ax.add_patch(Rectangle((0,y-.34),max(starts[q]+jobs.p[q]),.68,
                                       fill=False,edgecolor=INK,lw=.65))
        ax.set_xlim(0,.85);ax.set_ylim(-.5,3.65)
        ax.set_yticks(range(4), ["L4","L3","L2","L1"])
        ax.set_xticks([0,.2,.4,.6,.8]);ax.set_xlabel("Calculated time (stored units)",fontsize=7.5)
        ax.set_title(chr(99+k)+f"  Plan {r['role']}: full four-laser schedule",loc="left",fontsize=8.4,pad=6)
        ax.grid(axis="x",zorder=0)
        for label,ell in zip(ax.get_yticklabels(),[3,2,1,0]):label.set_color(LASERS[ell])
    ax=fig.add_axes([.20,.085,.69,.19])
    for y,model,r in [(3,"cv",rr[0]),(2.25,"cv",rr[1]),(.85,"fa",rr[0]),(.10,"fa",rr[1])]:
        loads=r["work"]+r[model+"_sum"]
        ell=int(np.argmax(loads));scan=float(r["work"][ell]);off=float(r[model+"_sum"][ell])
        ax.barh(y,scan,height=.49,color=SCAN)
        ax.barh(y,off,left=scan,height=.49,color=OFF)
        ax.text(scan+off+.012,y,f"{scan+off:.3f}",va="center",weight="bold",fontsize=9)
    ax.set_yticks([3,2.25,.85,.10],["Plan S","Plan B","Plan S","Plan B"])
    ax.set_xlim(0,.95);ax.set_ylim(-.4,3.8);ax.set_xticks([0,.2,.4,.6,.8])
    ax.grid(axis="x");ax.set_axisbelow(True)
    ax.set_xlabel("Critical-laser exposure + laser-off motion (stored time units)",fontsize=7.5)
    ax.text(0,3.62,"e  Constant-speed model",fontsize=8.5,weight="bold")
    ax.text(0,1.48,"Finite-acceleration model",fontsize=8.5,weight="bold")
    ax.text(.70,3,"S selected",color=MUTED,fontsize=8,va="center")
    ax.text(.55,1.48,"S costs 11.04% more than B",fontsize=8,color="#a64c13",weight="bold")
    fig.legend(handles=[Rectangle((0,0),1,1,color=SCAN,label="Exposure"),
                        Rectangle((0,0),1,1,color=OFF,label="Laser-off motion")],
               loc="center",bbox_to_anchor=(.53,.317),ncol=2,fontsize=8)
    fig.text(.5,.018,"Actual planar scan geometry; rendered thickness and surface texture are illustrative. No thermal field is shown.",
             ha="center",fontsize=6.8,color=MUTED)
    save(fig,"fig03_reversal_mechanism")
    evidence=[]
    for r in rr:
        evidence.append({k:(v.tolist() if isinstance(v,np.ndarray) else v)
                         for k,v in r.items() if k not in ["assignment","cv","fa","queues"]})
    (OUT/"mechanism_values.json").write_text(json.dumps(evidence,indent=2))


def framework():
    from prospective_geometry import new_sections
    from geometry_portfolio import hatch, spatial_queues
    polygons=new_sections()["U_bracket"]
    jobs=hatch(polygons,45);queues=spatial_queues(jobs,4,0)
    rec={"queues":queues,"assignment":assignment_of(queues,len(jobs.p))}
    fig=plt.figure(figsize=(7.25,4.55))
    ax=fig.add_axes([.00,.36,.40,.61],projection="3d",computed_zorder=False)
    scene(ax,jobs,polygons,rec,critical=False)
    ax.set_box_aspect((64,56,12),zoom=1.10)
    fig.text(.035,.958,"a  Fixed geometry and directed exposures",weight="bold",fontsize=9)
    fig.text(.20,.395,"Rendered U-bracket case, 45° hatching",ha="center",fontsize=7.5,color=MUTED)
    ax=fig.add_axes([.43,.37,.55,.58]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x,y,w,h,title,body,face):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01,rounding_size=.015",
                                   facecolor=face,edgecolor="#a9b8c1",linewidth=.65))
        ax.text(x+.035,y+h-.06,title,va="top",weight="bold",fontsize=8)
        ax.text(x+.035,y+h-.14,body,va="top",fontsize=7.0,linespacing=1.25)
    ax.text(0,1.03,"b  Change the evaluator; retain the candidates",weight="bold",fontsize=9)
    box(.01,.73,.95,.21,"Common portfolio","Same assignments, vector directions and queue orders", "#eff3f5")
    box(.01,.36,.44,.25,"Simplified evaluator","Constant-speed\nlaser-off travel", "#edf4f7")
    box(.52,.36,.44,.25,"Reference evaluator","Brake/rest/ramp\nOptional cubic connection", "#fbf0e7")
    box(.01,.015,.95,.22,"Audit the resulting decisions","Timing bias  |  Selection loss  |  Fixed-order bound", "#edf5f1")
    for a,b in [((.23,.73),(.23,.62)),((.74,.73),(.74,.62)),((.23,.36),(.23,.25)),((.74,.36),(.74,.25))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color=MUTED,linewidth=.8))
    fig.text(.035,.31,"c  Computational inputs",weight="bold",fontsize=9)
    all_shapes={**sections(),**new_sections()}
    names=["Rectangle","L section","Annulus","Equal islands","U bracket","Dogbone","Perforated plate","Unequal islands"]
    for k,((key,poly),label) in enumerate(zip(all_shapes.items(),names)):
        bx=fig.add_axes([.035+k*.118,.13,.098,.14]);bx.set_aspect("equal");bx.axis("off")
        # Use actual hatching rather than inventing a filled mask at holes.
        angle=30 if k<4 else 45
        jj=hatch(poly,angle)
        from matplotlib.collections import LineCollection
        bx.add_collection(LineCollection(np.stack([jj.a,jj.b],axis=1),colors=BLUE,linewidths=.3))
        for polygon in poly:bx.plot(*np.vstack([polygon,polygon[0]]).T,lw=.5,color=INK)
        bx.autoscale_view();bx.set_title(label,fontsize=6.4,pad=3,weight="normal")
    fig.text(.5,.062,"8 constructed families × 3 angles + 2 released layers  →  26 cases × 2 laser counts",
             ha="center",fontsize=8)
    fig.text(.5,.017,"Exposure geometry is fixed throughout; the render adds only illustrative thickness and shading.",
             ha="center",fontsize=6.8,color=MUTED)
    save(fig,"fig01_decision_framework")


def motion_comparison():
    jobs,_,rr=witness();r=rr[0];q=r["queues"][r["critical"]]
    j=max(q[1:],key=lambda k:r["fa"][k]);i=q[q.index(j)-1]
    v,s,V,a=motion_parameters(jobs,10.,10.)
    d=jobs.a[j]-jobs.b[i];origin=jobs.b[i]
    tt,xx,vv,aa=phase_samples(jobs,i,j)
    tf=float(tt[-1]);tc=float(np.linalg.norm(d)/V)
    th=float(hermite_duration(d[None],v[i:i+1],v[j:j+1],V,a)[0])
    tau=np.linspace(0,1,501);xh,vh,ah=endpoint_sample(d,v[i],v[j],th,tau)
    models=[("Constant-speed jump",np.linspace([0.,0.],d,100),np.linspace(0,tc,100),
             np.full(100,V),None,BLUE),
            ("Brake / rest / ramp",xx-origin,tt,np.linalg.norm(vv,axis=1),np.linalg.norm(aa,axis=1),OFF),
            ("Feasible cubic option",xh,tau*th,np.linalg.norm(vh,axis=1),np.linalg.norm(ah,axis=1),TEAL)]
    fig,axes=plt.subplots(3,3,figsize=(7.25,5.65),gridspec_kw={"height_ratios":[1.4,1,1]})
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.27,hspace=.60)
    allxy=np.vstack([jobs.a[i]-origin,jobs.b[j]-origin,xx-origin,xh]);lo=allxy.min(0)-1.5;hi=allxy.max(0)+1.5
    for col,(name,path,times,speeds,accs,color) in enumerate(models):
        ax=axes[0,col]
        ax.plot(*np.vstack([jobs.a[i]-origin,[0,0]]).T,lw=2.8,color=SCAN)
        ax.plot(*np.vstack([d,jobs.b[j]-origin]).T,lw=2.8,color=SCAN)
        ax.plot(*path.T,lw=1.7,color=color)
        ax.scatter([0,d[0]],[0,d[1]],s=18,c=INK,zorder=8)
        for x,vel,label in [(np.zeros(2),v[i],"$v_i$"),(d,v[j],"$v_j$")]:
            delta=vel/np.linalg.norm(vel)*2
            ax.annotate("",xy=x+delta,xytext=x,arrowprops=dict(arrowstyle="->",color=INK,lw=.8))
            ax.text(*(x+delta+[.6,.1]),label,fontsize=8)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect("equal")
        ax.set_title(chr(97+col)+"  "+name,loc="left",fontsize=8.2)
        ax.set_xlabel("Relative x (stored units)",fontsize=7)
        if col==0:ax.set_ylabel("Relative y")
        ax.tick_params(labelsize=7)
        ax=axes[1,col];ax.plot(times/tf,speeds/s[i],color=color,lw=1.6)
        ax.axhline(V/s[i],color="#a7afb5",ls=":",lw=.7)
        if col==0:
            for t in [0,tc/tf]:ax.plot([t,t],[1,V/s[i]],ls="--",lw=.9,color=color)
            ax.scatter([0,tc/tf],[1,1],s=13,c=INK,zorder=5)
            ax.text(.45,6.7,"Endpoint velocity\nnot enforced",fontsize=7.2,color=MUTED)
        ax.set_ylim(-.2,11);ax.set_xlim(-.025,1.13);ax.grid(axis="y")
        ax.set_title(f"Duration / brake duration = {times[-1]/tf:.3f}",fontsize=7.1,weight="normal")
        if col==0:ax.set_ylabel("Speed / exposure speed")
        ax.tick_params(labelsize=7)
        ax=axes[2,col]
        if accs is None:
            ax.text(.5,.52,"Acceleration is\nunconstrained",transform=ax.transAxes,ha="center",fontsize=8,color=MUTED)
        else:
            ax.plot(times/tf,accs/a,color=color,lw=1.6)
        ax.axhline(1,color="#a7afb5",ls=":",lw=.7)
        ax.set_xlim(-.025,1.13);ax.set_ylim(-.05,1.15);ax.set_xlabel("Time / brake duration",fontsize=7.5)
        if col==0:ax.set_ylabel("Acceleration / limit")
        ax.tick_params(labelsize=7);ax.grid(axis="y")
    fig.text(.5,.016,"One retained transition from Plan S; prescribed kinematic models, not a measured scanner response.",
             ha="center",fontsize=6.8,color=MUTED)
    save(fig,"fig02_motion_assumptions")
    (OUT/"motion_illustration_values.json").write_text(json.dumps({"previous_job":int(i),"next_job":int(j),
        "constant_speed_duration":tc,"brake_duration":tf,"feasible_cubic_duration":th,
        "cap":V,"acceleration":a,"units":"consistent stored units"},indent=2))


def continuous_envelope():
    d=json.loads((ROOT/"execution_cost_envelopes.json").read_text())["details"]["Layer 10_m4"]
    a,b=np.array(d["A_scan"]),np.array(d["B_jump"]);names=d["plan_names"]
    q=np.linspace(.05,.2,2003);value=(a[:,:,None]+b[:,:,None]*q).max(axis=1)
    best=value.min(0);selected=names.index("append_EFT_5")
    fig,ax=plt.subplots(1,3,figsize=(7.25,3.0),gridspec_kw={"width_ratios":[1,1,1]})
    fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.86,wspace=.48)
    for row in value:ax[0].plot(q,row,color="#afbac2",lw=.75)
    ax[0].plot(q,best,color=INK,lw=1.7,label="Portfolio minimum")
    ax[0].set_ylabel("Calculated makespan (stored units)")
    ax[0].set_title("a  All 13 candidates",loc="left",fontsize=8.6)
    for label,color in [("append_EFT_5",BLUE),("append_EFT_10",OFF),("append_EFT_20",TEAL)]:
        idx=names.index(label)
        ax[1].plot(q,value[idx],color=color,lw=1.4,label=label.replace("append_",""))
    ax[1].plot(q,best,color=INK,lw=1,ls="--",label="Minimum")
    ax[1].set_title("b  Competitive candidates",loc="left",fontsize=8.6)
    ax[1].legend(fontsize=6.4,loc="upper left",labelspacing=.25,handlelength=1.5)
    winner=np.argmin(value,axis=0);changes=np.flatnonzero(winner[1:]!=winner[:-1])+1
    for idx in changes:
        ax[1].axvline(q[idx],lw=.6,ls=":",color=MUTED)
    curve=100*(value[selected]/best-1)
    ax[2].fill_between(q,0,curve,color=BLUE,alpha=.13)
    ax[2].plot(q,curve,color=BLUE,lw=1.7)
    points=json.loads((ROOT/"constant_speed_regret_curves.json").read_text())["Layer 10_m4"]
    qp=np.array(points["q"]);rp=np.array(points["regret_selected_pct"])
    ax[2].plot(qp,rp,"o",ms=2.3,color=BLUE)
    ax[2].set_ylabel("Relative selection loss (%)")
    ax[2].set_ylim(-.04,1.6);ax[2].set_title("c  One fixed plan: EFT_5",loc="left",fontsize=8.6)
    ax[2].annotate("Worst bound: 1.3411%",xy=(.05,rp[0]),xytext=(.076,1.43),
                   fontsize=7,arrowprops=dict(arrowstyle="-",color=MUTED,lw=.6))
    for axis in ax:
        axis.set_xlim(.05,.2);axis.set_xticks([.05,.1,.15,.2]);axis.tick_params(labelsize=7)
        axis.set_xlabel(r"Inverse speed ratio $q=1/\kappa$");axis.grid(alpha=.2)
    fig.text(.5,.045,"Layer 10, four lasers, no separation waiting. Dots evaluate affine-cell boundaries; curves share the same portfolio.",
             fontsize=6.7,ha="center",color=MUTED)
    save(fig,"fig04_continuous_envelope")


def authoritative_comparisons():
    rows=[]
    for name in ["motion_fidelity_results.json","prospective_geometry_results.json"]:
        for r in json.loads((ROOT/name).read_text())["comparisons"]:
            if r["radius"]!=0:continue
            r=dict(r)
            if "observed_selection_loss_pct" not in r:
                r["observed_selection_loss_pct"]=r["constant_speed_selection_regret_pct"]
            rows.append(r)
    rows.extend(json.loads((ROOT/"interrupted_motion_results.json").read_text())["comparisons"])
    assert len(rows)==624
    return rows


def discrete_map():
    rows=authoritative_comparisons()
    early=[f"{f}_{a}" for f in ["rectangle","concave_L","annulus","four_islands"] for a in [0,30,60]]
    later=[f"{f}_{a}" for f in ["U_bracket","dogbone","perforated_plate","unequal_islands"] for a in [15,45,75]]
    cases=early+["Layer 10","Nozzle"]+later
    columns=[(0,k,a) for a in [10,100,1000] for k in [5,10,20]]+[(8,10,a) for a in [10,100,1000]]
    fig=plt.figure(figsize=(7.25,6.3));axes=[fig.add_axes([.21,.185,.36,.67]),fig.add_axes([.62,.185,.36,.67])]
    images=[]
    for ax,m in zip(axes,[2,4]):
        lookup={(r["case"],r["radius"],r["kappa"],r["alpha"]):r["observed_selection_loss_pct"] for r in rows if r["lasers"]==m}
        matrix=np.array([[lookup[(name,*col)] for col in columns] for name in cases])
        # Linear color scale, no interpolation between untested settings.
        images.append(ax.imshow(matrix,aspect="auto",interpolation="nearest",cmap="YlOrBr",vmin=0,vmax=18))
        for iy,ix in np.argwhere(matrix>1e-6):
            txt="<.1" if matrix[iy,ix]<.05 else f"{matrix[iy,ix]:.1f}"
            ax.text(ix,iy,txt,ha="center",va="center",fontsize=7.0,
                    color="white" if matrix[iy,ix]>8 else INK)
        ax.axvline(8.5,color=INK,lw=1)
        for yy in [2.5,5.5,8.5,11.5,13.5,16.5,19.5,22.5]:ax.axhline(yy,color="white",lw=1.3)
        ax.set_xticks(range(12),["5","10","20"]*3+["10"]*3,fontsize=6)
        ax.set_yticks(range(26),[s.replace("_"," ") for s in cases] if m==2 else [""]*26,fontsize=6.7)
        ax.set_title(f"{chr(97+(m==4))}  {m} lasers",loc="left",fontsize=9,pad=39)
        for x,txt in [(1,r"$\alpha=10$"),(4,"100"),(7,"1000"),(9,"10"),(10,"100"),(11,"1000")]:
            ax.text(x,-.9,txt,fontsize=6,ha="center",clip_on=False)
        ax.text(4,-1.9,"r = 0",fontsize=7,ha="center",clip_on=False,color=MUTED)
        ax.text(10,-1.9,"r = 8",fontsize=7,ha="center",clip_on=False,color=MUTED)
        ax.set_xlabel(r"Speed ratio $\kappa$",fontsize=8)
        for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(images[1],cax=fig.add_axes([.26,.073,.48,.018]),orientation="horizontal")
    cb.set_label("Observed selection loss (%)",fontsize=8,labelpad=3);cb.ax.tick_params(labelsize=7)
    fig.text(.5,.010,"Original 13 plans; brake/rest/ramp reference. Blank cells have zero loss. All 624 settings are retained.",
             ha="center",fontsize=6.7,color=MUTED)
    save(fig,"fig05_discrete_fidelity_map")


def robustness():
    data=json.loads((ROOT/"motion_portfolio_sensitivity_results.json").read_text())["comparisons"]
    def index(model,portfolio):
        return {(r["case"],r["lasers"],r["kappa"],r["alpha"]):r for r in data if r["model"]==model and r["portfolio"]==portfolio}
    base=index("brake_rest_ramp","original_13");cubic=index("optional_cubic","original_13");expanded=index("optional_cubic","expanded_19")
    fig=plt.figure(figsize=(7.25,5.9))
    matrices=[]
    for col,(left,right,title) in enumerate([(base,cubic,"Motion family: brake → optional cubic"),(cubic,expanded,"Portfolio: 13 → 19 plans")]):
        ax=fig.add_axes([.09+.50*col,.57,.34,.29])
        mat=np.zeros((2,2),int)
        for k in left:
            x=int(left[k]["selection_loss_pct"]>1e-6);y=int(right[k]["selection_loss_pct"]>1e-6);mat[y,x]+=1
        matrices.append(mat.tolist())
        ax.imshow(mat,cmap="Blues",vmin=0,vmax=400)
        for y in range(2):
            for x in range(2):ax.text(x,y,str(mat[y,x]),ha="center",va="center",weight="bold",fontsize=14,color="white" if mat[y,x]>250 else INK)
        ax.set_xticks([0,1],["Zero","Positive"]);ax.set_yticks([0,1],["Zero","Positive"])
        ax.set_xlabel("Before change");ax.set_ylabel("After change")
        ax.set_title(chr(97+col)+"  "+title,fontsize=8.2,loc="left",pad=12)
        ax=fig.add_axes([.09+.50*col,.15,.34,.27])
        changed=[(k,right[k]["selection_loss_pct"]-left[k]["selection_loss_pct"]) for k in left
                 if abs(right[k]["selection_loss_pct"]-left[k]["selection_loss_pct"])>1e-6]
        changed.sort(key=lambda x:x[1])
        for a,color in [(10,OFF),(100,BLUE),(1000,TEAL)]:
            pos=[i for i,(k,v) in enumerate(changed) if k[-1]==a]
            ax.scatter(pos,[changed[i][1] for i in pos],s=11,color=color,label=fr"$\alpha={a}$",zorder=4)
        ax.axhline(0,color=INK,lw=.65)
        ax.set_ylabel("Loss change (percentage points)")
        ax.set_xlabel("Changed settings, ordered by difference")
        ax.set_title(chr(99+col)+f"  {len(changed)} changed; {468-len(changed)} unchanged",fontsize=8.2,loc="left")
        ax.grid(axis="y");ax.tick_params(labelsize=7)
    fig.legend(handles=[Line2D([0],[0],marker="o",ls="",color=c,label=fr"$\alpha={a}$",markersize=4)
                        for a,c in [(10,OFF),(100,BLUE),(1000,TEAL)]],loc="center",bbox_to_anchor=(.5,.04),ncol=3,fontsize=7.5)
    save(fig,"fig06_robustness")
    (OUT/"robustness_counts.json").write_text(json.dumps({"rows":"after_zero/positive","columns":"before_zero/positive","matrices":matrices},indent=2))


def validation_figure():
    d=json.loads((ROOT/"revision_validation_results.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.25,3.65))
    fig.subplots_adjust(left=.08,right=.98,bottom=.30,top=.86,wspace=.52)
    ax=axes[0]
    for role,color in [("S",OFF),("B",BLUE)]:
        rows=[r for r in d["segmentation"] if r["plan"]==role]
        x=[r["split_each_vector_into"] for r in rows]
        ax.plot(x,[r["fixed_acceleration_cost"] for r in rows],"o-",ms=3,color=color,label=f"{role}: fixed a")
        ax.plot(x,[r["fixed_alpha_cost"] for r in rows],"s--",ms=3,color=color,label=f"{role}: fixed α")
    ax.set_xticks([1,2,4]);ax.set_xlabel("Equal pieces per original vector")
    ax.set_ylabel("Calculated makespan (stored units)");ax.set_title("a  Segmentation control",loc="left",fontsize=8.5)
    ax.legend(fontsize=6.5,labelspacing=.3,loc="upper center",bbox_to_anchor=(.5,-.30),ncol=2,columnspacing=.8,handlelength=1.6);ax.grid(axis="y")
    ax=axes[1]
    models=["constant_speed","brake_rest_ramp","optional_cubic"]
    for ix,model in enumerate(models):
        rows=[r for r in d["small_exact_results"] if r["model"]==model]
        gaps=np.array([r["gap_19_pct"] for r in rows]);offset=np.linspace(-.12,.12,len(gaps))
        ax.scatter(ix+offset,gaps,s=13,color=[MUTED,OFF,TEAL][ix],alpha=.8)
        ax.plot([ix-.23,ix+.23],[np.median(gaps)]*2,color=INK,lw=1.5)
    ax.set_xticks(range(3),["CV\n(n=10)","Brake\n(n=20)","Cubic\n(n=20)"],fontsize=7)
    ax.set_ylabel("Portfolio best vs. exact optimum (%)")
    ax.set_title("b  Exact 12-vector subproblems",loc="left",fontsize=8.2)
    ax.set_ylim(-.3,14);ax.grid(axis="y")
    ax.text(.5,.96,"13- and 19-plan gaps coincide",fontsize=6.4,ha="center",transform=ax.transAxes,va="top",color=MUTED)
    ax=axes[2]
    vals=[]
    for rule in ["reference_best_tie","lexicographic_tie","reference_worst_tie"]:
        rows=[r for r in d["tie_sensitivity"] if r["model"]=="optional_cubic" and r["portfolio"]=="expanded_19"]
        vals.append(sum(r["loss_pct"][rule]>1e-6 for r in rows))
    ax.barh([2,1,0],vals,color=[SCAN,BLUE,MUTED],height=.54)
    for y,value in zip([2,1,0],vals):ax.text(value+1,y,str(value),va="center",fontsize=8)
    ax.set_yticks([2,1,0],["Best tie","Fixed name","Worst tie"],fontsize=7)
    ax.set_xlim(0,115);ax.set_xlabel("Positive losses / 468 settings")
    ax.set_title("c  Selection-tie sensitivity",loc="left",fontsize=8.5)
    ax.grid(axis="x");ax.set_axisbelow(True)
    fig.text(.5,.035,"Tiny instances are selected computational subsets, not full-layer optimality tests or independent manufacturing experiments.",
             fontsize=6.5,ha="center",color=MUTED)
    save(fig,"fig07_validation")


if __name__=="__main__":
    main_mechanism()
    framework()
    motion_comparison()
    continuous_envelope()
    discrete_map()
    robustness()
    validation_figure()
    source_names=["prospective_geometry_inputs.npz","prospective_geometry_plans.json",
                  "execution_cost_envelopes.json","constant_speed_regret_curves.json",
                  "motion_portfolio_sensitivity_results.json","interrupted_motion_results.json",
                  "revision_validation_results.json"]
    (OUT/"figure_provenance.json").write_text(json.dumps({
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_names},
        "render_semantics":"Planar geometry and quantitative overlays are computed. Extrusion, baseplate, illumination and surface texture are illustrative. No thermal or printed-surface data.",
        "output_formats":["PDF: quantitative panels vector, 3D panels raster","PNG: 300 dpi"],
        "figure_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
