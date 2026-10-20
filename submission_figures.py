"""Editable vector figures from submission_revision outputs; no fabricated fields."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon, ConnectionPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
import revise_paper_figures as old
from revise_paper_figures import witness,phase_samples,INK,MUTED,SCAN,OFF,BLUE,TEAL,LASERS
from pilot import ROOT
from geometry_portfolio import free_starts
from motion_portfolio_sensitivity import motion_parameters,hermite_duration,endpoint_sample
from submission_revision import OUT,load

FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,
    'axes.labelsize':8,'axes.titlesize':9,'savefig.dpi':220,'pdf.fonttype':42})

def save(fig,name):
    # No bitmap in any deliverable vector graphic, including the 3-D framework.
    for ax in fig.axes:ax.set_rasterization_zorder(None)
    for obj in fig.findobj():
        if hasattr(obj,'set_rasterized'):obj.set_rasterized(False)
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',facecolor='white',dpi=220)
    plt.close(fig)

def motion(edge=None,name='fig02_motion_assumptions'):
    jobs,_,rr=witness();vel,s,cap,acc=motion_parameters(jobs,10.,10.)
    q=rr[0]['queues'][rr[0]['critical']];ii=np.array(q[:-1]);jj=np.array(q[1:])
    d=jobs.a[jj]-jobs.b[ii];tb=rr[0]['fa'][jj];th=hermite_duration(d,vel[ii],vel[jj],cap,acc)
    if edge is None:
        # Largest absolute saving on this already specified critical chain.
        k=int(np.argmax(tb-th));i,j=int(ii[k]),int(jj[k])
    else:i,j=edge
    origin=jobs.b[i];delta=jobs.a[j]-origin
    tt,xx,vv,aa=phase_samples(jobs,i,j);brake=float(tt[-1]);cv=float(np.linalg.norm(delta)/cap)
    cubic=float(hermite_duration(delta[None],vel[i:i+1],vel[j:j+1],cap,acc)[0]);tau=np.linspace(0,1,501)
    xh,vh,ah=endpoint_sample(delta,vel[i],vel[j],cubic,tau)
    models=[('Constant-speed',np.linspace([0.,0.],delta,100),np.linspace(0,cv,100),np.full(100,cap),None,BLUE),
            ('Brake / rest / ramp',xx-origin,tt,np.linalg.norm(vv,axis=1),np.linalg.norm(aa,axis=1),OFF),
            ('Feasible cubic',xh,tau*cubic,np.linalg.norm(vh,axis=1),np.linalg.norm(ah,axis=1),TEAL)]
    fig,axes=plt.subplots(3,3,figsize=(7.25,5.6),gridspec_kw={'height_ratios':[1.3,1,1]})
    fig.subplots_adjust(left=.085,right=.98,bottom=.10,top=.92,wspace=.35,hspace=.62)
    near_i=-vel[i]/s[i]*min(.8,float(np.linalg.norm(jobs.b[i]-jobs.a[i])))
    near_j=delta+vel[j]/s[j]*min(.8,float(np.linalg.norm(jobs.b[j]-jobs.a[j])))
    allxy=np.vstack([near_i,near_j,xx-origin,xh]);lo=allxy.min(0)-.18;hi=allxy.max(0)+.18
    for col,(label,path,times,speed,accs,color) in enumerate(models):
        ax=axes[0,col];ax.plot(*np.vstack([near_i,[0.,0.]]).T,color=SCAN,lw=3)
        ax.plot(*np.vstack([delta,near_j]).T,color=SCAN,lw=3);ax.plot(*path.T,color=color,lw=1.6)
        ax.scatter([0,delta[0]],[0,delta[1]],s=18,c=INK,zorder=5)
        arrow=min(hi-lo)*.20
        for x,v in [(np.zeros(2),vel[i]),(delta,vel[j])]:
            ax.annotate('',xy=x+v/np.linalg.norm(v)*arrow,xytext=x,arrowprops={'arrowstyle':'->','color':INK,'lw':.8})
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='Relative x (stored units)')
        ax.set_title(chr(97+col)+'  '+label,loc='left',fontsize=8.4)
        if col==0:ax.set_ylabel('Relative y')
        ax=axes[1,col];ax.plot(times,speed/s[i],color=color,lw=1.5);ax.axhline(cap/s[i],lw=.65,ls=':',color=MUTED)
        if col==0:
            ax.text(.5,.48,'Endpoint velocity\nnot enforced',ha='center',transform=ax.transAxes,fontsize=7.3,color=MUTED)
            for t in [0,cv]:ax.plot([t,t],[1,cap/s[i]],'--',color=color,lw=.8)
        ax.set_title(f'Duration = {times[-1]:.6f}',fontsize=8,weight='normal')
        ax.set_ylim(-.2,11);ax.set_xlim(-.02*brake,max(brake,cubic)*1.07);ax.grid(axis='y')
        if col==0:ax.set_ylabel('Speed / exposure speed')
        ax=axes[2,col]
        if accs is not None:ax.plot(times,accs/acc,color=color,lw=1.5)
        else:ax.text(.5,.5,'Acceleration\nunconstrained',ha='center',transform=ax.transAxes,fontsize=8,color=MUTED)
        ax.axhline(1,color=MUTED,ls=':',lw=.65);ax.set(xlim=(-.02*brake,max(brake,cubic)*1.07),ylim=(-.05,1.15),xlabel='Time (stored units)');ax.grid(axis='y')
        if col==0:ax.set_ylabel('Acceleration / limit')
        for ax in axes[:,col]:ax.tick_params(labelsize=7)
    fig.text(.5,.975,f'Actual retained transition {i} → {j}  |  cubic duration change: {100*(cubic/brake-1):+.1f}%',ha='center',fontsize=9,weight='bold')
    fig.text(.5,.025,'Common time scale and endpoint states. These are prescribed motions, not scanner measurements.',ha='center',fontsize=7,color=MUTED)
    save(fig,name)
    (FIG/(name+'.json')).write_text(json.dumps(dict(case='unequal_islands_75',plan='region_y_file',i=i,j=j,brake=brake,cubic=cubic,cv=cv,cap=cap,acceleration=acc,selection='largest absolute cubic saving in fixed critical chain' if edge is None else 'previous longest-duration edge; no-advantage control'),indent=2))
    if edge is None:
        k=int(np.argmax(tb));motion((int(ii[k]),int(jj[k])),'figS02_no_cubic_advantage')

def mechanism_figure():
    jobs,polygons,rr=witness();fig=plt.figure(figsize=(7.25,8.3))
    for k,r in enumerate(rr):
        left=.07+.50*k;ax=fig.add_axes([left,.635,.395,.30])
        for poly in polygons:ax.add_patch(Polygon(poly,facecolor='#edf0f2',edgecolor='#b5bfc5',lw=.5))
        seg=np.stack([jobs.a,jobs.b],axis=1);ax.add_collection(LineCollection(seg,colors=[LASERS[x] for x in r['assignment']],lw=.7))
        q=r['queues'][r['critical']];j=max(q[1:],key=lambda u:r['fa'][u]);i=q[q.index(j)-1]
        for u,v in zip(q[:-1],q[1:]):
            if r['fa'][v]<=1e-10:continue
            _,xy,_,_=phase_samples(jobs,u,v);ax.plot(*xy.T,color=INK,alpha=.30,lw=.55)
        _,xy,_,_=phase_samples(jobs,i,j);ax.plot(*xy.T,color='#a72b4a',lw=1.5)
        for ell,z in enumerate(r['queues']):ax.scatter(*jobs.a[z[0]],s=15,color=LASERS[ell],edgecolor='white',lw=.4,zorder=8)
        ax.set(xlim=(-38,39),ylim=(-30,28),aspect='equal');ax.set_xlabel('x (stored units)',fontsize=7)
        if k==0:ax.set_ylabel('y (stored units)',fontsize=7)
        ax.tick_params(labelsize=7);ax.set_title(f'{chr(97+k)}  Plan {r["role"]}: critical laser L{r["critical"]+1}',loc='left',fontsize=8.5)
        lo=xy.min(0)-.4;hi=xy.max(0)+.4;ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,ec='#a72b4a',lw=.7,ls='--'))
        zoom=fig.add_axes([left,.465,.395,.12]);zoom.plot(*xy.T,color='#a72b4a',lw=1.5)
        zoom.plot(*np.stack([jobs.a[i],jobs.b[i]]).T,color=LASERS[r['critical']],lw=2)
        zoom.plot(*np.stack([jobs.a[j],jobs.b[j]]).T,color=LASERS[r['critical']],lw=2)
        zoom.scatter([jobs.b[i,0],jobs.a[j,0]],[jobs.b[i,1],jobs.a[j,1]],s=16,c=INK)
        zoom.set(xlim=(lo[0]-.3,hi[0]+.3),ylim=(lo[1]-.3,hi[1]+.3),aspect='equal');zoom.tick_params(labelsize=7)
        zoom.set_title(f'{chr(99+k)}  Enlarged transition {i} → {j}',fontsize=8,loc='left')
        ax=fig.add_axes([left,.265,.395,.12]);st=free_starts(jobs,r['queues'],r['fa'])
        for ell,z in enumerate(r['queues']):
            y=3-ell;ax.broken_barh([(st[u],jobs.p[u]) for u in z],(y-.3,.6),facecolor=SCAN)
            ax.broken_barh([(st[u]-r['fa'][u],r['fa'][u]) for u in z if r['fa'][u]>0],(y-.3,.6),facecolor=OFF)
            if ell==r['critical']:ax.add_patch(Rectangle((0,y-.35),max(st[z]+jobs.p[z]),.70,fill=False,ec=INK,lw=.65))
        ax.set(xlim=(0,.85),ylim=(-.5,3.6),yticks=range(4),yticklabels=['L4','L3','L2','L1']);ax.tick_params(labelsize=7)
        ax.set_title(f'{chr(101+k)}  Full schedule (critical row outlined)',loc='left',fontsize=8);ax.set_xlabel('Time (stored units)',fontsize=7)
    ax=fig.add_axes([.22,.055,.66,.125])
    for y,key,r in [(3,'cv',rr[0]),(2.2,'cv',rr[1]),(.8,'fa',rr[0]),(0,'fa',rr[1])]:
        ell=int(np.argmax(r['work']+r[key+'_sum']));w=r['work'][ell];off=r[key+'_sum'][ell]
        ax.barh(y,w,height=.55,color=SCAN);ax.barh(y,off,left=w,height=.55,color=OFF)
        ax.text(w+off+.01,y,f'{w+off:.3f}',va='center',fontsize=8,weight='bold')
    ax.set(yticks=[3,2.2,.8,0],yticklabels=['CV: S','CV: B','Finite a: S','Finite a: B'],xlim=(0,.92),xlabel='Critical-chain cost (stored units)');ax.tick_params(labelsize=7)
    ax.set_title('g  Exposure load versus laser-off motion',loc='left',fontsize=8.5)
    fig.legend([Patch(fc=SCAN),Patch(fc=OFF)]+[Line2D([0],[0],color=c,lw=2) for c in LASERS],['Exposure','Laser off','L1','L2','L3','L4'],ncol=6,loc='upper center',bbox_to_anchor=(.5,.998),fontsize=7)
    save(fig,'fig03_reversal_mechanism')

def sensitivity_map():
    d=load('primary_results.json')['comparisons'];rows=[r for r in d if r['portfolio']=='original_13' and r['model']=='brake_rest_ramp']
    names=list(dict.fromkeys(r['case'] for r in rows));fig=plt.figure(figsize=(7.25,8.0))
    positions=[(.205,.27,0,2),(.495,.27,0,4),(.79,.075,8,2),(.89,.075,8,4)]
    norm=Normalize(0,max(r['selection_loss_pct'] for r in rows));cmap=plt.get_cmap('YlOrRd')
    for n,(x,w,radius,m) in enumerate(positions):
        ax=fig.add_axes([x,.16,w,.71]);settings=[(a,k) for a in [10,100,1000] for k in ([5,10,20] if radius==0 else [10])]
        values=np.full((26,len(settings)),np.nan)
        for i,name in enumerate(names):
            for j,(a,k) in enumerate(settings):
                found=[r for r in rows if r['case']==name and r['lasers']==m and r['radius']==radius and r['alpha']==a and r['kappa']==k]
                if found:values[i,j]=found[0]['selection_loss_pct']
        for i in range(26):
            for j in range(len(settings)):
                val=values[i,j];zero=np.isfinite(val) and val<=1e-6
                color='#eef1f3' if zero else ('white' if np.isnan(val) else cmap(norm(val)))
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fc=color,ec='white',lw=.45,hatch='///' if np.isnan(val) else None))
                if zero:ax.text(j,i,'·',ha='center',va='center',color='#75858e',fontsize=8)
                elif np.isfinite(val) and val>=5:ax.text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=6.1,color='white' if norm(val)>.55 else INK)
        for pos in [2,3,6,9,12,15,18,21,24]:
            # Separators are computed from family transitions below instead.
            pass
        family=lambda name:name.rsplit('_',1)[0] if name not in ['Layer 10','Nozzle'] else name
        for i in range(1,26):
            if family(names[i])!=family(names[i-1]):ax.axhline(i-.5,color='#9baab3',lw=.8)
        ax.set(xlim=(-.5,len(settings)-.5),ylim=(25.5,-.5),xticks=range(len(settings)))
        if n==0:ax.set_yticks(range(26),[z.replace('_',' ') for z in names]);ax.tick_params(axis='y',labelsize=7)
        else:ax.set_yticks([])
        ax.set_xticklabels([str(k) if radius==0 else str(a) for a,k in settings],fontsize=7,rotation=0 if radius==0 else 90)
        ax.set_xlabel('κ' if radius==0 else 'α',fontsize=8);ax.set_title(f'{m} lasers',fontsize=8)
        if radius==0:
            for col,a in [(1,10),(4,100),(7,1000)]:ax.text(col,-1.6,f'α={a}',ha='center',fontsize=7,clip_on=False)
            for col in [2.5,5.5]:ax.axvline(col,color='white',lw=2)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.48,.955,'a  No separation (r = 0)',ha='center',weight='bold',fontsize=9)
    fig.text(.88,.955,'b  Separation\nr = 8; κ = 10',ha='center',weight='bold',fontsize=8.2)
    cbax=fig.add_axes([.26,.065,.42,.018]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',label='Selection loss (%)')
    bar.solids.set_edgecolor('face')  # Avoid hairline seams between vector colorbar cells.
    fig.legend([Patch(fc='#eef1f3'),Patch(fc='white',hatch='///',ec='#adb7bd')],['Zero loss (·)','Not evaluated'],loc='lower right',bbox_to_anchor=(.98,.055),fontsize=7)
    fig.text(.5,.012,'All 624 settings retained. Values ≥5% annotated; full precision in CSV. At r=8, κ=5 and 20 were not evaluated.',ha='center',fontsize=6.7,color=MUTED)
    save(fig,'fig05_discrete_fidelity_map')

def robustness():
    d=load('summary.json');fig,axes=plt.subplots(2,2,figsize=(7.25,5.5));fig.subplots_adjust(left=.095,right=.97,bottom=.14,top=.90,wspace=.36,hspace=.54)
    for col,key in enumerate(['motion','portfolio']):
        data=d['pairs'][key];mat=np.array(data['matrix']);ax=axes[0,col]
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fc=plt.get_cmap('Blues')(mat[i,j]/468),ec='white',lw=.5))
        ax.set(xlim=(-.5,1.5),ylim=(1.5,-.5))
        for i in range(2):
            for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=15,weight='bold',color='white' if mat[i,j]>230 else INK)
        ax.set_xticks([0,1],['Zero','Positive']);ax.set_yticks([0,1],['Zero','Positive']);ax.set_xlabel('After →');ax.set_ylabel('Before')
        ax.set_title(('a  Motion: brake → optional cubic' if col==0 else 'b  Portfolio: 13 → 19'),loc='left',fontsize=8.5)
        rows=sorted([r for r in data['rows'] if abs(r['delta'])>1e-6],key=lambda r:r['delta']);ax=axes[1,col]
        colors={10:OFF,100:BLUE,1000:TEAL}
        ax.axhline(0,color=MUTED,lw=.8)
        for k,r in enumerate(rows):ax.vlines(k,0,r['delta'],color=colors[r['alpha']],lw=.75);ax.scatter(k,r['delta'],s=12,c=colors[r['alpha']])
        ax.set_title(f'{chr(99+col)}  {len(rows)} changed; {data["unchanged"]} unchanged',loc='left',fontsize=8.5)
        ax.set_xlabel('Changed settings, ordered by difference');ax.set_ylabel('Loss change (percentage points)');ax.tick_params(labelsize=7);ax.grid(axis='y')
    fig.legend([Line2D([0],[0],marker='o',color=c,lw=0) for c in [OFF,BLUE,TEAL]],['α=10','α=100','α=1000'],ncol=3,loc='lower center',bbox_to_anchor=(.5,.02),fontsize=8)
    save(fig,'fig06_robustness')

def exact_figure():
    d=load('exact_results.json')['rows'];fig,axs=plt.subplots(1,3,figsize=(7.25,3.4));fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.84,wspace=.32)
    methods=[('gap13_pct','13 plans','#9caeb9'),('gap19_pct','19 plans',BLUE),('gap_search_pct','Local search',TEAL)]
    for col,model in enumerate(['constant_speed','brake_rest_ramp','optional_cubic']):
        ax=axs[col]
        for k,size in enumerate([8,12,16]):
            rows=[r for r in d if r['size']==size and r['model']==model]
            for h,(metric,label,color) in enumerate(methods):
                vals=np.array([r[metric] for r in rows]);pos=k+(h-1)*.24
                ax.plot([pos,pos],[0,max(vals)],color=color,lw=.8,alpha=.6)
                ax.scatter(pos,np.median(vals),s=35,marker=['s','o','D'][h],color=color,label=label if k==0 else None)
                ax.scatter(pos,max(vals),s=20,marker='_',color=color)
        ax.set_xticks(range(3),['8','12','16']);ax.set_xlabel('Vectors in spatial window');ax.grid(axis='y');ax.tick_params(labelsize=7)
        ax.set_title(chr(97+col)+'  '+['Constant speed','Brake / rest / ramp','Optional cubic'][col],loc='left',fontsize=8.5)
        if col==0:ax.set_ylabel('Gap to exact scheduling optimum (%)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,fontsize=7,loc='upper center',bbox_to_anchor=(.52,.995))
    fig.text(.5,.06,'Markers: median; top caps: maximum. 20 spatial windows per size; two accelerations for finite-motion models.',ha='center',fontsize=6.7,color=MUTED)
    save(fig,'fig07_exact_comparison')

def cross_geometry():
    d=load('mechanism_results.json')['rows'];r=[z for z in d if z['mode']=='alpha10'];fig,ax=plt.subplots(2,2,figsize=(7.25,5.9));fig.subplots_adjust(left=.10,right=.97,bottom=.19,top=.91,wspace=.32,hspace=.56)
    for k,(a,key,title,unit) in enumerate(zip(ax.ravel(),['jumps','direct_distance','endpoint_dv_sum','off_time'],['Nonzero jump count','Direct jump distance','Sum of endpoint velocity changes','Laser-off time'],['',' (stored distance)',' (stored distance/time)',' (stored time)'])):
        for z in r:
            s=z['selected_diagnostics']['critical_chain'];b=z['alternative_diagnostics']['critical_chain'];pos=z['selection_loss_pct']>1e-6
            a.scatter(s[key]-b[key],z['selection_loss_pct'],c=OFF if pos else '#7b919f',s=24,marker='o' if z['lasers']==2 else '^',alpha=.75)
        a.axvline(0,color=MUTED,lw=.6,ls=':');a.set_xlabel('Selected − alternative'+unit,fontsize=7.2);a.set_ylabel('Selection loss (%)');a.grid(axis='y');a.set_title(chr(97+k)+'  '+title,loc='left',fontsize=8.1)
    fig.legend([Line2D([0],[0],marker=m,color=INK,lw=0) for m in ['o','^']]+[Patch(fc=OFF),Patch(fc='#7b919f')],['2 lasers','4 lasers','Positive loss','Zero loss'],ncol=4,loc='lower center',bbox_to_anchor=(.53,.065),fontsize=7)
    fig.text(.5,.025,'All 52 configurations, κ=10, α=10, original 13. Each plan uses its own critical laser; counts are not independent experiments.',ha='center',fontsize=6.6,color=MUTED)
    save(fig,'fig08_cross_geometry')

def supplement():
    olddata=json.loads((ROOT/'revision_validation_results.json').read_text());d=load('summary.json')
    fig,ax=plt.subplots(1,2,figsize=(7.25,2.9));fig.subplots_adjust(left=.10,right=.97,bottom=.24,top=.84,wspace=.4)
    for role,color in [('S',OFF),('B',BLUE)]:
        r=[z for z in olddata['segmentation'] if z['plan']==role];x=[z['split_each_vector_into'] for z in r]
        ax[0].plot(x,[z['fixed_acceleration_cost'] for z in r],'-o',color=color,label=role+': fixed a')
        ax[0].plot(x,[z['fixed_alpha_cost'] for z in r],'--s',color=color,label=role+': fixed α')
    ax[0].set(xlabel='Pieces per original vector',ylabel='Fixed-chain cost (stored units)',xticks=[1,2,4]);ax[0].legend(fontsize=6.4);ax[0].set_title('a  Subdivision of unchanged queues',loc='left',fontsize=8.4)
    row=next(z for z in d['ties'] if z['model']=='optional_cubic' and z['portfolio']=='expanded_19')
    vals=[row[z]['positive'] for z in ['optimistic','lexicographic','pessimistic']]
    ax[1].bar(range(3),vals,color=['#b6c3cb',BLUE,'#b6c3cb']);ax[1].set(xticks=range(3),xticklabels=['Reference\nbest','Fixed\nname','Reference\nworst'],ylabel='Positive losses / 468',ylim=(0,max(vals)*1.2))
    for k,v in enumerate(vals):ax[1].text(k,v+2,str(v),ha='center',fontsize=9)
    ax[1].set_title('b  Tie rule and outcome bounds',loc='left',fontsize=8.2)
    save(fig,'figS01_representation_checks')

def continuous_envelope():
    d=next(r for r in load('continuous_results.json')['rows'] if r['case']=='Layer 10' and r['lasers']==4)
    a,b=np.array(d['A']),np.array(d['B']);names=d['plan_names'];q=np.linspace(.05,.2,2003)
    values=(a[:,:,None]+b[:,:,None]*q).max(1);best=values.min(0);selected=names.index(d['selected_plan'])
    fig,ax=plt.subplots(1,3,figsize=(7.25,3.15));fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.86,wspace=.48)
    for row in values:ax[0].plot(q,row,color='#adbac3',lw=.7)
    ax[0].plot(q,best,color=INK,lw=1.5);ax[0].set_ylabel('Calculated makespan (stored units)')
    ax[0].set_title('a  All 13 candidates',loc='left',fontsize=8.5)
    for label,color in [('append_EFT_5',BLUE),('append_EFT_10',OFF),('append_EFT_20',TEAL)]:
        ax[1].plot(q,values[names.index(label)],color=color,lw=1.3,label=label.replace('append_',''))
    ax[1].plot(q,best,color=INK,ls='--',lw=.8);ax[1].legend(loc='upper left',fontsize=6.2);ax[1].set_title('b  Competing cost envelopes',loc='left',fontsize=8.2)
    for switch in d['switches']:
        ax[1].axvline(switch,color=MUTED,lw=.6,ls=':');ax[1].text(switch+.004,ax[1].get_ylim()[0]+.01,f'q={switch:.5f}',fontsize=6.5,rotation=90,va='bottom')
    qp=np.array(d['q']);rp=np.array(d['regret']);curve=100*(values[selected]/best-1)
    ax[2].fill_between(q,0,curve,color=BLUE,alpha=.12);ax[2].plot(q,curve,color=BLUE,lw=1.5);ax[2].plot(qp,rp,'o',ms=2,color=BLUE)
    ax[2].set(ylabel='Relative selection loss (%)',ylim=(-.04,1.6));ax[2].set_title('c  Fixed plan: EFT_5',loc='left',fontsize=8.2)
    k=int(np.argmax(rp));ax[2].annotate(f'{rp[k]:.4f}% at q={qp[k]:.2f}',xy=(qp[k],rp[k]),xytext=(.07,1.45),fontsize=6.7,arrowprops={'arrowstyle':'-','color':MUTED,'lw':.6})
    for x in ax:x.set(xlim=(.05,.2),xticks=[.05,.1,.15,.2],xlabel=r'Inverse speed ratio $q=1/\kappa$');x.tick_params(labelsize=7);x.grid(alpha=.2)
    fig.text(.5,.04,'Released Layer 10, four lasers, r=0. The labeled crossing switches the best candidate; dots evaluate affine-cell boundaries.',ha='center',fontsize=6.5,color=MUTED)
    save(fig,'fig04_continuous_envelope')

def main():
    old.save=save
    scene=old.scene
    def vector_scene(*args,**kw):scene(*args,**kw);args[0].set_rasterization_zorder(None)
    old.scene=vector_scene
    mechanism_figure();motion();sensitivity_map();robustness();exact_figure();cross_geometry();supplement();old.framework();continuous_envelope()
    sources=['primary_results.json','exact_results.json','mechanism_results.json','search_results.json','summary.json','protocol.json','continuous_results.json']
    (FIG/'provenance.json').write_text(json.dumps({'source_sha256':{s:hashlib.sha256((OUT/s).read_bytes()).hexdigest() for s in sources},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'format':'PDF and SVG are vectors; SVG text retained as text; PNG only preview','physical_rendering':'Framework thickness/texture illustrative; no thermal or printed specimen images'},indent=2))

if __name__=='__main__':main()
