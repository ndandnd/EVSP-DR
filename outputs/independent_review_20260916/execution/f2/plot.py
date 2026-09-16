"""Scientific plot from audited per_case.csv; no embedded narrative claims."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'per_case.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),gridspec_kw={'width_ratios':[1,1.4]})
colors=['#28649b','#ba5a37']
for i,g in enumerate(['LP_weight_k','LP_weight_k_minus_1']):
 r=[x for x in rows if x['cohort']=='original_1h' and x['lp_weight_group']==g]
 y=np.array([float(x['mean_route_jaccard']) for x in r]);j=np.linspace(-.16,.16,len(y));j=np.random.default_rng(20260916+i).permutation(j)
 axes[0].scatter(i+j,y,s=23,color=colors[i],alpha=.65,edgecolors='none')
 axes[0].plot([i-.22,i+.22],[y.mean()]*2,color='black',lw=2)
axes[0].set_xticks([0,1],['Route weight k\n93 cases','Route weight k−1\n9 cases'])
axes[0].set_ylabel('Mean nearest-duty Jaccard per case')
axes[0].set_title('Original one-hour MIP results')
ids=[r['case_id'] for r in rows if r['cohort']=='original_1h' and r['lp_weight_group']=='LP_weight_k_minus_1']
for i,cid in enumerate(ids):
 a=next(r for r in rows if r['case_id']==cid and r['cohort']=='original_1h');b=next(r for r in rows if r['case_id']==cid and r['cohort']=='longer_3h_fleet')
 va,vb=float(a['mean_route_jaccard']),float(b['mean_route_jaccard'])
 axes[1].plot([i,i],[va,vb],color='#bbbbbb',lw=2)
 axes[1].scatter(i,va,c=colors[1],s=30,label='Original' if i==0 else None)
 axes[1].scatter(i,vb,c=colors[0],s=30,label='Longer MIP' if i==0 else None)
axes[1].set_xticks(range(9),[c.replace('w','C').replace('_k',' / ') for c in ids],rotation=55,ha='right')
axes[1].set_title('Same nine pools: longer MIP sensitivity')
axes[1].legend(frameon=False,loc='lower right')
for ax in axes:ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(P/'jaccard_comparison.png',dpi=180);fig.savefig(P/'jaccard_comparison.svg');plt.close(fig)
