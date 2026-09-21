import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=pathlib.Path(__file__).resolve().parent
rows={x['case']:x for x in json.loads((r/'capacity_summary.json').read_text())}
labels=['k1 / 14 trips','k2 / 23 trips','k3 / 35 trips'];cases=['k1_duty13406','e1_short_k2','e1_short_k3'];colors=['#96A4B5','#137D92']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(1,2,figsize=(12.5,4.9),gridspec_kw={'width_ratios':[1,1.05]})
x=np.arange(3)
for j,mode in enumerate(['off','on']):
 vals=[rows[c+'__'+mode]['pool_fleet'] for c in cases]
 b=ax[0].bar(x+(.18 if j else -.18),vals,.34,color=colors[j],label=['Original','Capacity shortcut'][j])
 ax[0].bar_label(b,padding=3)
 vals=[rows[c+'__'+mode]['completed_iterations'] for c in cases]
 b=ax[1].bar(x+(.18 if j else -.18),vals,.34,color=colors[j])
 ax[1].bar_label(b,padding=3)
for a in ax:a.set_xticks(x,labels);a.grid(axis='y',alpha=.14);a.set_axisbelow(True)
ax[0].set_ylim(0,24);ax[0].set_ylabel('Buses in saved-pool optimum');ax[0].set_title('Better integer pools within the same budget',loc='left',fontweight='bold');ax[0].legend(frameon=False,loc='upper left')
ax[1].set_ylim(0,150);ax[1].set_ylabel('Completed pricing iterations');ax[1].set_title('Pricing throughput explains the improvement',loc='left',fontweight='bold')
f.suptitle('Capacity pricing shortcut: verified six-run comparison',x=.055,ha='left',fontsize=18,fontweight='bold')
f.text(.055,.04,'4h CG budget. k1 shortcut certified in 26.9 min; all other CGs timed out uncertified.\nAll fleet bounds are finite-pool proofs. k3 shortcut still has 6 duplicated trip assignments.',fontsize=10,color='#45536A')
f.subplots_adjust(left=.065,right=.985,top=.80,bottom=.23,wspace=.28);f.savefig(r/'capacity_on_off.png',dpi=200,facecolor='white');f.savefig(r/'capacity_on_off.pdf',facecolor='white')
