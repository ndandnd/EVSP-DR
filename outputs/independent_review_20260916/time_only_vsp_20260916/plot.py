"""Static figure; all data and labels remain editable beside it."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
labels={'x':'Number of instances','legend_equal':'Same fleet as GIRO','legend_one_less':'One fewer bus','rows':['18E1','18E2','Groups separate','Groups mixed']}
if (B/'plot_labels.json').exists():labels=json.loads((B/'plot_labels.json').read_text())
else:(B/'plot_labels.json').write_text(json.dumps(labels,indent=2)+'\n')
rs=list(csv.DictReader((B/'per_case.csv').open()));data=[]
for group in labels['rows']:
 vals=[]
 for r in rs:
  if group in ['18E1','18E2']:g=int(r[group+'_giro_duties'])-int(r[group+'_closure_relaxation_minimum'])
  else:g=int(r['target_buses'])-int(r['segregated_closure_lp_fleet_lower_bound' if group=='Groups separate' else 'mixed_closure_relaxation_minimum'])
  vals.append(g)
 assert set(vals)<={0,1};data.append(dict(group=group,same_as_giro=sum(v==0 for v in vals),one_fewer=sum(v==1 for v in vals)))
with (B/'plot_data.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
fig,ax=plt.subplots(figsize=(8.5,3.8));y=list(range(4));eq=[r['same_as_giro'] for r in data];less=[r['one_fewer'] for r in data]
ax.barh(y,eq,color='#7b8f9f',height=.6,label=labels['legend_equal']);ax.barh(y,less,left=eq,color='#007f73',height=.6,label=labels['legend_one_less'])
for i,(a,b) in enumerate(zip(eq,less)):
 ax.text(a/2,i,str(a),ha='center',va='center',color='white',weight='bold')
 if b:ax.text(a+b/2,i,str(b),ha='center',va='center',color='white',weight='bold')
ax.set_yticks(y,labels['rows']);ax.invert_yaxis();ax.set_xlabel(labels['x']);ax.set_xlim(0,106);ax.set_xticks([0,20,40,60,80,100]);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2,frameon=False);fig.tight_layout()
fig.savefig(B/'time_only_fleet_counts.png',dpi=200,bbox_inches='tight');fig.savefig(B/'time_only_fleet_counts.pdf',bbox_inches='tight')
