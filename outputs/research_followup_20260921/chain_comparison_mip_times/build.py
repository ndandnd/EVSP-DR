"""Actual last-MIP phase timing for the frozen 24-pair chain panel. No solver/network."""
from pathlib import Path
import csv,json,hashlib,statistics,subprocess,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=P.parent/'chain_comparison';SRC=P.parent/'battery_rounding'/'sources';sources={};checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):sources[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def check(x,msg):assert x,msg;checks.append(msg)
def csvout(name,rows):
 with (P/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
prior=load(OLD/'provenance.json');bp=load(P.parent/'battery_rounding/provenance.json');raw=OLD/'chain_comparison.csv';sources[str(raw.relative_to(ROOT))]=sha(raw);check(sha(raw)==prior['output_sha256']['chain_comparison.csv'],'prior comparison CSV matches manifest')
paired=list(csv.DictReader(raw.open()));times=[];combined=[]
for pair in paired:
 z=dict(pair)
 for arm,filearm,label in [('fresh','base','Fresh'),('sequential','warm','Sequential')]:
  path=SRC/(pair['case_id']+'_'+filearm+'_result.json');r=load(path);key='fresh' if arm=='fresh' else 'warm';check(sha(path)==pair[key+'_mip_source_sha256'],path.name+' endpoint hash');check(sha(path)==bp['source_hashes'][str(path.relative_to(ROOT))]['sha256'],path.name+' extraction hash')
  t=r['two_stage'];a=r['gurobi_optimize_stage_wall_s'];check(len(a)==2 and t['stage2_executed'],path.name+' both stages executed');check(abs(sum(a)-r['gurobi_optimize_wall_s'])<1e-8,path.name+' sum optimize calls')
  check(r['runtime_s']>=sum(a)-1e-6,path.name+' solver phase includes optimize calls');check(t['stage1_runtime_s']>=a[0],path.name+' fleet elapsed includes optimize call');check(r['buses']==int(pair['fresh_buses' if arm=='fresh' else 'warm_buses']),path.name+' buses match panel')
  check(r['mip_provenance']['arguments']['timelimit']==3600 and t['stage1_time_limit_s']==1800,path.name+' nominal allowances');check(r['physical_pool_audit']['input_hashes']['instance_sha256']==pair['input_sha256'],path.name+' input identity');check(r.get('progress') is None,path.name+' no first-target trace in frozen payload')
  row=dict(case_id=pair['case_id'],chain=int(pair['chain']),target_k=int(pair['target_k']),arm=arm,buses=r['buses'],fleet_bound=t['stage1_bound'],finite_pool_fleet_proven=t['fleet_proven'],fleet_status=t['stage1_status_name'],charging_status=t['stage2_status_name'],final_status=r['status_name'],target_attained=r['buses']==int(pair['target_k']),fleet_optimize_s=a[0],charging_optimize_s=a[1],total_optimize_s=sum(a),fleet_phase_elapsed_s=t['stage1_runtime_s'],solver_phase_elapsed_s=r['runtime_s'],phase_non_optimize_s=r['runtime_s']-sum(a),physical_preparation_s=r['physical_pool_preparation_wall_s'],source_hashing_s=r.get('source_hashing_wall_s'),end_to_end_before_publication_s=r.get('end_to_end_before_publication_s'),nominal_total_limit_s=3600,nominal_fleet_limit_s=1800,actual_charging_limit_s=t['stage2_available_time_s'],charging_fleet_cap=t['stage2_fleet_cap'],charging_fleet_cap_proven=t['stage2_fleet_cap_proven'],charging_variable_bound=t['stage2_variable_bound'],charging_variable_obj=t['stage2_variable_obj'],charging_gap_absolute=t['stage2_variable_absolute_gap'],first_target_time_s=None,first_target_time_status='not identified: no progress trace in pinned JSON; fleet-stage end is not first-hit time',timing_scope='last MIP only; earlier sequential MIPs excluded',source=str(path.relative_to(ROOT)),remote_source=pair[key+'_mip_source_path'],source_sha256=sha(path),execution_commit=r['mip_provenance']['git_commit'])
  times.append(row)
  for field in ['fleet_optimize_s','charging_optimize_s','total_optimize_s','fleet_phase_elapsed_s','solver_phase_elapsed_s','fleet_status','charging_status','finite_pool_fleet_proven']:z[arm+'_'+field]=row[field]
 combined.append(z)
csvout('mip_stage_times.csv',times);csvout('chain_comparison_with_mip.csv',combined)
# Source-code audit identifies the two direct optimize timers and broader runtime counters.
commit=times[0]['execution_commit'];code=subprocess.check_output(['git','show',commit+':src/run_exact_pool_mip.py'],cwd=ROOT,text=True);(P/'timing_source_excerpt.txt').write_text('\n'.join(f'{i}: {ln}' for i,ln in enumerate(code.splitlines(),1) if 2530<=i<=2560 or 2605<=i<=2620 or 2698<=i<=2720 or 3115<=i<=3127 or 3193<=i<=3222)+'\n')
summary={'pairs':24,'endpoints':48,'per_arm':{},'source_mip_commit':commit,'first_target_times_identified':0,'timing_source':'gurobi_optimize_stage_wall_s (direct wall timer around each optimize_with_start_audit call); sum verified against gurobi_optimize_wall_s','solver_phase_runtime_semantics':'runtime_s includes inter-stage validation and postsolve work; physical preparation/source hashing outside the phase are separate','no_new_solver_runs':True}
for arm in ['fresh','sequential']:
 rs=[r for r in times if r['arm']==arm]
 summary['per_arm'][arm]={'fleet_optimal':sum(r['fleet_status']=='OPTIMAL' for r in rs),'fleet_time_limit':sum(r['fleet_status']=='TIME_LIMIT' for r in rs),'charging_optimal':sum(r['charging_status']=='OPTIMAL' for r in rs),'charging_time_limit':sum(r['charging_status']=='TIME_LIMIT' for r in rs),'target_hits':sum(r['target_attained'] for r in rs),'total_optimize_s_range':[min(r['total_optimize_s'] for r in rs),max(r['total_optimize_s'] for r in rs)],'total_optimize_median_min':statistics.median(r['total_optimize_s'] for r in rs)/60,'fleet_optimize_median_min':statistics.median(r['fleet_optimize_s'] for r in rs)/60,'charging_optimize_median_min':statistics.median(r['charging_optimize_s'] for r in rs)/60,'all_24_total_optimizer_hours':sum(r['total_optimize_s'] for r in rs)/3600,'actual_under_10min':sum(r['total_optimize_s']<600 for r in rs)}
(P/'timing_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'lp_similarity.json').write_bytes((OLD/'lp_similarity.json').read_bytes())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'axes.axisbelow':True,'grid.color':'#e6ebee','grid.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#2477a8';ORANGE='#c57135';ARMS=[('fresh',BLUE,'o','-'),('sequential',ORANGE,'s','--')]
handles=[Line2D([],[],color=BLUE,marker='o',lw=1.7,label='Fresh: direct CG / final pool MIP'),Line2D([],[],color=ORANGE,marker='s',mfc='none',lw=1.7,ls='--',label='Sequential: cumulative CG / final pool MIP')]
TITLES=['CG computation','Integer outcome','Weighted LP, magnified','MIP: fleet search','MIP: charging-cost search','MIP: total optimizer time']
def lines(ax,k,values,arm):
 _,color,marker,ls=next(x for x in ARMS if x[0]==arm);ax.plot(k,values,color=color,marker=marker,ls=ls,lw=1.8,ms=4.8 if arm=='fresh' else 6.5,mfc=color if arm=='fresh' else 'none')
def panels(axs,rs,ylabels=True,titles=True):
 ks=[int(r['target_k']) for r in rs]
 for arm,col,marker,ls in ARMS:
  field='fresh_cg_minutes' if arm=='fresh' else 'sequential_cg_minutes_including_ancestors';lines(axs[0],ks,[float(r[field]) for r in rs],arm)
  field='fresh_integer_excess' if arm=='fresh' else 'sequential_integer_excess';lines(axs[1],ks,[float(r[field]) for r in rs],arm)
  lines(axs[2],ks,[float(r[arm+'_lp_above_k_bus_cost']) for r in rs],arm)
  for n,f in enumerate(['fleet_optimize_s','charging_optimize_s','total_optimize_s'],3):lines(axs[n],ks,[float(r[arm+'_'+f])/60 for r in rs],arm)
 for r in rs:
  k=int(r['target_k']);lo=float(r['fresh_pool_bound_excess']);hi=float(r['fresh_integer_excess']);axs[1].plot([k,k],[lo,hi],color=BLUE,lw=1.1,alpha=.5);axs[1].plot(k,lo,'_',color=BLUE,ms=8)
 axs[0].set_yscale('log');axs[0].set_ylim(3,1800);axs[0].set_yticks([5,30,100,300,1000]);axs[0].set_yticklabels(['5','30','100','300','1000'])
 axs[1].set_ylim(-.3,5.4);axs[1].set_yticks(range(6));axs[2].set_ylim(0,800)
 for ax in axs[3:]:
  ax.set_yscale('log');ax.set_ylim(.0045,85);ax.set_yticks([.01,.1,1,10,60]);ax.set_yticklabels(['0.01','0.1','1','10','60'])
 axs[3].axhline(30,color='#a1abb2',lw=.65,ls=':');axs[5].axhline(60,color='#a1abb2',lw=.65,ls=':')
 labels=['CG minutes (log)','Buses above GIRO target','LP − 100,000 × k\n(synthetic cost units)','Fleet optimizer minutes (log)','Charging optimizer minutes (log)','Total optimizer minutes (log)']
 for n,ax in enumerate(axs):
  ax.set_xticks(ks);ax.set_xlim(4.5,15.5);ax.set_xlabel('GIRO target k (buses)')
  if ylabels:ax.set_ylabel(labels[n])
  if titles:ax.set_title(TITLES[n],pad=8)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(P/(name+'.'+ext),dpi=210,facecolor='white')
 plt.close(fig)
for chain in range(1,7):
 rs=sorted([r for r in combined if int(r['chain'])==chain],key=lambda r:int(r['target_k']));fig,axs=plt.subplots(2,3,figsize=(13,6.2));panels(axs.ravel(),rs);fig.subplots_adjust(left=.075,right=.985,bottom=.105,top=.86,wspace=.39,hspace=.66);fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.98),ncol=2,frameon=False,fontsize=10);save(fig,f'chain{chain}_charts')
fig,axes=plt.subplots(6,6,figsize=(19,16),sharey='row')
for col in range(6):
 rs=sorted([r for r in combined if int(r['chain'])==col+1],key=lambda r:int(r['target_k']));panels(axes[:,col],rs,ylabels=col==0,titles=False);axes[0,col].set_title(f'Chain {col+1}',fontsize=12)
 for ax in axes[:-1,col]:ax.set_xlabel('');ax.tick_params(labelbottom=False)
fig.subplots_adjust(left=.07,right=.99,bottom=.055,top=.95,wspace=.2,hspace=.2);fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.985),ncol=2,frameon=False);save(fig,'all_chains_charts')
# Small native-text stage table per chain, useful independently of the charts.
for chain in range(1,7):
 rs=[r for r in times if r['chain']==chain];fig,ax=plt.subplots(figsize=(11,3.7));ax.axis('off');table=[]
 for r in rs:table.append([r['target_k'],'Fresh' if r['arm']=='fresh' else 'Sequential',f"{r['fleet_optimize_s']:.2f}",f"{r['charging_optimize_s']:.2f}",f"{r['total_optimize_s']:.2f}",f"{r['buses']}/{r['fleet_bound']:.0f}",r['fleet_status'],r['charging_status']])
 tb=ax.table(cellText=table,colLabels=['Target','Pool','Fleet s','Charging s','Total s','Fleet/bound','Fleet stop','Charging stop'],colWidths=[.065,.115,.09,.11,.10,.10,.17,.18],cellLoc='center',loc='center');tb.auto_set_font_size(False);tb.set_fontsize(9.5);tb.scale(1,1.8)
 for (r,c),cell in tb.get_celld().items():
  cell.set_linewidth(.4);cell.set_edgecolor('#cdd7df')
  if r==0:cell.set_facecolor('#e8eff3');cell.set_text_props(weight='bold')
 fig.tight_layout();save(fig,f'chain{chain}_mip_table')
manifest={'sources':sources,'check_count':len(checks),'checks':checks,'summary':summary,'figure_scope':'24 paired observations at k5/8/10/15; lines guide the eye, not intermediate measurements','time_scope':'CG fresh direct versus sequential cumulative ancestors; sequential cumulative excludes earlier MIPs. MIP panels show actual final-stage optimizer calls only, not nominal allowance and not first-target time. Graph, queue and physical preparation excluded.','code_semantics_sha256':hashlib.sha256(code.encode()).hexdigest(),'output_sha256':{f.name:sha(f) for f in P.iterdir() if f.is_file() and f.name!='provenance.json'}};(P/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))
