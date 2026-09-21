"""Chain-oriented display of the frozen paired panel; no optimization calls."""
from pathlib import Path
import csv, hashlib, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {}
CHECKS = []
def read(rel):
    p = ROOT / rel
    data = p.read_bytes()
    SOURCES[rel] = hashlib.sha256(data).hexdigest()
    return data
def load(rel): return json.loads(read(rel))
def check(ok, name):
    assert ok, name
    CHECKS.append(name)
def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
def save(fig, name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

rel = 'outputs/research_management_20260921/paper_results/figure1_paired_budget.csv'
raw = read(rel)
prior = load('outputs/research_management_20260921/paper_results/provenance.json')
check(hashlib.sha256(raw).hexdigest() == prior['output_hashes']['figure1_paired_budget.csv'], 'published panel hash')
rows = list(csv.DictReader(raw.decode().splitlines()))
snap = load('outputs/post_meeting_20260910/monitor/20260914T004739Z.json')['campaigns']['cumulative_budget_20260913']
anc = load('outputs/cumulative_budget_20260913/audit/ancestry.json')['nodes']
fresh = {(r['case_id'],r['budget_arm']):r for r in snap['cg']}
out = []
for r in rows:
    f = fresh[r['case_id'],'base']; s = anc[r['sequential_target_source_path']]
    check(f['sha256'] == r['fresh_cg_source_sha256'] and s['sha256'] == r['sequential_target_source_sha256'], r['case_id']+' endpoint hashes')
    check(f['certified_rc_optimal'] and s['certified_rc_optimal'], r['case_id']+' certificates')
    for key in ['instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','pricing_certificate_scope','rc_eps']:
        check(f['provenance'][key] == s['provenance'][key], r['case_id']+' '+key)
    for key in ['g_kwh','charge_kw','soc_step','block_min','min_soc_frac','master_sense','time_model']:
        check(f[key] == s[key], r['case_id']+' '+key)
    k = int(r['target_k']); zf = f['final']['lp_obj']; zs = s['final']['lp_obj']
    wf = f['final']['route_weight']; ws = s['final']['route_weight']
    check(zf == float(r['fresh_weighted_lp_objective']) and zs == float(r['sequential_weighted_lp_objective']),r['case_id']+' objective payload')
    check(abs(wf-k)<1e-9 and abs(ws-k)<1e-9,r['case_id']+' fractional fleet equals target numerically')
    check(f['final']['artificials']==0 and s['final']['artificials']==0,r['case_id']+' no artificial route weight')
    check(min(f['final']['min_rc'],s['final']['min_rc']) >= -float(f['provenance']['rc_eps']),r['case_id']+' minimum reduced cost within stopping tolerance')
    out.append({**r,'fresh_lp_above_k_bus_cost':zf-100000*k,'sequential_lp_above_k_bus_cost':zs-100000*k,
      'fresh_lp_charging_component':zf-100000*wf,'sequential_lp_charging_component':zs-100000*ws,
      'lp_difference_signed_fresh_minus_sequential':zf-zs,'lp_difference_absolute':abs(zf-zs),
      'lp_relative_difference':abs(zf-zs)/max(abs(zf),abs(zs)), 'fractional_route_weight_difference':abs(wf-ws),
      'fresh_min_rc':f['final']['min_rc'],'sequential_min_rc':s['final']['min_rc'],
      'rc_eps':f['provenance']['rc_eps'],'certificate_scope':f['provenance']['pricing_certificate_scope']})
write_csv('chain_comparison.csv',out)
summary = {'paired_cases':len(out),'measured_targets':[5,8,10,15],
 'maximum_weighted_lp_absolute_difference':max(r['lp_difference_absolute'] for r in out),
 'maximum_weighted_lp_relative_difference':max(r['lp_relative_difference'] for r in out),
 'maximum_weighted_lp_relative_difference_percent':100*max(r['lp_relative_difference'] for r in out),
 'maximum_fractional_route_weight_difference':max(r['fractional_route_weight_difference'] for r in out),
 'maximum_charging_component_absolute_difference':max(abs(r['fresh_lp_charging_component']-r['sequential_lp_charging_component']) for r in out),
 'all_48_endpoints_pricing_certified':True,'pricing_certificate_scope':'conservative_expanded_grid_model_only',
 'rc_eps':.0001,'fresh_min_rc_range':[min(r['fresh_min_rc'] for r in out),max(r['fresh_min_rc'] for r in out)],
 'sequential_min_rc_range':[min(r['sequential_min_rc'] for r in out),max(r['sequential_min_rc'] for r in out)],
 'worst_case':max(out,key=lambda r:r['lp_difference_absolute'])['case_id'],
 'new_solver_calls':0,'interpretation':'numerically indistinguishable weighted LP values, not equality of route supports or a continuous-charging certificate'}
(OUT/'lp_similarity.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'axes.axisbelow':True,
 'grid.color':'#e6ebee','grid.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#2477a8';ORANGE='#c57135';GRAY='#8e989f'
handles=[Line2D([],[],color=BLUE,marker='o',lw=1.8,label='Fresh: direct solve'),
         Line2D([],[],color=ORANGE,marker='s',markerfacecolor='none',lw=1.7,ls='--',label='Sequential: inherited columns')]
def panels(axs, rs, labels=True):
    ks = [int(r['target_k']) for r in rs]
    for field,color,marker,ls in [('fresh_cg_minutes',BLUE,'o','-'),('sequential_cg_minutes_including_ancestors',ORANGE,'s','--')]:
        axs[0].plot(ks,[float(r[field]) for r in rs],color=color,marker=marker,ls=ls,lw=1.8,ms=5,markerfacecolor='white' if color==ORANGE else color)
    axs[0].set_yscale('log');axs[0].set_ylim(3,1800);axs[0].set_yticks([5,10,30,100,300,1000]);axs[0].set_yticklabels(['5','10','30','100','300','1000'])
    for r in rs:
        k=int(r['target_k']);lo=float(r['fresh_pool_bound_excess']);hi=float(r['fresh_integer_excess'])
        axs[1].plot([k,k],[lo,hi],color=BLUE,lw=1.2,alpha=.5);axs[1].plot(k,lo,'_',color=BLUE,ms=8)
    axs[1].plot(ks,[float(r['fresh_integer_excess']) for r in rs],color=BLUE,marker='o',lw=1.8,ms=5)
    axs[1].plot(ks,[float(r['sequential_integer_excess']) for r in rs],color=ORANGE,marker='s',mfc='none',ms=8,ls='--',lw=1.7)
    axs[1].set_ylim(-.3,5.4);axs[1].set_yticks(range(6))
    axs[2].plot(ks,[r['fresh_lp_above_k_bus_cost'] for r in rs],color=BLUE,marker='o',lw=2,ms=5)
    axs[2].plot(ks,[r['sequential_lp_above_k_bus_cost'] for r in rs],color=ORANGE,marker='s',mfc='none',ls='--',lw=1.5,ms=8)
    axs[2].set_ylim(0,800)
    if labels:
        axs[0].set_ylabel('CG time (minutes, log scale)')
        axs[1].set_ylabel('Integer buses above target')
        axs[2].set_ylabel('LP objective − 100,000 × k\n(synthetic cost units)')
    for ax in axs:ax.set_xticks(ks);ax.set_xlim(4.5,15.5);ax.set_xlabel('GIRO target k (buses)')

for chain in range(1,7):
    rs=sorted([r for r in out if int(r['chain'])==chain],key=lambda r:int(r['target_k']))
    check(len(rs)==4, f'C{chain} four measured targets')
    cumulative=[float(r['sequential_cg_minutes_including_ancestors']) for r in rs]
    check(all(x<=y for x,y in zip(cumulative,cumulative[1:])),f'C{chain} nondecreasing accumulated work')
    fig,axs=plt.subplots(1,3,figsize=(12,3.65),gridspec_kw={'wspace':.42})
    panels(axs,rs)
    for ax,title in zip(axs,['Accumulated sequential CG work','One-hour integer solve','LP comparison, magnified']):ax.set_title(title,fontsize=11,pad=11)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,1.16),ncol=2,frameon=False)
    save(fig,f'chain{chain}_comparison')

fig,axs=plt.subplots(3,6,figsize=(17.2,7.8),sharey='row',gridspec_kw={'hspace':.28,'wspace':.17})
for j in range(6):
    rs=sorted([r for r in out if int(r['chain'])==j+1],key=lambda r:int(r['target_k']))
    panels(axs[:,j],rs,labels=j==0)
    axs[0,j].set_title(f'Chain {j+1}',fontsize=12,pad=12)
    axs[0,j].set_xlabel('');axs[1,j].set_xlabel('')
    axs[0,j].tick_params(labelbottom=False);axs[1,j].tick_params(labelbottom=False)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,1.0),ncol=2,frameon=False)
fig.subplots_adjust(top=.90,bottom=.10)
save(fig,'all_chains_comparison')
manifest={'sources':SOURCES,'checks':CHECKS,'check_count':len(CHECKS),'summary':summary,
 'figure_scope':'24 paired observations only; connecting lines are guides, not intermediate measured endpoints',
 'time_scope':'Fresh direct CG versus sum of sequential ancestor CG; graph construction, MIP and queue time excluded; graph times remain in CSV',
 'baseline_physics':'240kWh/240kW, no reserve/shared capacity/terminal floor, flat tariff, fee5, covering,2.5kWh/5min grid',
 'historical_code_hardware_limit':'not a causal controlled runtime benchmark',
 'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2));print('Checks:',len(CHECKS))
