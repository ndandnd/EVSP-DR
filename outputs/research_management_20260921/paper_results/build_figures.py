"""Rebuild paper-preview figures from pinned local experimental evidence; no solves."""
from pathlib import Path
import csv, hashlib, json, math, re, statistics, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {}
CHECKS = []
def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw
def js(path): return json.loads(read(path))
def rows(path): return list(csv.DictReader(read(path).decode().splitlines()))
def write_csv(name, data):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, list(data[0])); w.writeheader(); w.writerows(data)
def check(condition, label):
    assert condition, label
    CHECKS.append(label)
def same(a, b): return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-7)
def save(fig, name):
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelsize':11,'axes.edgecolor':'#52616b','xtick.color':'#35434b',
 'ytick.color':'#35434b','svg.fonttype':'none','pdf.fonttype':42,'axes.grid':True,
 'grid.color':'#e7ebee','grid.linewidth':0.65,'axes.axisbelow':True})
BLUE='#2477a8'; ORANGE='#c57135'; GREEN='#168578'; GRAY='#8e989f'; PURPLE='#8c569f'
E='outputs/week_20260921/evidence/'
C='outputs/cumulative_budget_20260913/'
P='outputs/controlled_comparison_20260913/'
snap_path='outputs/post_meeting_20260910/monitor/20260914T004739Z.json'
snap=js(snap_path)['campaigns']['cumulative_budget_20260913']
comparison=rows(C+'status_20260914T004739Z/comparison.csv')
budgets={r['id']:r for r in rows(C+'audit/budgets.csv')}
targets=js(C+'audit/targets.json')['targets']
ancestry=js(C+'audit/ancestry.json')['nodes']
cg={(r['case_id'],r['budget_arm']):r for r in snap['cg']}
mip={(r['case_id'],r['budget_arm']):r for r in snap['mip']}
paired=[]
for r in comparison:
    case=r['case_id']; a=cg[case,'base']; f=mip[case,'base']; w=mip[case,'warm']; b=budgets[case]
    t=targets[case]; n=ancestry[t['status_path']]
    check(a['sha256']==r['fresh_cg_source_sha256'] and f['sha256']==r['fresh_mip_source_sha256'] and w['sha256']==r['warm_mip_source_sha256'],case+' result identity')
    check(a['input_sha256']==f['input_sha256']==w['input_sha256']==t['input_sha256'],case+' matched input')
    check(same(a['wall_s']/60,r['fresh_cg_minutes']) and same(a['final']['lp_obj'],r['fresh_weighted_lp_objective']) and f['buses']==int(r['fresh_buses']) and w['buses']==int(r['warm_buses']),case+' payload agreement')
    check(a['certified_rc_optimal'] and n['certified_rc_optimal'],case+' both pricing certificates')
    paired.append({**r,'input_sha256':a['input_sha256'],'sequential_cg_minutes_including_ancestors':float(b['native_cumulative_wall_s'])/60,
      'sequential_cg_plus_ancestor_graph_minutes':(float(b['native_cumulative_wall_s'])+float(b['ancestor_external_graph_build_s']))/60,
      'common_target_graph_minutes':float(b['target_external_graph_build_s'])/60,
      'sequential_target_cg_minutes':n['wall_s']/60,'sequential_target_cg_certified':n['certified_rc_optimal'],
      'sequential_weighted_lp_objective':n['final']['lp_obj'],'sequential_fractional_route_weight':n['final']['route_weight'],
      'sequential_target_source_path':t['status_path'],'sequential_target_source_sha256':t['status_sha256'],
      'fresh_fleet_stage_minutes':f['two_stage']['stage1_runtime_s']/60,'sequential_fleet_stage_minutes':w['two_stage']['stage1_runtime_s']/60,
      'mip_total_allowance_s':3600,'mip_fleet_allowance_s':1800,'fresh_cg_execution_commit':a['execution_commit'],
      'mip_execution_commit':f['execution_commit'],'fresh_integer_excess':f['buses']-int(r['target_k']),
      'fresh_pool_bound_excess':f['fleet_bound']-int(r['target_k']),'sequential_integer_excess':w['buses']-int(r['target_k'])})
write_csv('figure1_paired_budget.csv',paired)
fig,axs=plt.subplots(2,4,figsize=(12.3,6.0),sharex=True,sharey='row',gridspec_kw={'hspace':.16,'wspace':.09})
for j,k in enumerate((5,8,10,15)):
    rs=[r for r in paired if int(r['target_k'])==k]; xs=np.arange(1,7)
    for field,color,marker,ls in [('fresh_cg_minutes',BLUE,'o','-'),('sequential_cg_minutes_including_ancestors',ORANGE,'s','-'),('sequential_cg_plus_ancestor_graph_minutes',GRAY,None,'--')]:
        axs[0,j].plot(xs,[float(r[field]) for r in rs],color=color,marker=marker,lw=1.4,ms=4,ls=ls)
    axs[0,j].set_yscale('log');axs[0,j].set_ylim(3,4200)
    for x,r in zip(xs,rs):
        y=float(r['fresh_integer_excess']);low=float(r['fresh_pool_bound_excess'])
        axs[1,j].plot([x-.09,x-.09],[low,y],color=BLUE,lw=1.4)
        axs[1,j].plot(x-.09,low,'_',color=BLUE,ms=8)
        axs[1,j].plot(x-.09,y,'o',color=BLUE,ms=5)
    axs[1,j].scatter(xs+.09,[0]*6,c=ORANGE,marker='s',s=22,zorder=4)
    axs[1,j].set_xticks(xs,[f'C{i}' for i in xs]);axs[1,j].set_xlabel(f'GIRO target {k} buses');axs[1,j].set_ylim(-.3,5.4);axs[1,j].set_yticks(range(6))
axs[0,0].set_ylabel('CG minutes\nincluding smaller instances')
axs[1,0].set_ylabel('Integer buses above\nGIRO target')
fig.legend([Line2D([],[],color=BLUE,marker='o'),Line2D([],[],color=ORANGE,marker='s'),Line2D([],[],color=GRAY,ls='--')],['Fresh','Sequential: accumulated CG','Sequential: CG + smaller graph builds'],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.015))
fig.subplots_adjust(top=.89,bottom=.12);save(fig,'figure1_paired_budget')

def audited_logs(filename):
    rs=rows(E+filename)
    for r in rs:
        raw=read(r['local_log']); check(hashlib.sha256(raw).hexdigest()==r['log_sha256'],r['case_arm']+' log hash')
        lines=raw.decode().splitlines(); line=lines[int(r['fleet_proof_line'])-1]
        match=re.search(r'Best objective ([\deE+.-]+), best bound ([\deE+.-]+)',line)
        check(bool(match) and same(match[1],r['fleet_buses']) and same(match[2],r['fleet_bound']),r['case_arm']+' fleet objective/bound proof line')
    return rs
witness=audited_logs('k8_witness_summary.csv');pilot=audited_logs('pilot_summary.csv');follow=audited_logs('c1_followup_summary.csv');long=audited_logs('k15_12h_summary.csv')
mechanism=rows(E+'mechanism_summary.csv')
write_csv('figure2_k8_witness.csv',witness);write_csv('figure2_k8_mechanism.csv',mechanism)
for r in pilot:
    if '/treatment/' in r['case_arm']:
        t=js(E+'pilot/'+r['case_arm']+'/timing.json');r['treatment_end_to_end_seconds']=t['total_stage_s'];r['dive_incumbent_buses']=t['dive_integer_solution']['buses']
    else:r['treatment_end_to_end_seconds']='';r['dive_incumbent_buses']=''
write_csv('figure2_k8_pilot.csv',pilot);write_csv('figure2_c1_followup.csv',follow)
fig,axs=plt.subplots(1,2,figsize=(11,3.6),sharey=True)
for arm,shift,color,marker in [('control',-.1,BLUE,'o'),('augmented',.1,GREEN,'s')]:
    rs=sorted([r for r in witness if '/'+arm+'/' in r['case_arm']],key=lambda r:r['case_arm'])
    for i,r in enumerate(rs,1):
        axs[0].plot([i+shift]*2,[float(r['fleet_bound']),float(r['fleet_buses'])],color=color)
        axs[0].plot(i+shift,float(r['fleet_bound']),'_',color=color)
        axs[0].plot(i+shift,float(r['fleet_buses']),marker,color=color,ms=6,label={'control':'Fresh pool','augmented':'Fresh + sequential witness routes'}[arm] if i==1 else None)
axs[0].set_xticks(range(1,6),[f'C{i}' for i in range(1,6)]);axs[0].set_xlabel('GIRO target 8 buses · witness experiment')
cs=[1,3,4,5]
for arm,shift,color,marker,label in [('control_arm_a',-.14,BLUE,'o','Pool MIP · 1 hour'),('control_arm_b',0,GRAY,'x','Pool MIP · 1 hour + setup allowance'),('treatment',.14,GREEN,'s','Original dive + final MIP')]:
    rs=sorted([r for r in pilot if '/'+arm+'/' in r['case_arm']],key=lambda r:r['case_arm'])
    axs[1].scatter(np.arange(4)+shift,[float(r['fleet_buses']) for r in rs],c=color,marker=marker,s=35,label=label)
axs[1].scatter([.27],[float(follow[0]['fleet_buses'])],c=PURPLE,marker='D',s=42,label='C1 later incumbent transfer')
axs[1].set_xticks(range(4),[f'C{i}' for i in cs]);axs[1].set_xlabel('GIRO target 8 buses · diving pilot')
for ax in axs:ax.set_yticks([8,9]);ax.set_ylim(7.8,9.25);ax.legend(frameon=False,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02))
axs[0].set_ylabel('Integer buses');save(fig,'figure2_k8_column_enrichment')

write_csv('figure3_k15_12h.csv',long)
fig,ax=plt.subplots(figsize=(8.1,4.3))
for arm,offset,color,marker in [('plain',-.14,BLUE,'o'),('heuristic',.14,ORANGE,'s')]:
    rs=sorted([r for r in long if '__'+arm in r['case_arm']],key=lambda r:r['case_arm'])
    # Some source labels abbreviate heuristic settings to heur.
    if not rs:rs=sorted([r for r in long if '__heur' in r['case_arm']],key=lambda r:r['case_arm'])
    check(len(rs)==6,arm+' has six k15 searches')
    for i,r in enumerate(rs):
        y=i+offset;lo=float(r['fleet_bound']);hi=float(r['fleet_buses'])
        ax.plot([lo,hi],[y,y],color=color,lw=1.7);ax.plot(lo,y,'|',color=color,ms=11)
        ax.plot(hi,y,marker,color=color,ms=6,label={'plain':'Plain search','heuristic':'Heuristic-focused search'}[arm] if i==0 else None)
ax.set_yticks(range(6),[f'C{i}' for i in range(1,7)]);ax.invert_yaxis();ax.set_xticks(range(15,20));ax.set_xlim(14.8,19.3)
ax.set_xlabel('Integer buses: saved-pool bound to best solution');ax.set_ylabel('GIRO target 15 buses');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2)
save(fig,'figure3_k15_open_bounds')

pairs=rows(P+'status_20260913T063519Z/pairs.csv');collection=js(P+'status_20260913T063519Z/collection.json');manifest=js(P+'manifest.json')
cr={(r['pair_id'],r['arm']):r for r in collection['cg']}; mr={(r['pair_id'],r['arm']):r for r in collection['mip']}
for r in pairs:
    for side in ('control','treatment'):
        a=cr[r['pair'],r[side]]
        check(a['sha256']==r[side+'_cg_sha256'] and same(a['wall_s']/60,r[side+'_cg_min']),r['pair']+' '+side+' payload identity/time')
        if r[side+'_mip_sha256']: check(mr[r['pair'],r[side]]['sha256']==r[side+'_mip_sha256'],r['pair']+' '+side+' MIP identity')
write_csv('figure4_controlled_algorithms.csv',pairs)
fig,ax=plt.subplots(figsize=(9,3.6));cases=['w1_k08','w4_k10','w3_k15'];colors=[BLUE,ORANGE,GREEN]
for j,contrast in enumerate(['index','master','pool']):
    for c,(case,color) in enumerate(zip(cases,colors)):
        rs=[r for r in pairs if r['contrast']==contrast and r['case']==case]
        for r in rs:ax.scatter(float(r['cg_time_reduction_percent']),j+(c-1)*.15,c=color,marker='o' if r['repetition']=='1' else 's',s=38,label=case.replace('w','C').replace('_k',' / k') if j==0 and r['repetition']=='1' else None)
ax.set_yticks(range(3),['Indexed route replay','Omit unused LP setup','Full vs 512-route inheritance']);ax.invert_yaxis();ax.set_xlabel('Reduction in complete CG time (%)');ax.set_xlim(0,70)
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.19));save(fig,'figure4_controlled_algorithms')

benchpath='outputs/week_20260921/capacity_strict/recovery/benchmark/646674_r0/result.json';bench=js(benchpath)
check(bench['all_five_fixed_duals_match'] and bench['physical_replay_pass'],'packed benchmark checks passed')
br=[]
for key,d in bench['results'].items():
    check(len(d['pricing'])==5 and all(p['physical_replay_pass'] for p in d['pricing']),key+' five route replays')
    br.append({'implementation':key,'build_seconds':d['build_s'],'peak_process_gib':d['peak_rss_kib']/1024**2,'mean_pricing_seconds':statistics.mean(p['pricing_s'] for p in d['pricing']),
      'input_sha256':d['input_sha256'],'event_lattice_sha256':d['metrics']['event_lattice_sha256'],'retained_arcs':d['metrics']['dag_arcs'],'fixed_dual_vectors':5,'instances':1})
check(len({r['input_sha256'] for r in br})==1 and len({r['event_lattice_sha256'] for r in br})==1,'packed benchmark identical input and lattice')
for i in range(5):check(len({d['pricing'][i]['rc'] for d in bench['results'].values()})==1,f'fixed dual {i} identical minimum reduced cost')
write_csv('figure5_packed_benchmark.csv',br)
fig,axs=plt.subplots(1,3,figsize=(11,3.3))
for ax,field,label in zip(axs,['build_seconds','peak_process_gib','mean_pricing_seconds'],['Graph build (seconds)','Peak process memory (GiB)','Mean pricing call (seconds)']):
    ax.bar(range(3),[r[field] for r in br],color=[GRAY,BLUE,GREEN],width=.65);ax.set_xticks(range(3),['Original\nexplicit','Deferred\nexplicit','Deferred\npacked']);ax.set_ylabel(label)
    if field=='mean_pricing_seconds':ax.set_yscale('log');ax.set_ylim(.01,100)
fig.subplots_adjust(wspace=.38);save(fig,'figure5_packed_benchmark')

# All descriptive claims and settings remain editable outside figures.
budget_manifest=js(C+'manifest.json')
k15_manifest=js(E+'k15_manifest.json')
packed_manifest=js('outputs/week_20260921/capacity_strict/recovery/manifest.json')
read(E+'source_manifest.json')
read(E+'pool_identity_remote_audit.json')
for p in (ROOT/E/'pilot').glob('*/*/*/execution.json'): read(p)
experiment_settings={
 'cumulative_budget':{k:budget_manifest[k] for k in ('execution_commit','mip_execution_commit','settings','resources','static_sha256','warm_reference','interpretation')},
 'controlled_algorithms':{k:manifest[k] for k in ('cg_commit','mip_commit','common','arms','resources','inputs')},
 'k15_12h':k15_manifest,
 'strict_packed':packed_manifest,
 'k8_witness_and_pilot':{'source_manifest':E+'source_manifest.json','pool_identity_audit':E+'pool_identity_remote_audit.json',
   'selection':'five witness cases; four pilot cases selected from proved nine-bus pools',
   'scope':'source result hashes and execution commits preserved per row; full logs independently checked; original pilot and later C1 follow-up remain separate'},
 'new_computation':'none; figures use local evidence only',
 'dependencies':'historical sequential pools preserve actual previous-k ancestry; fresh MIPs depend on their fresh CG; warm-reference MIPs are independent reads of target pools; no submission in this package'}
(OUT/'experiment_settings.json').write_text(json.dumps(experiment_settings,indent=2)+'\n')
audit={'generated_by':'build_figures.py','repository_head_at_build':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'scope':'local read-only reanalysis; no new optimization or submission','source_files':SOURCES,
 'checks':CHECKS,'check_count':len(CHECKS),'paired_instances':24,'controlled_instances':3,'controlled_order_repeats':2,
 'packed_instances':1,'packed_fixed_dual_vectors':5,'k15_distinct_pools':6,'k15_search_configurations':2,
 'paired_settings':snap['physics'],'paired_source_hashes':snap['source_hashes'],'controlled_settings':manifest['common'],
 'controlled_execution_commit':manifest['cg_commit'],'controlled_mip_execution_commit':manifest['mip_commit'],
 'proof_scopes':{'cg':'conservative expanded event-grid weighted objective at recorded reduced-cost tolerance',
 'mip':'finite saved-pool fleet proof only','dispatch':'route replay does not establish exact-once service or shared capacity'},
 'output_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ('.csv','.png','.pdf','.svg','.py','.md') or p.name=='experiment_settings.json'}}
(OUT/'provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'checks_passed':len(CHECKS),'sources':len(SOURCES),'figures':5,'fresh_targets':sum(int(r['fresh_buses'])==int(r['target_k']) for r in paired),'sequential_targets':sum(int(r['warm_buses'])==int(r['target_k']) for r in paired)}))
