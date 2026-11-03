"""Collect only the three registered k8 SOC-grid pairs. Run on Unicorn with its EVSP Python.

Default stdout is public aggregate JSON. --native-envelope adds the small original
JSON receipts and accounting (private). No scheduler or experiment source mutations.
"""
import argparse, base64, collections, csv, datetime, hashlib, io, json, math, os, re, subprocess, tempfile
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-envelope',action='store_true'); args=parser.parse_args()
root=Path.home()/'ladder-lite'; new=root/'partille_soc_grid_20261103'; owner=root/'capacity_v15_20261002'; repo=owner/'code'; reg=new/'registration'; src=repo/'src'; slurm=Path('/usr/local/slurm/slurm-25.05.5/bin');py=Path.home()/'evsp_env/bin/python'
expected_registration='2010e4e0d875e547fac3f9d2303b0dce6b3d6db355b02552deb602eaebca7310'; expected_array=917989
native={}; utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def snapshot(path,name):
    b=path.read_bytes(); native[name]=base64.b64encode(b).decode();return json.loads(b),sha_bytes(b)

def run(*a):return subprocess.check_output(list(map(str,a)),text=True).strip()

def take(obj,fields):return {k:obj.get(k) for k in fields.split()}

def accounts(ids,start,end=None):
    names='JobID,JobIDRaw,State,ExitCode,Partition,AllocCPUS,ReqMem,ElapsedRaw,MaxRSS,TotalCPU,CPUTimeRAW,Start,End,NodeList'
    cmd=[slurm/'sacct','-nP','-S',start,'-j',ids,'--format='+names];
    if end:cmd.extend(['-E',end])
    raw=run(*cmd); native['sacct_'+ids+'.txt']=base64.b64encode((raw+'\n').encode()).decode(); rows={r['JobID']:r for r in csv.DictReader(io.StringIO(raw),fieldnames=names.split(','),delimiter='|')};return rows

def accounting(rows,key):
    a=rows[key];b=rows.get(key+'.batch',{});rss=b.get('MaxRSS') or None
    require(a['ReqMem']=='48G' and a['Partition']=='default_partition' and a['AllocCPUS']=='8','resource mismatch')
    if rss:require(re.fullmatch(r'[0-9]+K',rss),'unexpected RSS unit')
    return {'job_id':key,'raw_job':a['JobIDRaw'],'state':a['State'],'exit_code':a['ExitCode'],'partition':a['Partition'],'requested_cpus':int(a['AllocCPUS']),'requested_ram_gib':48,'elapsed_s':int(a['ElapsedRaw']),'batch_peak_rss_kib':int(rss[:-1]) if rss else None,'total_cpu':a['TotalCPU'],'allocated_cpu_seconds':int(a['CPUTimeRAW']),'raw_step_max_rss':rss,'start_cluster_local':a['Start'],'end_cluster_local':a['End']}

def mip_log(path):
    text=path.read_text(errors='replace');p=re.findall(r'Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds',text);t=re.findall(r'Thread count was (\d+)',text);cpu=re.findall(r'CPU model: ([^\n]+)',text)
    return {'gurobi_log_sha256':sha(path),'optimize_seconds':float(p[-1][2]) if p else None,'nodes':int(p[-1][0]) if p else None,'reported_threads':int(t[-1]) if t else None,'cpu_model':cpu[-1].split(', instruction set')[0] if cpu else None}

def service_floor(path,trusted):
    require(sha(path)==trusted,'wrong instance for fleet floor')
    times=collections.Counter();n=0
    def minutes(t):
        hour,minute=map(int,t.split(':'));return 60*hour+minute
    with path.open() as f:
        for n,r in enumerate(csv.DictReader(f),1):
            a,b=minutes(r['Start1']),minutes(r['End1']);require(a<b,'invalid trip');times[a]+=1;times[b]-=1
    load=peak=0
    for _,delta in sorted(times.items()):
        load+=delta;peak=max(peak,load)
    return {'trip_count':n,'peak_simultaneous_passenger_trips':peak,'instance_sha256':trusted}

def certificate(s):
    it=s['last_iteration']; result=take(it,'iteration phase rmp_objective route_weight min_reduced_cost'); result['cg_pricing_certified']=s['cg_pricing_certified'];result['stop']=s['cg_stop'];result['effective_tolerance']=s.get('certificate_tolerance');result['weighted_grid_lp_lower_bound']=s.get('weighted_full_graph_capacity_lp_lower_bound');
    if s['cg_pricing_certified']:
        require(it['phase']=='combined-cost' and it['min_reduced_cost'] >= -result['effective_tolerance'] and not it.get('smoothing_mispriced') and math.isclose(result['weighted_grid_lp_lower_bound'],it['rmp_objective']+s['fleet_cap']*min(0.,it['min_reduced_cost']),rel_tol=0.,abs_tol=1e-8),'weighted certificate arithmetic mismatch')
    return result

def collect(q,case,grid,october):
    path,jobkey,acct,recfile,exitfile,expected_sum=q
    row={'case':case,'soc_step_kwh':grid,'accounting':acct,'terminal_verified':False};sfile=path/'summary.json'
    if not sfile.exists():return row
    s,s_hash=snapshot(sfile,case+f'_{grid}_summary.json'); row.update(summary_sha256=s_hash,root=certificate(s),root_cg_seconds=s['cg_seconds'],root_lp_seconds=s['lp_seconds'],root_pricing_seconds=s['pricing_seconds'],root_columns=s['columns'],root_plug_rows=s['plug_rows_generated'],root_sha256=s['columns_sha256'])
    sa=json.loads((owner/'results/chain'/case/'parx60'/('job'+str(case_info['owner_raw_job'])+'_r0')/'summary.json').read_text());instance=Path(sa['inputs']['instance']); row.update(service_floor(instance,case_info['input_hashes'][instance.name]))
    if not october:
        start,st_hash=snapshot(path.parent/'start.json',case+'_start.json'); argv=start['argv'];require(start['manifest_sha256']==expected_registration and start['case']==case and start['grid_kwh']==grid and start['job']==acct['raw_job'] and argv.count('--soc-step')==1 and float(argv[argv.index('--soc-step')+1])==grid,'treatment start receipt mismatch');row['start_sha256']=st_hash
    if acct['state']!='COMPLETED' or not recfile.is_file() or (exitfile and not exitfile.exists()):
        h=hashlib.sha256();n=0
        with (path/'columns.jsonl').open('rb') as f:
            for _ in range(s['columns']):
                b=f.readline();require(bool(b),'truncated active root prefix');h.update(b);n+=1
        require(h.hexdigest()==s['columns_sha256'],'active root journal prefix mismatch');row['active_checkpoint']='authenticated root prefix; no terminal MIP or selection promoted';return row
    require(acct['exit_code']=='0:0','nonzero terminal accounting')
    if october: require(s_hash==expected_sum,'historical summary differs from registration')
    rec,rhash=snapshot(recfile,case+f'_{grid}_replay.json');require(rec['summary_sha256']==s_hash and rec['soc_step']==grid,'bad replay binding')
    if exitfile:
        ex,eh=snapshot(exitfile,case+'_exit.json'); require(ex['validated'] and ex['replay_sha256']==rhash and ex['summary_sha256']==s_hash and ex['state']==rec['selection_state'],'bad worker receipt');require((ex['solver_rc']==0)==(ex['state']=='selected'),'bad worker solver rc');row['exit_sha256']=eh
    with tempfile.TemporaryDirectory(prefix='partille-grid-readonly-') as temp:
        given=Path(temp)/'replayed.json'; env=dict(os.environ,PYTHONPATH=str(src),PYTHONDONTWRITEBYTECODE='1');subprocess.run([py,new/'code/replay.py','--run',path,'--output',given,'--soc-step',str(grid),'--start-fee','5','--expected-input-hashes',reg/(case+'_input_hashes.json')],env=env,check=True,stdout=subprocess.DEVNULL);require(given.read_bytes()==recfile.read_bytes(),'fresh independent replay did not reproduce published receipt byte-for-byte')
    if october:require(rec['final_pool_sha256']==case_info['control_post_dive_pool_sha256'] and rec['root_prefix_sha256']==case_info['control_root_sha256'] and rhash==case_info['replay_receipt_sha256'],'archived journal or replay differs from registration')
    else:require(rec['inputs_sha256']==case_info['input_hashes'],'new run inputs differ')
    if (path/'selected_routes.json').exists():require(sha(path/'selected_routes.json')==rec['selected_sha256'],'selected hash mismatch')
    row.update(terminal_verified=True,selection_replay=take(rec,'fleet extra_trip_visits exactly_once covers_all shared_plug_overloads grid_nonfleet_cost continuous_nonfleet_cost grid_terminal_energy_kwh per_route_replayed selected_sha256 final_pool_rows final_pool_sha256'),replay_sha256=rhash,replay_reproduced=True,graph_seconds=s['graph_seconds'],packed_arcs=s['packed_arcs'],capacity_window_keys=s['capacity_window_keys'],capacity_alternatives=s['capacity_alternatives'],terminal_alternatives=s['capacity_terminal_alternatives'],dive=take(s['dive'],'state steps fixed columns_after_dive'),fleet_stage=s['fleet_stage'],charging_stage=s.get('charging_stage'),mip_plug_rows=s['plug_rows_in_mip'])
    if s.get('charging_stage'):
        row['charging_stage_log']=mip_log(path/'charging.gurobi.log');row['fleet_stage_log']=mip_log(path/'fleet.gurobi.log')
    dive_path=path/'dive.json'
    if dive_path.is_file():
        event=json.loads(dive_path.read_bytes());row['dive']['trace_sha256']=sha(dive_path);row['dive']['secs_from_root_end_to_last_recorded_dive_event']=max(0.,event[-1]['elapsed_s']-s['cg_seconds']) if event and 'elapsed_s' in event[-1] else None
    return row

check_utc=utc();m,mh=snapshot(reg/'manifest.json','registration.json');require(mh==expected_registration,'wrong frozen registration');require(int((new/'submitted_job.txt').read_text())==expected_array,'wrong coarse array');require(run('git','-C',repo,'rev-parse','HEAD')==m['execution_commit'],'owner checkout moved');require(not run('git','-C',repo,'diff','HEAD','--name-only','--','src'),'owner source modified');require(all(sha(new/'code'/k)==v for k,v in m['protocol_files_sha256'].items()),'registered scripts changed')
controls=accounts('920164_7,920164_11,920164_23','2026-10-02','2026-10-05');coarse=accounts(str(expected_array),'2026-11-02'); results=[]
for idx,case_info in enumerate(m['cases']):
    case=case_info['case']; oldkey=case_info['owner_array_task'];op=owner/'results/chain'/case/'parx60'/('job'+str(case_info['owner_raw_job'])+'_r0');results.append(collect((op,oldkey,accounting(controls,oldkey),reg/(case+'_control_replay.json'),None,case_info['summary_sha256']),case,2.5,True))
    key=f'{expected_array}_{idx}';ac=accounting(coarse,key); attempts=sorted((new/'results'/case/'soc_5').glob('job'+ac['raw_job']+'_r*'),key=lambda p:int(p.name.rsplit('_r',1)[1])); require(bool(attempts),'missing active attempt');np=attempts[-1];results.append(collect((np/'solve',key,ac,np/'replay.json',np/'exit.json',None),case,5.,False))
public={'schema':'registered_soc_grid_compact_v1','as_of_utc':check_utc,'verified_at_utc':utc(),'execution_commit':m['execution_commit'],'registration_sha256':mh,'replay_source_sha256':sha(new/'code/replay.py'),'phase_resources':'8 CPUs, 48 GiB requested; graph in fresh process, CG and simple dive single-thread master LP, both finite-pool MIP stages up to 8 threads. Batch MaxRSS is whole-job. Per-phase RSS was not instrumented. Wall times single attempts on different hosts/dates. Last trace minus root time is not a complete dive duration.','scope':'certificates apply to each conservative weighted grid LP, finite-pool optima apply to that grid/run pool; separate continuous replays and exactly-once service audits','observed_subsets':results}
print(json.dumps({'public':public,'private_small_files_base64':native} if args.native_envelope else public,indent=2,allow_nan=False))
