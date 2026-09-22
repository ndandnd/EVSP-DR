from pathlib import Path
import json,csv,hashlib,shutil
P=Path(__file__).resolve().parent;C=P.parents[1]/'mip_structure';d=json.loads((P/'verification.json').read_text());rows=d['rows'];cases=['c1_k08_fresh','c4_k08_fresh','c1_k15_fresh','c3_k15_fresh','c1_k15_sequential'];arms=['default','focus1','focus2','presparsify1','strong_start'];labels={'c1_k08_fresh':'C1 k8 fresh','c4_k08_fresh':'C4 k8 fresh','c1_k15_fresh':'C1 k15 fresh','c3_k15_fresh':'C3 k15 fresh','c1_k15_sequential':'C1 k15 sequential'}
s='''# Fixed-pool MIP pilot: all 25 endpoints complete

One scoped scheduler snapshot at **2026-09-22 07:56:15 UTC** found all 25 active trials COMPLETED. All five preparations and 25 endpoint receipts are valid. **738 checks pass**: original ordered-pool/matrix hashes, source and manifest pins, recorded physical admission, intended starts and selected-route identities, exact model dimensions, settings, full native-log hashes, objective/bound, termination and timing checks. Fifteen trials prove their finite-pool fleet optimum; ten reach their 1,800-second optimizer allowance normally. No recovery or new submission is required.

[All 25 editable results and provenance](results.csv) · [Verification and source hashes](verification.json) · [Independent source/start audit](source_identity_audit.json) · [Scheduler snapshot](scheduler_snapshot.json) · [Full solver logs and completed artifacts](../../mip_structure/collections/20260922T075615Z/receipt.json).

## Fleet outcomes: incumbent / bound

A ✓ marks a finite-pool proof. Every k8 optimum is9, above target8. All sequential C1 k15 arms attain and prove target15. Neither fresh k15 pool reaches target15 in this pilot.

| Pool | Default | MIPFocus1 | MIPFocus2 | PreSparsify1 | Offline saved start |
|---|---:|---:|---:|---:|---:|
'''
for c in cases:
 rr=[next(r for r in rows if r['case']==c and r['arm']==a) for a in arms];s+='|'+labels[c]+'|'+'|'.join(f"{r['fleet']} / {r['bound']:.0f}"+(' ✓' if r['finite_pool_fleet_proven'] else ' open') for r in rr)+'|\n'
s+='''
## Actual optimizer wall seconds

Times exclude physical preparation, artifact loading, model construction and original saved-incumbent acquisition. † denotes time-limit termination, rather than time to proof. Source CSV retains full precision, final Gurobi Runtime, native log explored time/work, and all loading/building times separately.

| Pool | Default | MIPFocus1 | MIPFocus2 | PreSparsify1 | Offline saved start |
|---|---:|---:|---:|---:|---:|
'''
for c in cases:
 rr=[next(r for r in rows if r['case']==c and r['arm']==a) for a in arms];s+='|'+labels[c]+'|'+'|'.join(f"{r['actual_optimize_wall_s']:.3f}"+('†' if r['status']=='TIME_LIMIT' else '') for r in rr)+'|\n'
s+='''
## Interpretation and limits

- **Solver settings have mixed effects.** MIPFocus2 proves C1 k8 in849.588s versus default1232.992s (31.1% less time), but on C4 k8 default is fastest (461.957s). All five arms prove9 on both pools; no setting creates an8-bus route cover in those frozen pools.
- **PreSparsify1 is not a general improvement.** It improves the C3 k15 incumbent from18 to17 at the same allowance, but worsens C1 k15 fresh from18 to19. Sequential C1 k15 still proves15, taking1779.487s versus default104.168s (17.08×). These are one-seed, different-node observations.
- **The saved start has limited scope.** Sequential C1 k15's saved15-bus incumbent proves in11.150s versus ordinary greedy-start default104.168s. The ordinary-start MIPFocus2 run proves15 in80.752s. Saved-start acquisition occurred in an earlier run and is excluded here. C3's saved18-bus start later improves to17; C1 fresh's saved18-bus start remains18. A precomputed incumbent does not resolve either fresh15-bus target.
- **Finite-pool scope is unchanged.** Binary set covering, unit fleet objective, no new route generation, no applied dominance/screening reductions, no charging MIP or added capacity constraints. Physics remains240kWh battery/initial SOC, constant240kW charging, zero reserve, no terminal floor/shared capacity, original flat tariff/start fee5. Duplicate passenger coverage can occur. Native physical admission/ordered-pool evidence was authenticated; this monitor does not independently replay every physical route.
- **Timing revisions are explicit.** The15 original trials retain their slow NPZ loader; the10 replacement trials use the validated once-only loader. Report optimizer times for this comparison, not lifecycle totals across revisions. Final Gurobi Runtime exceeds the native log's rounded “Explored…seconds” line by at most0.277s; both are retained, and measured optimize wall agrees with final Runtime within0.01s. Time-limit wall overshoot is at most0.357s.

Original failed preparations729457/729775 and canceled loading/pending attempts remain preserved in the campaign. No historical holds were changed, no broad queue check was made, and no new jobs were launched. All five case scans previously completed with zero duplicate-incidence columns, zero redundant covering rows and one connected component; those diagnostics did not alter these trials.
'''
(P/'README.md').write_text(s)
summary=[]
for c in cases:
 rr=[r for r in rows if r['case']==c];base=next(r for r in rr if r['arm']=='default');summary.append({'case':c,'completed':len(rr),'proved':sum(r['finite_pool_fleet_proven'] for r in rr),'target_attained':sum(r['target_attained'] for r in rr),'best_fleet':min(r['fleet'] for r in rr),'worst_fleet':max(r['fleet'] for r in rr),'default_fleet':base['fleet'],'default_optimizer_s':base['actual_optimize_wall_s'],'all_final_bounds_rounded':round(base['bound'])})
with (P/'pool_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
old=C/'latest_recovery.json';shutil.copy2(old,P/'prior_latest_recovery.json');latest=json.loads(old.read_text());latest.update(created_utc=d['created_utc'],status='All25 active trials completed;15 finite-pool proofs and10 normal optimizer time limits; no active recovery needed.',completed_trial_count=25,completed_preparation_count=5,result_table='../monitor_20260922T075504Z/mip_structure/results.csv',snapshot='snapshots/20260922T075615Z.json',collection='collections/20260922T075615Z/receipt.json',endpoint_audit='../monitor_20260922T075504Z/mip_structure/verification.json',endpoint_report='../monitor_20260922T075504Z/mip_structure/README.md',recovery_needed=False)
latest['sha256']['../monitor_20260922T075504Z/mip_structure/results.csv']=hashlib.sha256((P/'results.csv').read_bytes()).hexdigest();old.write_text(json.dumps(latest,indent=2)+'\n')
f=C/'README.md';s=f.read_text();heading=s.split('\n',1)[0];update='''

**Final result update, 22 September07:56UTC:** all25 trials completed and738 audit checks pass. Fifteen prove the finite-pool fleet optimum; ten terminate normally at the30-minute allowance. All k8 arms prove9; all sequential C1 k15 arms prove15. Fresh C1 k15 ends18 except PreSparsify1=19; fresh C3 k15 ends18 except PreSparsify1/saved start=17, all bound15/open. Solver-setting effects are mixed; saved-incumbent acquisition and loading revisions remain excluded from optimizer-time comparisons. [Final25-cell tables, full logs and audit](../monitor_20260922T075504Z/mip_structure/README.md). No recovery is needed. The earlier partial snapshots below remain historical evidence.
''';f.write_text(heading+update+s[len(heading):])
(P/'receipt.json').write_text(json.dumps({'passed':True,'checks':738,'snapshot_utc':d['snapshot_utc'],'completed_trials':25,'finite_pool_proofs':15,'normal_time_limits':10,'target_matches':5,'new_submissions':0,'recovery_needed':False,'updated_campaign_files':['README.md','latest_recovery.json'],'source_artifact_count':len(d['source_artifacts']),'report_sha256':hashlib.sha256((P/'README.md').read_bytes()).hexdigest(),'table_sha256':hashlib.sha256((P/'results.csv').read_bytes()).hexdigest(),'verification_sha256':hashlib.sha256((P/'verification.json').read_bytes()).hexdigest()},indent=2)+'\n')
