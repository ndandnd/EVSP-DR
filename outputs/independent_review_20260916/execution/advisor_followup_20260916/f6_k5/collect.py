"""Read-only F6 endpoints; no submits/retries; ratios require matched fleets."""
from pathlib import Path
import argparse,csv,json
from process_support import read,save,sha,now

def main(root,out):
 b=Path(root);o=Path(out);o.mkdir(parents=True,exist_ok=True);m=read(b/'manifest.json');original=read(b/'original_revalidation.json');rows=[]
 for peak,source in original.items():
  p=source['physical'];invoice=source['invoice'];rows.append(dict(peak=peak,arm='original_GIRO',status='verified_profile_exists',buses=p['buses'],continuous_charging_cost=invoice['uniform_power_assumption_energy_cost'],cost_lower=invoice['charging_cost_lower'],cost_upper=invoice['charging_cost_upper'],cost_kind='interval;uniform-profile estimate is not observed',minimum_soc_kwh=min(r['minimum_soc_kwh'] for r in p['per_bus']),active_charge_starts=sum(r['active_charge_starts'] for r in p['per_bus']),minimum_active_minutes=min(r['shortest_active_minutes'] for r in p['per_bus']),ending_kwh=p['aggregate_continuous_ending_kwh'],duplicate_occurrences=0,matched_five_bus_comparison_eligible=True))
 for cid,c in m['cases'].items():
  row=dict(case_id=cid,peak=c['peak'],arm=c['arm'],status='not_completed',matched_five_bus_comparison_eligible=False);p=b/'cases'/cid/'completion.json'
  if p.exists():
   completion=read(p);r=read(completion['result_path']);audit=read(completion['physical_audit_path']);buses=audit['buses'];row.update(status=completion['scientific_status'],buses=buses if buses else None,continuous_charging_cost=r.get('selected_continuous_charging_cost'),grid_charging_cost=(r.get('charging_stage') or {}).get('cost',r.get('charging_cost')),cg_pricing_certified=r.get('cg_pricing_certified'),minimum_soc_kwh=min((x['minimum_soc_kwh'] for x in audit['per_bus']),default=None),active_charge_starts=sum(x['active_charge_starts'] for x in audit['per_bus']),idle_visits=sum(x['idle_station_visits'] for x in audit['per_bus']),minimum_active_minutes=min((x['shortest_active_minutes'] for x in audit['per_bus'] if x['shortest_active_minutes'] is not None),default=None),ending_kwh=audit['aggregate_continuous_ending_kwh'] if buses else None,duplicate_occurrences=audit['duplicate_occurrences'],matched_five_bus_comparison_eligible=completion['matched_five_bus_cost_comparison_eligible'],worker_wall_s=completion['execution']['wall_s'],graph_s=r.get('graph_seconds'),cg_s=r.get('cg_seconds'),frontier_s=r.get('frontier_seconds'),lp_s=r.get('lp_seconds'),pricing_s=r.get('pricing_seconds'),result_path=completion['result_path'],result_sha256=completion['result_sha256'])
  rows.append(row)
 comparisons=[]
 for peak in ['peak08','peak12','peak18']:
  arms={r['arm']:r for r in rows if r['peak']==peak};f,j=arms['fixed'],arms['cg'];eligible=all(x['matched_five_bus_comparison_eligible'] and x.get('continuous_charging_cost') is not None for x in [f,j]);comparisons.append(dict(peak=peak,status='available_same_fleet' if eligible else 'censored_pending_or_unmatched',joint_vs_fixed_fractional_cost_reduction=(1-j['continuous_charging_cost']/f['continuous_charging_cost']) if eligible and f['continuous_charging_cost']>0 else None,same_achieved_ending_energy=(abs(f['ending_kwh']-j['ending_kwh'])<=1e-6) if eligible else None,common_ending_minimum_kwh=m['terminal_target_kwh'],scope='Continuous modeled costs; no continuous-cost global proof. Covering duplicates retained and separately reported; equal minimum end energy does not force equal achieved end energy.'))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (o/'three_arm_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 save(o/'comparison.json',dict(utc=now(),manifest_sha256=sha(b/'manifest.json'),rows=rows,comparisons=comparisons));print(json.dumps(dict(cells=len(rows),completed_optimized=sum(r['arm'] in ['cg','fixed'] and r['status']!='not_completed' for r in rows))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)
