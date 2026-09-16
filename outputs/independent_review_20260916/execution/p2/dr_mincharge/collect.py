"""Read endpoints; keep all real-price-derived values in an internal directory."""
from pathlib import Path
import argparse,csv,json
from process_support import read,save,sha,now

def main(root,out):
 b=Path(root);o=Path(out);o.mkdir(parents=True,exist_ok=True);m=read(b/'manifest.json');public=[];private=[]
 for cid,c in m['cases'].items():
  p=b/'cases'/cid/'completion.json';row=dict(case_id=cid,pair=c['pair'],chain=c['chain'],tariff=c['tariff'],arm=c['arm'],k=c['k'],status='not_completed',terminal_target_kwh=c['terminal_target_kwh'])
  if p.exists():
   comp=read(p);r=read(comp['result_path']);row.update(status=comp['status'],worker_wall_s=comp['execution']['wall_s'],result_path=comp['result_path'],result_sha256=comp['result_sha256'],cg_stop=r.get('cg_stop'),cg_pricing_certified=r.get('cg_pricing_certified'),graph_s=r.get('graph_seconds'),cg_s=r.get('cg_seconds'),pricing_s=r.get('pricing_seconds'),lp_s=r.get('lp_seconds'),frontier_complete=r.get('frontier_complete'),infeasible_duties=json.dumps(r.get('infeasible_duties')),buses=(r.get('charging_stage') or {}).get('fleet',r.get('fleet')),grid_charging_cost=(r.get('charging_stage') or {}).get('cost',r.get('charging_cost')),continuous_charging_cost=r.get('selected_continuous_charging_cost'),active_charge_starts=r.get('selected_active_charge_starts'),idle_visits=r.get('selected_idle_station_visits'),terminal_grid_kwh=r.get('selected_terminal_kwh'),terminal_continuous_kwh=r.get('selected_continuous_terminal_kwh'),positive_active_minimum_verified=comp.get('positive_active_minimum_verified'))
  (private if c['publication']=='internal_only' else public).append(row)
 def write(rows,path):
  path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
  with path.open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 write(public,o/'synthetic_results.csv');write(private,o/'private_internal/se3_results.csv');save(o/'collection.json',dict(utc=now(),manifest_sha256=sha(b/'manifest.json'),public_synthetic_arms=len(public),internal_only_real_arms=len(private),publication_rule='Publish synthetic_results.csv only; private_internal contains restricted real-price-derived results.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)
