"""Future authorized preprocessing: hash-check original pool and dedupe sequences.

Reads the approximately 985 MB original journal. Not run during preparation.
"""
import argparse,json,sys,subprocess
from pathlib import Path
from sequence_replay import sha,canonical,ordered_sequence
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 m=json.load(open(a.root/'manifest.json'));sys.path.insert(0,str(a.code/'src'))
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.code,text=True).strip()==m['execution_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.code,text=True).strip()
 from run_exact_pool_mip import load_pool,ordered_pool_sha256,resolve_pool_journal
 assert sha(m['source_status'])==m['source_status_sha256']
 status=json.load(open(m['source_status']));journal=resolve_pool_journal(Path(m['source_status']),status)
 assert sha(journal)==m['source_journal_sha256']
 _,routes,trip_ids=load_pool(Path(m['source_status']))
 assert len(routes)==m['source_ordered_pool_columns']
 assert ordered_pool_sha256(routes)==m['source_ordered_pool_sha256']
 assert trip_ids==list(range(m['trip_count']))
 unique={}
 for index,route in enumerate(routes):
  seq=ordered_sequence(route);key=canonical(seq)
  rec=unique.setdefault(key,{'sequence_sha256':key,'trip_sequence':list(seq),'source_pool_indices':[], 'source_route_sha256':[], 'source_charge_start_counts':[]})
  rec['source_pool_indices'].append(index);rec['source_route_sha256'].append(canonical(route))
  rec['source_charge_start_counts'].append(len((route.get('expanded_grid_charging_stops') or route.get('charging_stops') or {}).get('stations',[])))
 a.out.mkdir(parents=True,exist_ok=False)
 with open(a.out/'sequences.jsonl','w') as f:
  for i,rec in enumerate(unique.values()):rec['sequence_index']=i;f.write(json.dumps(rec)+'\n')
 receipt={'source_journal':str(journal),'source_journal_sha256':sha(journal),'source_status_sha256':sha(m['source_status']),
 'source_ordered_pool_sha256':ordered_pool_sha256(routes),'source_columns':len(routes),'unique_ordered_sequences':len(unique),
 'sequences_sha256':sha(a.out/'sequences.jsonl'),'trip_count':len(trip_ids),'scope':'full_source_pool','definition':'First reconstruct the identical cheapest-per-trip-incidence MIP pool; then deduplicate by chronological route_nodes trip sequence, not frozenset or sorted IDs.'}
 (a.out/'extraction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
 # A first-20 pilot would mostly measure early singleton initializers. Sample
 # deterministically across ordered sequence-length / original-charge strata.
 ranked=sorted(unique.values(),key=lambda r:(len(r['trip_sequence']),max(r['source_charge_start_counts']),r['sequence_sha256']))
 n=min(20,len(ranked));indices=sorted({round(i*(len(ranked)-1)/max(n-1,1)) for i in range(n)})
 pilot=a.out/'pilot';pilot.mkdir()
 with open(pilot/'sequences.jsonl','w') as f:
  for i in indices:f.write(json.dumps(ranked[i])+'\n')
 (pilot/'extraction.json').write_text(json.dumps({**receipt,'scope':'pilot_subset_not_full_pool',
 'unique_ordered_sequences':len(indices),'sequences_sha256':sha(pilot/'sequences.jsonl'),
 'parent_sequences_sha256':receipt['sequences_sha256'],'sampling':'20 evenly spaced ranks after sorting by trip count, source charge starts, sequence hash'},indent=2)+'\n')
if __name__=='__main__': main()
