from pathlib import Path
import sqlite3
import worker as w
B=Path(__file__).resolve().parent;ix=w.read(B/'source_index.json');rows=[]
for pair,g in ix['groups'].items():
 d=sqlite3.connect(B/'pools'/('_build_'+pair)/'donor.sqlite');v=w.read(g['donor']['status_path']);keys=[__import__('json').dumps(sorted(r['trips']),separators=(',',':')) for r in v['final_lp']['positive_routes'] if r['value']>0]
 for r in g['recipients']:
  old=sqlite3.connect(B/'pools'/('_build_'+pair)/(r['case_id']+'.sqlite'));shared=0;different=0;maxdiff=0.
  for k in keys:
   a=old.execute('SELECT cost,payload FROM cols WHERE tripkey=?',(k,)).fetchone();b=d.execute('SELECT cost,payload FROM cols WHERE tripkey=?',(k,)).fetchone()
   if a:
    shared+=1;maxdiff=max(maxdiff,abs(a[0]-b[0]));different+=__import__('json').loads(a[1])['trips']!=__import__('json').loads(b[1])['trips']
  rows.append(dict(case_id=r['case_id'],shared_positive_incidences=shared,max_shared_cost_difference=maxdiff,shared_ordered_trip_paths_differ=different));old.close()
 d.close()
w.save(B/'shared_support_audit.json',dict(manifest_sha256=w.sha(B/'manifest.json'),rows=rows,scope='Audit shared donor positive incidences retained from recipient; added treatments never replace old columns.'))
print(rows)
