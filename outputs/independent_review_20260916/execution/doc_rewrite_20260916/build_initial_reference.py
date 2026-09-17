from pathlib import Path
import re,html,json,shutil,hashlib
E=Path('outputs/independent_review_20260916/execution');O=E/'doc_rewrite_20260916'
for name,file in [('figures','05e6490e-4243-464a-8565-cbe9ad7f142b'),('curves','0aa9a919-b7f9-4762-9031-31eb0c50248a')]:shutil.copyfile('/var/folders/hk/6g28n39j5_s33j2kn2w4xmp40000gn/T/browser-use/exports/EVSP DR Current Research-'+file+'.md',O/(name+'_before.md'))
shutil.copyfile('/tmp/doc_expected.py',O/'calculate_expected.py')
s=(E.parent/'DOC_REWRITE.md').read_text()
s=s.replace('*Updated 16 September 2026, evening. Replace the "current research" tab with this text. Keep the figure and history tabs.*','*Updated 16 September 2026, 20:06 EDT.*')
s=s.replace('we match it in 67;','we match the lower bound in 67;')
s=s.replace('| Bound matched by a validated schedule (after 3 h searches) | 67 |','| Bound matched by a validated schedule (including 3 h searches) | 67 |\n| GIRO fleet count matched (a different test) | 70 |\n| Open gap to the lower bound | 35 |')
s=s.replace('because we model the depot at 240 kW and GIRO uses 60 kW','under the baseline 240 kW replay')
s=s.replace('## Running now','## Running now\n\nTimes are EDT. For running jobs, Expected gives the Slurm allocation deadline (start + wall limit). Sequential projections use the remaining dependency chain at full wall limits, with zero queue delay; they are not scheduled finish times.')
replacements=[('17 Sep AM','16 Sep, 22:22'),('17 Sep','17 Sep, 07:22'),('17 Sep AM','16 Sep, 22:22 (8 new runs; 4 seed-zero results reused)'),('17 Sep','16 Sep, 21:55 (5/6 completed)'),('17–18 Sep','17 Sep, 03:04 fixed / 07:04 fresh CG'),('~19 Sep','21 Sep, 13:24 — full-budget dependency projection'),('~18 Sep','17 Sep, 22:37 — full-budget dependency projection'),('TBD','Held; no scheduled start'),('~18 Sep','18 Sep, 18:12')]
lines=s.splitlines(); ix=next(i for i,l in enumerate(lines) if l.startswith('| Fresh k=15 pools'))
for j,(old,new) in enumerate(replacements):
 l=lines[ix+j];assert l.split('|')[-2].strip()==old,(l,old);parts=l.split('|');parts[-2]=' '+new+' ';lines[ix+j]='|'.join(parts)
s='\n'.join(lines)+'\n'
s=s.replace('held — checkpoint fix + memory sizing first','held — checkpoint fix tested; memory sizing unresolved')
s=s.replace('- Cluster: 88 running / 79 pending on the shared tier. Approved to proceed with sequencing; no new submissions until the four results marked "17 Sep" arrive.','- Cluster: six full single-factor arms are now approved. Other new submissions remain on hold until the fresh-k15, C5 k31, constrained-k5 and k32 seed results are complete.')
s=s.replace('- Full-Partille job: fix checkpoint fallback, size memory from k=32 MaxRSS, move to `scaglione` partition if ≤ 120 GB.','- Full-Partille job: checkpoint fallback fixed and tested. k=32 aggregate MaxRSS is 206–232 GiB; the ≤120 GB condition for a Scaglione move is not met. CG and MIP remain held.')
# Keep source footer exactly; update linkable source paragraph through the footer.
footer=(O/'before.html').read_text().split('<p><a href="https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md">')[-1]
footer='<p><a href="https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/independent_review_20260916/execution/README.md">'+footer.split('</body>')[0]
(O/'source_footer.html').write_text(footer)
(O/'resolved.md').write_text(s)
def inline(x):
 x=html.escape(x)
 x=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',x);x=re.sub(r'(?<!\*)\*([^*]+)\*',r'<i>\1</i>',x);x=re.sub(r'`([^`]+)`',r'<code>\1</code>',x)
 return x
out=[];table=False
for l in s.splitlines():
 if l.startswith('|'):
  if re.match(r'^\|[ :|\-]+$',l):continue
  if not table:out.append('<table style="border-collapse:collapse;width:100%;font-size:10pt">');table=True;header=True
  else:header=False
  out.append('<tr>'+''.join('<'+('th' if header else 'td')+' style="border:1px solid #bfc8cf;padding:5px;vertical-align:top;'+('background-color:#eef2f5;font-weight:bold;' if header else '')+'">'+inline(c.strip())+'</'+('th' if header else 'td')+'>' for c in l.split('|')[1:-1])+'</tr>')
  continue
 if table:out.append('</table>');table=False
 if not l:continue
 if l.startswith('#'):
  n=len(l)-len(l.lstrip('#'));out.append(f'<h{n} style="color:#000000;font-size:{20 if n==1 else 14}pt;margin-top:18px;margin-bottom:8px">'+inline(l[n:].strip())+f'</h{n}>')
 else:out.append('<p style="margin:7px 0;line-height:1.2">'+inline(l)+'</p>')
if table:out.append('</table>')
res='<html><body style="font-family:Arial;font-size:11pt;color:#000000">'+''.join(out)+footer+'</body></html>'
(O/'after.html').write_text(res)
(E/'doc/current.html').write_text(res)
(O/'changes.json').write_text(json.dumps({'source_sha256':hashlib.sha256((E.parent/'DOC_REWRITE.md').read_bytes()).hexdigest(),'counts':{'schedules':128,'instances':102,'bound_matched':67,'open':35,'target_matched':70},'timing_basis':'ledger-referenced submitted job records + live Slurm starts/limits; EDT','corrections':['67 labelled lower-bound matches;70GIRO matches separate','0/42 replay clause removes unsupported sole-cause assertion','replaced outdated dates and held-job state','latest6fullarms authorization reflected'],'source_footer_preserved':True},indent=2)+'\n')
print('Built',len(res),'characters;',res.count('<table'),'editable tables')
