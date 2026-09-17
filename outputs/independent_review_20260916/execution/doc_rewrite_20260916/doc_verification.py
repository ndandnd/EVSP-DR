"""Verify exported tab content, audited counts and preserved sibling tabs/footer."""
from pathlib import Path
import hashlib,json,re
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;E=O.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
norm=lambda s:re.sub(r'\s+',' ',s.replace('\\','').replace('**','').replace('*','').replace('`','')).strip()
exports={'after':'797ec386-6d09-4755-aff2-e3b81ee4b5e6','history_after':'3e9457a0-5e42-4acb-a131-3b0744b90a01','figures_after':'a9e8e3c1-7419-42d1-ac6b-c16f16819027','curves_after':'ff8a83b3-1d89-4f14-bcef-dc525fff17f5'}
for key,uuid in exports.items():
 p=Path('/var/folders/hk/6g28n39j5_s33j2kn2w4xmp40000gn/T/browser-use/exports')/('EVSP DR Current Research-'+uuid+'.md')
 if p.exists():(O/(key+'.md')).write_bytes(p.read_bytes())
after=(O/'after.md').read_text();html=(O/'after.html').read_text();wanted=(O/'resolved.md').read_text()
counts=json.loads((E/'advisor_followup_20260916/proof_counts/counts.json').read_text())
checks={'102_instances':counts['total']==102,'67_closed':counts['including_audited_longer']['matched_numerical_bound']==67,'35_open':counts['including_audited_longer']['open_fleet_gaps']==35,'128_replays':len((E/'f1/per_case.csv').read_text().splitlines())-1==128,'four_editable_tables':html.count('<table')==4,'footer_html_preserved':(O/'source_footer.html').read_text() in html}
for name in ['history','figures','curves']:
 before=(O/(name+'_before.md')).read_text();other=(O/(name+'_after.md')).read_text()
 checks[name+'_unchanged']=before==other
for t in ['22:22','07:22','21:55','03:04','07:04','21 Sep, 13:24','17 Sep, 22:37','18 Sep, 18:12','Held; no scheduled start','Replay started 16 Sep, 20:22','17.38%','8.28%','67','35','70','128']:
 checks['contains_'+t]=t in norm(after)
# Compare non-table source paragraphs after normalizing exporter markdown wrapping.
paragraphs=[p for p in wanted.split('\n\n') if p and not p.startswith('|')]
missing=[]
for p in paragraphs:
 p=re.sub(r'^#+\s*','',p);p=re.sub(r'\n#+\s*','\n',p)
 if norm(p) not in norm(after):missing.append(p[:100])
checks['replacement_prose_present']=not missing
links=re.findall(r'href="([^"]+)"',(O/'source_footer.html').read_text())
checks['all_footer_links_in_export']=all(l in after for l in links)
r={'verified_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'passed':all(checks.values()),'missing_paragraphs':missing,'before':{'current_export_sha256':sha(O/'before.md'),'current_html_sha256':sha(O/'before.html')},'after':{'current_export_sha256':sha(O/'after.md'),'current_html_sha256':sha(O/'after.html'),'saved_to_drive_observed':True,'pdf_sha256':sha(O/'after.pdf')},'sibling_exports':{n:{'before':sha(O/(n+'_before.md')),'after':sha(O/(n+'_after.md'))} for n in ['history','figures','curves']},'slides_edited':False,'timing_evidence':'expected_by_job.csv + slurm_timing.txt','visual_check':'Five pages inspected; tables editable and readable; retained six source/tab footer links.'}
(O/'doc_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['passed']
