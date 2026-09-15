"""Preserve the first passed fixture and correct the measured timeout description."""
import campaign as c
v=c.read(c.B/'manifest.json')
old=c.sha(c.B/'manifest.json')
for name in ['manifest.json','input_validation.json','validation.json','validation_submission.json','validation_scheduler_initial.json']:
 p=c.B/name
 if p.exists(): p.rename(c.B/'preparation'/('before_resource_evidence_'+name))
v['resource_basis']=v['resource_basis'].replace('Campaign14 C1 k28 native graph exceeded its original 12.5h Slurm allocation; recovery job189917 uses a longer allowance.', 'Campaign14 C1 k28 graph187967_2 failed124 after12:01:11 at its12h native watchdog, within its12:30 Slurm allocation (sacct); recovery job189917 uses24h native watchdog and24:30 allocation.')
v['tooling_sha256']['campaign.py']=c.sha(c.B/'campaign.py')
v['prelaunch_amendment']={'superseded_manifest_sha256':old,'reason':'Correct timeout cause from native12h watchdog evidence; resources and scientific settings unchanged. Add explicit validation-tooling hash equality gate. Repeat bounded native fixture against final manifest.', 'resource_evidence':'preparation/graph_resource_evidence.txt','resource_evidence_sha256':c.sha(c.B/'preparation/graph_resource_evidence.txt'),'superseded_validation_job':'224563'}
c.save(c.B/'manifest.json',v)
print(c.sha(c.B/'manifest.json'))
