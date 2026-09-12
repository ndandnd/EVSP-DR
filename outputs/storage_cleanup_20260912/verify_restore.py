"""Verify a small real archived journal without changing its original path."""
import csv
import datetime
import hashlib
import json
import pathlib
import subprocess
import tempfile

root=pathlib.Path('/home/nc437/ladder-lite/storage_cleanup_20260912_archive')
rows=[]
for p in root.glob('status_*.tsv'):
    rows.extend(r for r in csv.DictReader(p.open(),delimiter='\t')
                if r['phase']=='success' and r['source_path'].endswith('.columns.jsonl'))
row=min(rows,key=lambda r:int(r['source_size_bytes']))
assert int(row['source_size_bytes']) < 20_000_000, 'Wait for a small journal before testing on login node'
arc=pathlib.Path(row['archive_path'])
archive_hash=hashlib.sha256(arc.read_bytes()).hexdigest()
assert archive_hash==row['archive_sha256']
with tempfile.TemporaryDirectory(prefix='evsp-real-restore-check-') as tmp:
    dest=pathlib.Path(tmp)/'restored.columns.jsonl'
    command=['bash',str(root/'restore_archive.sh'),str(arc),str(dest),row['source_sha256']]
    subprocess.run(command,check=True,capture_output=True,text=True)
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    assert digest==row['source_sha256']
    assert dest.stat().st_size==int(row['source_size_bytes'])
    assert subprocess.run(command,capture_output=True,text=True).returncode!=0
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source_path':row['source_path'],'archive_path':str(arc),
                  'source_sha256':digest,'archive_sha256':archive_hash,
                  'restored_bytes':int(row['source_size_bytes']),
                  'real_archive_restoration_passed':True,
                  'existing_destination_rejected':True,
                  'temporary_copy_removed':True},indent=2))
