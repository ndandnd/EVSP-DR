"""Read-only completion reconciliation; workers perform full hash/gzip checks."""
import csv
import datetime
import hashlib
import json
import pathlib
import sys

root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/home/nc437/ladder-lite/storage_cleanup_20260912_archive')
manifest=json.loads((root/'manifest.json').read_text())
expected={f['path']: f for f in manifest['files']}
records={}
status_hashes={}
errors=[]
failures=[]
for p in sorted(root.glob('status_*.tsv')):
    data=p.read_bytes()
    status_hashes[p.name]=hashlib.sha256(data).hexdigest()
    for row in csv.DictReader(data.decode().splitlines(),delimiter='\t'):
        source=row.get('source_path')
        if source not in expected:
            errors.append('unknown source in status: '+str(source));continue
        if row.get('phase')=='failed': failures.append(row)
        if row.get('phase') in ('prepared','success'): records[source]=row
complete=[]
original_bytes=archive_bytes=0
for source,row in records.items():
    if row.get('phase')!='success':continue
    f=expected[source]
    arc=pathlib.Path(row.get('archive_path',''))
    if str(arc)!=f['archive_path']:
        errors.append('archive mapping mismatch: '+source);continue
    if int(row['source_size_bytes'])!=f['size']:
        errors.append('source size mismatch: '+source);continue
    if len(row.get('source_sha256',''))!=64 or len(row.get('archive_sha256',''))!=64:
        errors.append('missing full hashes: '+source);continue
    if pathlib.Path(source).exists() or pathlib.Path(source).is_symlink():
        errors.append('source still exists after success: '+source);continue
    if not arc.is_file() or arc.is_symlink() or arc.stat().st_size!=int(row['archive_size_bytes']):
        errors.append('archive missing or size mismatch: '+source);continue
    complete.append(row)
    original_bytes+=f['size'];archive_bytes+=arc.stat().st_size
summary={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
         'status_sha256':status_hashes,'expected_files':len(expected),
         'completed_files':len(complete),'completed_source_bytes':original_bytes,
         'archive_bytes':archive_bytes,'saved_bytes':original_bytes-archive_bytes,
         'errors':errors,'failed_events':failures,
         'all_complete':len(complete)==len(expected) and not errors,
         'verification':'Worker verified gzip integrity and decompressed SHA before unlink; collector verifies manifest mappings, recorded hashes, archive sizes and source absence.',
         'records':complete}
print(json.dumps(summary,indent=2))
