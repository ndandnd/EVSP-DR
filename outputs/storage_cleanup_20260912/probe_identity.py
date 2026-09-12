import json,os,pathlib,socket
root=pathlib.Path('/home/nc437/ladder-lite/storage_cleanup_20260912_archive')
manifest=json.loads((root/'manifest.json').read_text())
differences=[]
for f in manifest['files']:
    s=os.lstat(f['path'])
    actual={'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,
            'device':s.st_dev,'nlink':s.st_nlink,'mode':s.st_mode}
    d={k:{'expected':f[k],'actual':v} for k,v in actual.items() if f[k]!=v}
    if d: differences.append({'path':f['path'],'differences':d})
print(json.dumps({'host':socket.gethostname(),'differences':differences},indent=2))
