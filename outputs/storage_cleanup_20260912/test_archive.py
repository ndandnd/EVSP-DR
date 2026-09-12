"""Small destructive-file fixtures in a temporary directory, never live data."""
import gzip
import hashlib
import pathlib
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = (HERE/'archive.sbatch').read_text()

def fixture(mode):
    with tempfile.TemporaryDirectory(prefix='evsp-archive-fixture-') as tmp:
        base = pathlib.Path(tmp)
        src = base/'ladder-lite/phys240kw/test.columns.jsonl'
        arc = base/'archive/phys240kw/test.columns.jsonl.gz'
        src.parent.mkdir(parents=True)
        arc.parent.mkdir(parents=True)
        payload = b'{"trips":[1,2],"cost":123.4}\n'*1000
        src.write_bytes(payload)
        original = hashlib.sha256(payload).hexdigest()
        stat = src.stat()
        status = base/'archive/status.tsv'
        status.write_text('source_path\tphase\tsource_sha256\tsource_size_bytes\tarchive_path\tarchive_sha256\tarchive_size_bytes\tevent_utc\n')
        functions = SOURCE[SOURCE.index('fsync_file()'):SOURCE.index('if [ ! -s "$STATUS"')]
        functions += SOURCE[SOURCE.index('append_event()'):SOURCE.index('\nsuccess_count=0')]
        functions = functions.replace('/home/nc437/ladder-lite/', str(base/'ladder-lite')+'/')
        if mode in ('prepared_present','prepared_absent','changed_after_prepared','bad_mapping'):
            arc.write_bytes(gzip.compress(payload,compresslevel=1))
            arc_sha=hashlib.sha256(arc.read_bytes()).hexdigest()
            recorded = str(arc) if mode!='bad_mapping' else str(arc)+'.wrong'
            with status.open('a') as f:
                f.write(f'{src}\tprepared\t{original}\t{len(payload)}\t{recorded}\t{arc_sha}\t{arc.stat().st_size}\tfixture\n')
            if mode=='prepared_absent': src.unlink()
            if mode=='changed_after_prepared': src.write_bytes(b'x'*len(payload))
        elif mode=='symlink':
            real=src.with_suffix('.real');src.rename(real);src.symlink_to(real)
        elif mode=='hardlink':
            import os
            os.link(src,src.with_suffix('.other'))
        elif mode=='orphan_archive':
            arc.write_bytes(gzip.compress(payload))
        elif mode=='record_failure':
            functions += '\nappend_event() { return 1; }\n'
        bash=f'''set -u -o pipefail
ROOT='{base}/archive'
STATUS='{status}'
SLURM_JOB_ID=fixture
{functions}
archive_one '{src}' '{stat.st_size}' '{int(stat.st_mtime)}' '{arc}'
'''
        result=subprocess.run(['bash','-c',bash],capture_output=True,text=True)
        success=mode in ('normal','prepared_present','prepared_absent')
        if success:
            assert result.returncode==0,(mode,result.stderr)
            assert not src.exists()
            assert hashlib.sha256(gzip.decompress(arc.read_bytes())).hexdigest()==original
            restored=base/'restored.columns.jsonl'
            rr=subprocess.run(['bash',str(HERE/'restore_archive.sh'),str(arc),str(restored),original],capture_output=True,text=True)
            assert rr.returncode==0,(mode,rr.stderr)
            assert restored.read_bytes()==payload
            assert subprocess.run(['bash',str(HERE/'restore_archive.sh'),str(arc),str(restored),original],capture_output=True).returncode!=0
        else:
            assert result.returncode!=0,(mode,result.stdout,result.stderr)
            assert src.exists(),mode
            if mode!='changed_after_prepared': assert src.read_bytes()==payload,mode
        print(mode,'PASS')

if __name__=='__main__':
    for mode in ('normal','prepared_present','prepared_absent','changed_after_prepared',
                 'bad_mapping','symlink','hardlink','orphan_archive','record_failure'):
        fixture(mode)
