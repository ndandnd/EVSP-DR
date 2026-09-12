import hashlib,json,multiprocessing,signal,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/private/tmp/evsp-algorithm-review-source-20260912/src')
import exact_pricer_expanded as exact

def handler(signum,frame): pass

def reset_worker_signals():
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    signal.signal(signal.SIGINT,signal.SIG_DFL)
    signal.signal(signal.SIGUSR1,signal.SIG_DFL)

class SlowNetwork:
    def fixed_sequence_record(self,trips):
        print(json.dumps({'worker_started':True,'term_handler_inherited':signal.getsignal(signal.SIGTERM)==handler}),flush=True)
        time.sleep(30)
        return None

if __name__=='__main__':
    mode=sys.argv[1]
    signal.signal(signal.SIGTERM,signal.SIG_DFL if mode=='default' else handler)
    if mode=='initializer_reset':
        context=multiprocessing.get_context('fork')
        class Proxy:
            def Pool(self,workers):return context.Pool(workers,initializer=reset_worker_signals)
        exact.multiprocessing.get_context=lambda method:Proxy()
    with tempfile.TemporaryDirectory(prefix='evsp-import-probe-') as td:
        root=Path(td);csv=root/'input.csv';csv.write_text('count_trip_id,Ordered_Trip_ID\n0,42\n')
        journal=root/'pool.jsonl';journal.write_text(json.dumps({'trips':[0],'cost':100000})+'\n')
        status=root/'status.json';status.write_text(json.dumps({'columns_journal':str(journal),'trip_ids':[0],'csv':str(csv),'provenance':{'instance_sha256':hashlib.sha256(csv.read_bytes()).hexdigest()}}))
        start=time.monotonic()
        records,audit=exact.inherited_event_pool_records(status,child_csv_path=csv,child_problem=SimpleNamespace(trips=[0]),child_network=SlowNetwork(),g_kwh=240,charge_kw=240,reserve_kwh=0,workers=1,max_columns=1,time_limit_s=.2)
        print(json.dumps({'returned':True,'elapsed':time.monotonic()-start,'audit':audit}),flush=True)
