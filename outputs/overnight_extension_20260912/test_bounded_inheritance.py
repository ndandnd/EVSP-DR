import sys,time,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import exact_pricer_expanded as mod

def quick(item):
    seq,stable,cost=item
    return ({'trips':seq,'cost':cost},stable,cost),None

def slow(item):
    time.sleep(2)
    return quick(item)

def setup(tmp_path,monkeypatch):
    p=tmp_path/'parent.json';p.write_text(json.dumps({'columns_journal':'pool.jsonl','trip_ids':[0,1,2,3],'csv':'parent.csv','provenance':{'instance_sha256':'sha'}}))
    records=[{'trips':[0],'cost':100001},{'trips':[0,1],'cost':100005},{'trips':[1,2,3],'cost':100010},{'trips':[2,3],'cost':100002}]
    monkeypatch.setattr(mod,'_file_sha256',lambda _: 'sha')
    monkeypatch.setattr(mod,'read_jsonl_records',lambda *a,**k:records)
    monkeypatch.setattr(mod,'load_column_pool',lambda *a,**k:{i:r for i,r in enumerate(records)})
    monkeypatch.setattr(mod,'_trip_id_maps',lambda _:({i:i for i in range(4)},{i:i for i in range(4)}))
    return p,dict(child_csv_path=tmp_path/'child.csv',child_problem=SimpleNamespace(trips={0,1,2,3}),child_network=None,g_kwh=240,charge_kw=240,reserve_kwh=0)

def test_default_retains_full_pool(tmp_path,monkeypatch):
    p,kw=setup(tmp_path,monkeypatch);monkeypatch.setattr(mod,'_replay_inherited_event_sequence',quick)
    routes,audit=mod.inherited_event_pool_records(p,**kw)
    assert len(routes)==4 and not audit['import_deadline_reached']
    assert not audit['inherited_lp_certificate']

def test_explicit_cap_is_deterministic(tmp_path,monkeypatch):
    p,kw=setup(tmp_path,monkeypatch);monkeypatch.setattr(mod,'_replay_inherited_event_sequence',quick)
    routes,audit=mod.inherited_event_pool_records(p,**kw,max_columns=2)
    assert [r['trips'] for r in routes]==[[1,2,3],[2,3]]
    assert audit['source_unique_columns']==4 and audit['selected_for_replay']==2

def test_deadline_terminates_worker_and_returns_uncertified_partial_import(tmp_path,monkeypatch):
    p,kw=setup(tmp_path,monkeypatch);monkeypatch.setattr(mod,'_replay_inherited_event_sequence',slow)
    t=time.monotonic();routes,audit=mod.inherited_event_pool_records(p,**kw,workers=1,time_limit_s=.1)
    assert time.monotonic()-t<1.5
    assert routes==[] and audit['import_deadline_reached']
    assert audit['unprocessed_selected']==4 and not audit['inherited_lp_certificate']
