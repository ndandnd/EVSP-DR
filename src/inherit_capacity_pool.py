"""Hash-gated previous-instance columns for nested constant-physics experiments."""
from __future__ import annotations
import copy,csv,hashlib,json,math
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def identity(row):return int(float(row['Ordered_Trip_ID']))
def trip_mapping(parent_csv,child_csv):
    with open(parent_csv) as handle: parent=list(csv.DictReader(handle))
    with open(child_csv) as handle: child=list(csv.DictReader(handle))
    ids=[identity(r) for r in child]
    if len(ids)!=len(set(ids)):raise ValueError('child repeats source trip IDs')
    lookup={identity(r):(i,r) for i,r in enumerate(child)};mapping={}
    for i,row in enumerate(parent):
        key=identity(row)
        if key not in lookup:raise ValueError('parent trip absent in child')
        j,new=lookup[key]
        for field in ('From1','Start1','End1','To1','Usage kWh'):
            same=(math.isclose(float(row[field]),float(new[field]),abs_tol=1e-8,rel_tol=1e-10) if field=='Usage kWh' else row[field]==new[field])
            if not same:raise ValueError(f'changed trip {key}: {field}')
        mapping[i]=j
    if len({identity(r) for r in parent})!=len(parent):raise ValueError('parent repeats source trip IDs')
    return mapping

def remap_route(route,mapping,new_checkpoint_id):
    out=copy.deepcopy(route)
    out['trips']=[mapping[t] for t in route['trips']]
    out['route_nodes']=[mapping[t] if isinstance(t,int) else t for t in route['route_nodes']]
    out['inheritance']={'parent_checkpoint_id':route.get('cg_checkpoint_id'),'parent_route_sha256':canonical(route),'parent_found_iter':route.get('found_iter'),'local_trip_ids_remapped':True}
    out['cg_checkpoint_id']=new_checkpoint_id;out['found_iter']=0;out['origin']='nested_previous_instance_pool'
    realization=out.get('continuous_realization')
    if realization:
        realization['trip_sequence_sha256']=canonical(out['trips'])
        realization['route_nodes_sha256']=canonical(out['route_nodes'])
        realization['mapping_sha256']=canonical({k:v for k,v in realization.items() if k not in ('mapping_sha256','trace')})
        if out.get('physical_realization'):
            out['physical_realization']['realization_mapping_sha256']=realization['mapping_sha256']
    return out

def inherit_pool(parent_status,parent_pool,parent_csv,child_csv,*,expected_physics,child_provenance,new_checkpoint_id,route_validator):
    status=json.loads(Path(parent_status).read_text());prov=status['provenance']
    if status['pool_sha256']!=sha(parent_pool):raise ValueError('parent pool hash mismatch')
    if prov['instance_sha256']!=sha(parent_csv):raise ValueError('parent instance hash mismatch')
    for key in ('git_commit','prices_sha256','reference_sha256','deadhead_sha256'):
        if prov[key]!=child_provenance[key]:raise ValueError(f'parent model/input provenance mismatch: {key}')
    for key,value in expected_physics.items():
        if status['physics'].get(key)!=value:raise ValueError(f'parent physics mismatch: {key}')
    mapping=trip_mapping(parent_csv,child_csv)
    routes=[]
    for line in Path(parent_pool).read_text().splitlines():
        if not line.strip():continue
        old=json.loads(line)
        if old.get('cg_checkpoint_id')!=status['checkpoint']['id']:raise ValueError('parent route checkpoint mismatch')
        if not math.isfinite(float(old['cost'])):raise ValueError('nonfinite inherited route cost')
        new=remap_route(old,mapping,new_checkpoint_id)
        reason=route_validator(new)
        if reason is not None:raise ValueError(f'inherited route physical replay failed: {reason}')
        routes.append(new)
    return routes,{'parent_status':str(parent_status),'parent_status_sha256':sha(parent_status),'parent_pool':str(parent_pool),'parent_pool_sha256':sha(parent_pool),'parent_instance_sha256':sha(parent_csv),'inherited_columns':len(routes),'trip_mapping_sha256':canonical(mapping),'every_inherited_route_replayed':True}
