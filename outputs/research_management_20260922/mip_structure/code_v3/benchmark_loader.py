"""Once-only decoder benchmark on the authenticated largest frozen matrix."""
from pathlib import Path
import sys,json,time,resource
import numpy as np
from core import unpack_incidence,digest
from runner import sha,write
root=Path(sys.argv[1]);gate=json.loads((root/'prepared/c1_k15_sequential.json').read_text());assert sha(gate['complete'])==gate['sha256'];complete=json.loads(Path(gate['complete']).read_text());assert sha(complete['matrix_file'])==complete['matrix_file_sha256'];assert sha(complete['matrix_metadata'])==complete['matrix_metadata_sha256'];meta=json.loads(Path(complete['matrix_metadata']).read_text())
class Counted:
 def __init__(self,z):self.z=z;self.counts={}
 def __getitem__(self,key):self.counts[key]=self.counts.get(key,0)+1;return self.z[key]
t=time.monotonic()
with np.load(complete['matrix_file'],allow_pickle=False) as z:
 proxy=Counted(z);columns,costs=unpack_incidence(proxy)
elapsed=time.monotonic()-t;assert proxy.counts=={'indptr':1,'indices':1,'costs':1};t=time.monotonic();identity=digest({'trips':meta['trips'],'columns':columns,'costs':costs,'route_hashes':meta['route_hashes']});assert identity==meta['matrix_identity_sha256']
write(root/'loader_benchmark_v3.json',{'passed':True,'decode_wall_s':elapsed,'identity_check_wall_s':time.monotonic()-t,'member_fetches':proxy.counts,'rows':meta['rows'],'columns':len(columns),'nonzeros':sum(map(len,columns)),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'matrix_identity_sha256':identity,'matrix_file_sha256':sha(complete['matrix_file']),'source_matrix':complete['matrix_file']})
