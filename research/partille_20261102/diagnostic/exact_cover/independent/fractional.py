"""Independent tolerance-checked fractional feasibility of the ALL-column relaxation."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from verify import read_inputs,sha,trip_bits

p=argparse.ArgumentParser(description=__doc__)
for k in ('instance','journal','summary','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args(); report,masks,_,n,fleet=read_inputs(a.instance,a.journal,a.summary)
ii=[];jj=[]
for j,mask in enumerate(masks):
    ii.extend(t.bit_length()-1 for t in trip_bits(mask));ii.append(n)
    jj.extend([j]*(mask.bit_count()+1))
A=coo_matrix((np.ones(len(ii)),(ii,jj)),shape=(n+1,len(masks))).tocsc()
b=np.ones(n+1);b[-1]=fleet
sol=linprog(np.zeros(len(masks)),A_eq=A,b_eq=b, bounds=(0,1),method='highs',options={'time_limit':30.})
if not sol.success: raise RuntimeError(sol.message)
# Rebuild constraints from original trip masks (not from A or the solver residuals).
service=np.zeros(n); total=0.
for mask,x in zip(masks,sol.x):
    if x==0.: continue
    total+=x
    for bit in trip_bits(mask):service[bit.bit_length()-1]+=x
trip_err=float(np.max(np.abs(service-1.)));fleet_err=float(abs(total-fleet));bound_err=float(max(0.,-min(sol.x),max(sol.x)-1.))
if max(trip_err,fleet_err,bound_err)>1e-8: raise RuntimeError('relaxation feasibility tolerance exceeded')
report.update(method='HiGHS LP on all trip-set projections with 0<=x<=1, trip coverage=1 and fleet=8',
              scipy_version=scipy.__version__, solver_message=sol.message,
              max_trip_equality_residual=trip_err, fleet_equality_residual=fleet_err,
              max_bound_violation=bound_err,interior_variables=int(np.count_nonzero((sol.x>1e-8)&(sol.x<1.-1e-8))),
              verification_tolerance=1e-8,verification_source_sha256=sha(Path(__file__).read_bytes()),
              input_adapter_sha256=sha(Path(__file__).with_name('verify.py').read_bytes()))
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
