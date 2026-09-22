"""Exact combinatorial diagnostics and conservative integer dual certificates."""
import hashlib,json,math,time
from fractions import Fraction


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_start(columns, m, selected, cap=None):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('empty or repeated start indices')
    if any(i < 0 or i >= len(columns) for i in selected):
        raise ValueError('start index outside pool')
    covered=set(t for i in selected for t in columns[i])
    if covered != set(range(m)) or (cap is not None and len(selected)>cap):
        raise ValueError('start does not cover rows or exceeds cap')
    return True


def structure(columns,costs,m,max_seconds=600):
    """Bitset indexed safe dominance for nonnegative-cost binary covering.

    No reductions are applied. Row implication holds for both objectives. Column
    witnesses respect both unit fleet cost and nonnegative charging cost, plus
    an optional at-most fleet cap; no other side constraints are covered.
    """
    started=time.monotonic();n=len(columns)
    if len(costs)!=n or any(not math.isfinite(c) or c<0 for c in costs):
        raise ValueError('safe replacement requires finite nonnegative costs')
    bits=[0]*m;counts=[0]*m;parent=list(range(m));seen={};duplicates=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j,col in enumerate(columns):
        if not col or len(set(col))!=len(col) or any(t<0 or t>=m for t in col):raise ValueError('invalid incidence')
        key=tuple(sorted(col))
        if key in seen:duplicates.append([j,seen[key]])
        else:seen[key]=j
        a=root(col[0])
        for t in col:
            bits[t]|=1<<j;counts[t]+=1;b=root(t)
            if a!=b:parent[b]=a
    if not all(counts):raise ValueError('uncovered row')
    row_witness=[]
    for j in range(m):
        for i in range(m):
            if i==j:continue
            if bits[i]&~bits[j]==0 and (bits[i]!=bits[j] or i<j):
                row_witness.append([j,i]);break
    # Equal-cost strict supersets sort before subsets, preventing cycles.
    order=sorted(range(n),key=lambda j:(costs[j],-len(columns[j]),j));eligible=0;witness=[];processed=0;intersections=0
    for j in order:
        if processed%128==0 and time.monotonic()-started>max_seconds:break
        cand=eligible
        for t in sorted(columns[j],key=lambda t:counts[t]):
            cand &= bits[t];intersections+=1
            if not cand:break
        if cand:
            k=(cand & -cand).bit_length()-1
            assert set(columns[j])<=set(columns[k]) and costs[k]<=costs[j]
            witness.append([j,k])
        eligible|=1<<j;processed+=1
    comps={}
    for t in range(m):comps.setdefault(root(t),{'rows':0,'columns':0})['rows']+=1
    for col in columns:comps[root(col[0])]['columns']+=1
    return {'rows':m,'columns':n,'nonzeros':sum(map(len,columns)),
            'identical_incidence_pairs':duplicates,'redundant_row_witnesses':row_witness,
            'safe_cost_respecting_column_witnesses':witness,'dominance_complete':processed==n,
            'dominance_processed_columns':processed,'dominance_candidate_intersections':intersections,
            'components':list(comps.values()),'wall_s':time.monotonic()-started,
            'scope':'nonnegative charging cost; binary covering; optional at-most fleet cap; no capacity/other side rows'}


def floor_scaled(x,scale):
    q=Fraction.from_float(float(x))*scale
    return q.numerator//q.denominator


def ceil_scaled(x,scale):
    q=Fraction.from_float(float(x))*scale
    return -((-q.numerator)//q.denominator)


def dual_certificate(columns,costs,y,mu,cap,incumbent,scale=10**8):
    """Integer arithmetic certificate for Ax>=1, 0<=x<=1, sum x<=cap.

    Solver duals are only candidate multipliers. Clipping enforces correct signs;
    quantization and downward objective coefficients give exact rational lower
    bounds even when solver duals have residual violations. The min(0,q) term
    includes upper-bound contributions often omitted in reduced-cost screening.
    All comparisons are strict and preserve every solution matching the UB.
    """
    m=len(y);validate_start(columns,m,incumbent,cap)
    if not all(math.isfinite(float(z)) for z in [*costs,*y,mu]):raise ValueError('nonfinite')
    yi=[max(0,floor_scaled(z,scale)) for z in y]
    mui=min(0,floor_scaled(mu,scale)) if cap is not None else 0
    ci=[floor_scaled(z,scale) for z in costs]
    q=[ci[j]-sum(yi[t] for t in col)-mui for j,col in enumerate(columns)]
    lower=sum(yi)+(cap*mui if cap is not None else 0)+sum(min(0,z) for z in q)
    upper=sum(ceil_scaled(costs[j],scale) for j in incumbent)
    force1=[lower+max(0,z) for z in q];force0=[lower+max(0,-z) for z in q]
    zero=[j for j,v in enumerate(force1) if v>upper]
    one=[j for j,v in enumerate(force0) if v>upper]
    assert lower<=upper
    assert not set(zero)&set(incumbent)
    assert set(one)<=set(incumbent)
    return {'scale':scale,'row_duals_integer':yi,'fleet_cap_dual_integer':mui,
            'cost_floor_integer':ci,'reduced_cost_integer':q,'lower_bound_integer':lower,
            'incumbent_upper_bound_integer':upper,'lower_bound':lower/scale,'incumbent_upper_bound':upper/scale,
            'forced_one_lower_bounds_integer':force1,'forced_zero_lower_bounds_integer':force0,
            'safe_fix_zero_indices':zero,'safe_fix_one_indices':one,
            'removable_fraction':len(zero)/len(columns),'incumbent_indices':incumbent,
            'certificate_formula':'L=sum(y)+K*mu+sum(min(0,c_floor-A^T*y-mu)); LB(xj=1)=L+max(0,qj); LB(xj=0)=L+max(0,-qj)',
            'scope':'Exact integer arithmetic for rounded-down objective and signed rational multipliers; 0<=x<=1 bound terms included; screening diagnostic only'}
