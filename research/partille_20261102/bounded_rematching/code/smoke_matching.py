from rematch import no_good_coefficients
import gurobipy as gp
from gurobipy import GRB
edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
with gp.Model() as m:
 m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.TimeLimit=5
 y=m.addVars(edges,vtype=GRB.BINARY)
 for u in range(4):
  m.addConstr(gp.quicksum(y[e] for e in edges if e[0]==u)<=1)
  m.addConstr(gp.quicksum(y[e] for e in edges if e[1]==u)<=1)
 m.addConstr(y.sum()==2)
 c,b=no_good_coefficients([1,2],edges);m.addConstr(gp.quicksum(v*y[e] for e,v in c.items())<=b)
 m.addConstr(y[0,1]==1);m.addConstr(y[1,2]==1)
 m.optimize();assert m.Status==GRB.OPTIMAL
 print('Boundary extension remains feasible under actual Gurobi no-good row.')
