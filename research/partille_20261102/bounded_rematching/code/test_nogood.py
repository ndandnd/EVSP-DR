"""Exhaustively verify whole-component no-good on a four-node acyclic graph."""
import itertools
from rematch import no_good_coefficients,paths_from_matching
edges=list(itertools.combinations(range(4),2));tests=0
for size in range(1,5):
 for path in itertools.combinations(range(4),size):
  coeff,rhs=no_good_coefficients(path,edges)
  for bits in itertools.product([0,1],repeat=len(edges)):
   selected=[e for e,b in zip(edges,bits) if b]
   if len({u for u,v in selected})!=len(selected) or len({v for u,v in selected})!=len(selected):continue
   components=paths_from_matching(list(range(4)),selected)
   forbidden=list(path) in components
   violated=sum(coeff.get(e,0) for e in selected)>rhs
   assert violated==forbidden,(path,selected,coeff,rhs)
   tests+=1
print('Exact whole-path/source/sink no-good exhaustive checks:',tests)
