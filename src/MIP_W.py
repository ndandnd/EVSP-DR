#%%
# mip_experiment_A.py
import gurobipy as gp
import json
import time

# 1. Load R_truck for post-processing later
print("Loading R_truck...")
with open('R_truck_export_1fulll.json', 'r') as f:
    R_truck = json.load(f)

# 2. Initialize Gurobi and load the pre-built model
env = gp.Env(empty=True)
env.setParam('LogFile', 'mip_run_full.log')
env.start()

print("Loading mathematical model...")
m = gp.read("evsp_master.mps", env=env)


# ==========================================
# Config W: Winning reference + binary fix + modest heuristic boost
# Matches the 2h reference that found the best incumbent on this pool,
# fixes the binary bug, and gives RINS/ImproveStart extra traction.
# ==========================================
for v in m.getVars():
    if v.VarName.startswith("a["):
        v.VType = gp.GRB.BINARY
m.update()

m.setParam('TimeLimit', 60*60)
m.setParam('Threads', 3)              # match reference; run ONLY this one
m.setParam('MIPFocus', 1)             # incumbent-focused, not bound
m.setParam('Heuristics', 0.5)         # match reference
# m.setParam('Cuts', 1)                 # default-level, not aggressive
m.setParam('RINS', 15)                # neighborhood search every 15 nodes
m.setParam('ImproveStartTime', 1200)  # give it 20 min to explore, then go all-in on incumbent
m.setParam('ImproveStartGap', 0.30)




# 4. Optimize
start_time = time.time()
m.optimize()
elapsed = time.time() - start_time

# 5. Extract results using your R_truck data
if m.SolCount > 0:
    print(f"\nOptimization finished in {elapsed:.1f}s")
    print(f"Best Objective: {m.ObjVal}")
    print(f"Lower Bound: {m.ObjBound}")
    print(f"Gap: {m.MIPGap * 100:.2f}%")
    
    # You can now use R_truck to decode the solution exactly like your notebook does
    # extract_route_from_solution(m, R_truck)
else:
    print("No integer solution found.")
# %%
#%%
# Continue Config W2 for another 40 minutes
m.setParam('TimeLimit', 60*45)

# Optional: tweak params mid-run based on what you saw
# m.setParam('Heuristics', 0.5)  # dial back if it was chewing too much time on heur
# m.setParam('ImproveStartTime', 0)  # force improvement mode immediately

start_time = time.time()
m.optimize()
elapsed = time.time() - start_time

if m.SolCount > 0:
    print(f"\nContinued run finished in {elapsed:.1f}s")
    print(f"Best Objective: {m.ObjVal}")
    print(f"Lower Bound: {m.ObjBound}")
    print(f"Gap: {m.MIPGap * 100:.2f}%")
# %%
