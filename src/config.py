# ---------- Column pool controls ----------
n_fast_cols  = 2000         
n_exact_cols = 200        

# ---------- Horizon ----------
granularity = 1
bar_t = 24 * granularity
time_blocks = list(range(1, bar_t + 1))

# ---------- Numerics ----------
tolerance =  1e-6  

# ---------- Energy / charging scale ----------
BLOCK_KWH = 30.0
G_KWH     = 240.0
G         = int(round(G_KWH / BLOCK_KWH))
assert G > 0

charge_mult = 1
charge_cost_premium = 1 + 1e-2

# ---------- Costs ----------
BUS_COST_KX = 10000.0

# ---------- Misc ----------
factor = 1

# ---------- Real-data names / aliases ----------
DEPOT_NAME = "PARX"
CHARGING_STATIONS = ["2190L", "4808", "3127L", "7880C", "JON_A", "PARX"]

# Aliases for run script
SOC_CAPACITY_KWH     = G_KWH
ENERGY_PER_BLOCK_KWH = BLOCK_KWH
CHARGING_POWER_KW    = BLOCK_KWH
CHARGE_EFFICIENCY    = 1.0
BUS_COST_SCALAR      = BUS_COST_KX
ALLOW_ONLY_LISTED_DEADHEADS = True
MODE_EVS_ONLY        = True

# ---------- NEW: Column Generation controls ----------
# accept *any* improving column (don’t discard mild improvements)
RC_EPSILON =  1e-6         
K_BEST = 500               

# CG loop limits / guards
MAX_CG_ITERS = 300
STAGNATION_ITERS = 20     
MASTER_IMPROVE_THRESHOLD = 1e-8  # was 5e-4 (count small progress as improvement)

# ---------- NEW: Solver resource knobs ----------
THREADS = 8
NODEFILE_START = 0.2
NODEFILE_DIR = None

# ---------- NEW: Timelimits/gaps ----------
# Master RMP (LP) per-iteration solve
MASTER_TIMELIMIT = 300   # was 60 (still modest, gives RMP time to settle)
MASTER_MIPGAP   = 1.0    # final MIP target gap (keep)

# Pricing MIPs
PRICING_TIMELIMIT = 180    # was 60; pricing is your bottleneck
PRICING_GAP       = 1.0   # keep
