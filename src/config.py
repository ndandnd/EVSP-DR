# ---------- Column pool controls ----------
n_fast_cols  = 200         
n_exact_cols = 10        

# ---------- Horizon ----------
bar_t = 24
time_blocks = list(range(1, bar_t + 1))

# ---------- Numerics ----------
tolerance = 1e-4

# ---------- Energy / charging scale ----------
BLOCK_KWH = 30.0
G_KWH     = 240.0
G         = int(round(G_KWH / BLOCK_KWH))
assert G > 0

charge_mult = 1
charge_cost_premium = 1 + 1e-2

# ---------- Costs ----------
BUS_COST_KX = 20.0

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
RC_EPSILON = 5.0           # was 300.0 (kept out many mildly negative routes)
K_BEST = 50               # was 30 (add more useful columns per iter)

# CG loop limits / guards
MAX_CG_ITERS = 80
STAGNATION_ITERS = 5      # was 3 (don’t stop so early)
MASTER_IMPROVE_THRESHOLD = 5e-4  # was 5e-4 (count small progress as improvement)

# ---------- NEW: Solver resource knobs ----------
THREADS = 8
NODEFILE_START = 0.2
NODEFILE_DIR = None

# ---------- NEW: Timelimits/gaps ----------
# Master RMP (LP) per-iteration solve
MASTER_TIMELIMIT = 60     # was 60 (still modest, gives RMP time to settle)
MASTER_MIPGAP   = 0.02     # final MIP target gap (keep)

# Pricing MIPs
PRICING_TIMELIMIT = 60    # was 60; pricing is your bottleneck
PRICING_GAP       = 0.05   # keep
