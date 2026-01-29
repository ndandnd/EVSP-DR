# ---------- Column pool controls ----------
n_fast_cols  = 200         
n_exact_cols = 10        

# ---------- Horizon ----------
TIMEBLOCKS_PER_HOUR = 60
bar_t = 26 * TIMEBLOCKS_PER_HOUR
time_blocks = list(range(1, bar_t + 1))

# ---------- Numerics ----------
tolerance = 1e-4

# ---------- Energy / charging scale ----------
BLOCK_KWH = 30.0
G_KWH     = 270.0
ENERGY_PER_BLOCK_KWH = BLOCK_KWH / TIMEBLOCKS_PER_HOUR
G         = int(round(G_KWH / BLOCK_KWH))
assert G > 0

charge_mult = 40
charge_cost_premium = 1 + 1e-2

# ---------- Costs ----------
BUS_COST_KX = 1e3

# ---------- Misc ----------
factor = 1

# ---------- Real-data names / aliases ----------
# IMPORTANT: Update constants to match Toy Model
# DEPOT_NAME = "DEPOT"
# CHARGING_STATIONS = ["DEPOT", "CHARGER1"]

# Only duplicate the charger if you want to test queueing/symmetry
# STATION_COPIES = {
#     "DEPOT": 2,
#     "CHARGER1": 2
# }

DEPOT_NAME = "PARX"
CHARGING_STATIONS = ["2190L", "4808", "3127L", "7880C", "JON_A", "PARX"]

STATION_COPIES = {
    "2190L": 3,
    "4808":  3,
    "PARX":  3,  
    "3127L": 3,
    "7880C": 3,
    "JON_A": 3
}





MODE_EVS_ONLY        = True
BIG_M_PENALTY       = 1e5

# ---------- NEW: Column Generation controls ----------
# accept *any* improving column (don’t discard mild improvements)
RC_EPSILON = 5.0           # was 300.0 (kept out many mildly negative routes)
K_BEST = 50               # was 30 (add more useful columns per iter)

# CG loop limits / guards
MAX_CG_ITERS = 300
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
