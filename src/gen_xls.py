import pandas as pd
import itertools
import os
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")  # Ensure this points to your data folder
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Define Locations
# Structure: Depot at 0, Loc1 at 30, Loc2 at 70. 
# Charger1 is near Loc1 (5 mins away).
LOCATIONS = ["DEPOT", "LOC_1", "LOC_2", "CHARGER_1"]

# Manually define travel times (Duration in minutes)
# We assume the graph is symmetric for simplicity.
# Energy is approximated as 0.5 kWh per minute of driving.
DISTANCES_MIN = {
    ("DEPOT", "LOC_1"): 30,
    ("DEPOT", "LOC_2"): 60,
    ("DEPOT", "CHARGER_1"): 35, # Slightly longer than to Loc1
    
    ("LOC_1", "LOC_2"): 45,     # The main route length
    ("LOC_1", "CHARGER_1"): 5,  # Very close! Good for mid-day topup.
    
    ("LOC_2", "CHARGER_1"): 50, # Loc2 -> Loc1 -> Charger1
}

# --- GENERATE DEADHEAD MATRIX (Fully Connected) ---
dhd_rows = []

# Generate all permutations (A->B, B->A)
for u, v in itertools.permutations(LOCATIONS, 2):
    # Lookup distance (checking both directions)
    dist = DISTANCES_MIN.get((u, v))
    if dist is None:
        dist = DISTANCES_MIN.get((v, u))
    
    if dist is not None:
        dhd_rows.append({
            "From": u,
            "To": v,
            "Duration": dist,
            "Energy used": dist * 0.5  # Simple assumption: 0.5 kWh/min
        })

df_dhd = pd.DataFrame(dhd_rows)
# Saving as CSV because your run_experiments.py expects .csv
csv_path = DATA_DIR / "Toy_DHD.csv"
df_dhd.to_csv(csv_path, index=False)
print(f"[CREATED] {csv_path} with {len(df_dhd)} arcs.")


# --- GENERATE TIMETABLED TRIPS ---
# Pattern:
# Even Hours (6, 8, ...): LOC_1 -> LOC_2
# Odd Hours  (7, 9, ...): LOC_2 -> LOC_1
# This creates a perfect "Chain" opportunity. A single bus can ping-pong all day.
trips = []
start_hour = 6
end_hour = 20 # 8 PM

for h in range(start_hour, end_hour + 1):
    trip_start_time = f"{h:02d}:00"
    trip_end_time   = f"{h:02d}:45" # 45 min trip duration
    
    if h % 2 == 0:
        # Even: Outbound
        sl, el = "LOC_1", "LOC_2"
    else:
        # Odd: Inbound
        sl, el = "LOC_2", "LOC_1"
    
    trips.append({
        "SL": sl,
        "EL": el,
        "ST": trip_start_time,
        "ET": trip_end_time,
        "Energy used": 15.0 # Fixed energy per trip
    })

df_trips = pd.DataFrame(trips)
routes_path = DATA_DIR / "Toy_Routes.csv"
df_trips.to_csv(routes_path, index=False)
print(f"[CREATED] {routes_path} with {len(df_trips)} trips.")

# --- GENERATE PRICES (Flat Rate) ---
# Just to ensure the code doesn't crash on missing price file
prices = [{"time_block": t, "cost": 1.0} for t in range(1, 1000)] # Enough for any resolution
df_prices = pd.DataFrame(prices)
df_prices.to_csv(DATA_DIR / "Toy_Prices.csv", index=False)
print(f"[CREATED] {DATA_DIR / 'Toy_Prices.csv'}")