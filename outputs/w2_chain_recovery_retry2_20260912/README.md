# Launcher retry with verified Gurobi preflight

Original37532 and37533 failed before Python/optimization because wrapper omitted PYTHON_BIN required by gurobi_worker_preflight.sh.37534 remained dependency-blocked. Source artifacts and pending original job preserved.

Corrected wrapper exports PYTHON_BIN before preflight. Exact shell environment and preflight executed successfully; evidence preflight_verified.json. All solver sources, settings and budgets unchanged. Outputs and Gurobi logs use a new isolated root; old attempt paths preserved. Generic worker/entrypoint remains the reviewed implementation.
