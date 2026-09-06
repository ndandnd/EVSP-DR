"""Fail-fast Gurobi license/import/optimization check for Slurm workers."""

from __future__ import annotations

import json

from master_lp_gurobi import gurobi_preflight


if __name__ == "__main__":
    print(json.dumps(gurobi_preflight(), sort_keys=True))
