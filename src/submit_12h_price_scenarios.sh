#!/usr/bin/env bash
set -euo pipefail

cd /home/nc437/demandresponse/src
sbatch submit_12h_price_scenarios.sub
