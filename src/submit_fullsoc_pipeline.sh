#!/usr/bin/env bash
set -euo pipefail

cd /home/nc437/demandresponse/src
mkdir -p logs

export PATH=/usr/local/slurm/current/bin:$PATH

colgen_job=$(sbatch --parsable submit_12h_price_scenarios.sub)
echo "Submitted 12h full-SOC column generation array: ${colgen_job}"

mip_job=$(sbatch --parsable --dependency=afterany:${colgen_job} submit_mip_40m_fullsoc.sub)
echo "Submitted dependent 40min scaglione MIP array: ${mip_job}"

echo
echo "Monitor with:"
echo "  squeue --me -n CG12HFS,MIP40FS"
echo
echo "The MIP array will start after the 12h column-generation array finishes."
