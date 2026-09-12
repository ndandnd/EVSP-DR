# Warm chain4 k10 timeout

## Warm chain 4, k=10 — initialization timeout, job810344

Verified at 00:17 UTC on12September (20:17EDT11September): Slurm TIMEOUT after08:17:05 against08:15:00 allocation. Cached network loaded in8.1s (19,205nodes;109,583,108arcs). Persisted status remains initializing, zero iterations, final LP absent, inherited-pool audit null, and column journal empty. The status wall_s=9.18 is the initial publication time, not the completed job runtime. No LP certificate or new integer result exists.

This repeats the P1k7 initialization failure. The predecessor P4k9 pool survives but there is no usable child pool for the dependent MIP. No unchanged retry submitted; use the same bounded/checkpointed-initialization recovery requirement already recorded for P1. P2k10 remains running. Evidence: outputs/parallel_research_20260911/warm_p4_k10_timeout_810344/evidence.json.
