# Expanded-network charging realization audit

The audit is read-only. Trip order, station choice, and charging windows are preserved.

## Root cause

The expanded network floors SOC to its lattice at transitions. The old extractor emitted the full lattice charging gain, while continuous replay retained discarded residual SOC. Repeated residuals therefore made recorded schedules overfill the battery.

## Cost and certificate conclusion

Existing certified_rc_optimal applies only to the conservative expanded-grid cost model. Continuous realized costs are not exact-priced and have no global reduced-cost certificate.

| Pool | Journal rows | Admitted unique | Valid recorded | Repairable | Infeasible |
|---|---:|---:|---:|---:|---:|
| k40_r1_ca_raw_m1440 | 42237 | 42237 | 621 | 41616 | 0 |
| k40_r1_cs_raw_m1440 | 47687 | 47687 | 1956 | 45731 | 0 |
| k40_r2_ca_raw_m1440 | 47307 | 47307 | 673 | 46634 | 0 |
| k40_r2_cs_raw_m1440 | 46367 | 46367 | 1930 | 44437 | 0 |

## Archived CS incumbents

- **k40_r1_cs_raw_m1440**: 19 of 496 selected routes were invalid as recorded; every one was deterministically repairable. The first was selected route 478: `charge at 3127L_0 raises SOC to 319.9 > capacity 300.0`.
- **k40_r2_cs_raw_m1440**: 15 of 516 selected routes were invalid as recorded; every one was deterministically repairable. The first was selected route 499: `charge at 2190L_0 raises SOC to 300.5 > capacity 300.0`.

No journal route in any of the four pools was infeasible after continuous realization. This does not make the archived CS incumbents physical schedules: their stored schedules remain invalid, and the CA pools remain finite-pool partition-infeasible.
