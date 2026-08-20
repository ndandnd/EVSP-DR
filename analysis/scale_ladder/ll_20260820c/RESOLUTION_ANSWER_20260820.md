# Can the exact pricer match GIRO's fleet on small instances?

**Answer: yes.** All three k2 cells reach LP route weight **exactly 2.0000 with a
pricing certificate** at 1 kWh / 5 min. Seven of the nine k2/k3/k5 cells reach
their exact GIRO fleet with certificates. The binding discretization is **time
(`block_min`), not SOC** — and that refutes the natural "GIRO uses 1% SOC steps"
explanation.

This document exists so the question is not reopened. Data:
`resolution_matrix.csv` (40 rows), campaign `ll_20260820c`, commit
`339db0ab917a3db1b47e63e4debcab3066d0af79`, plan sha256
`063e413fc24f88a8801c2ebc03f979176994aa075631b86d4258e83f9220a277`.

## 1. The resolution matrix

LP route weight; **bold** = reaches the GIRO target exactly. All cells certified
(`min_rc = 0`) except two noted below.

| cell | target | 15/10 | 5.0/10 | 2.5/10 | 1.0/10 | 1.0/5 |
|---|---:|---:|---:|---:|---:|---:|
| k02_s1 | 2 | 2.1818 | **2.0000** | **2.0000** | **2.0000** | **2.0000** |
| k02_s2 | 2 | 2.1875 | 2.1667 | 2.1538 | 2.1538 | **2.0000** |
| k02_s3 | 2 | 2.2747 | 2.1893 | 2.1471 | 2.1400 | **2.0000** |
| k03_s1 | 3 | 3.1818 | **3.0000** | **3.0000** | **3.0000** | not run |
| k03_s2 | 3 | 3.4047 | 3.3139 | 3.2619 | 3.2549 | not run |
| k03_s3 | 3 | **3.0000** | **3.0000** | **3.0000** | **3.0000** | not run |
| k05_s1 | 5 | 5.3237 | 5.2558 | 5.1786 | 5.1758 | not run |
| k05_s2 | 5 | **5.0000** | **5.0000** | **5.0000** | **5.0000** | not run |
| k05_s3 | 5 | **5.0000** | **5.0000**\* | **5.0000**\* | **5.0000** | not run |

\* `k05_s3` at 5.0/10 and 2.5/10 ended with `min_rc = -0.68` and `-8.5e-5` against
a ~500,000 objective: effectively converged, formally uncertified. Its 15/10 and
1.0/10 runs are certified at 5.0000.

## 2. The mechanism: time binds, SOC saturates

Two clean controlled comparisons — SOC step held at 1.0 kWh, only `block_min`
changed from 10 to 5:

- `k02_s2`: 2.1538 → **2.0000**
- `k02_s3`: 2.1400 → **2.0000**

Meanwhile refining SOC alone **plateaus**. `k02_s2` across 15 → 5 → 2.5 → 1.0 kWh
at 10-minute blocks: 2.1875 → 2.1667 → 2.1538 → 2.1538. It stops improving below
about 2.5 kWh. Same story for `k02_s3`: 2.2747 → 2.1893 → 2.1471 → 2.1400.

**SOC refinement below ~2.5 kWh buys nothing. Halving the time block closes the
gap completely.**

Physical reading: charging durations, deadhead legs, and layover connections are
quantized to the time lattice. A duty needing a 7-minute connection or a
25-minute charge cannot be expressed on a 10-minute lattice at any SOC
resolution. This is consistent with the duty-13411 grid oracle, whose failures
were all *transitions* (106->119, 119->132, 158->167) rather than SOC states.

## 3. Why "GIRO uses 1% SOC division" is NOT the explanation

1% of a 300 kWh battery is 3 kWh; of 240 kWh, 2.4 kWh. Our 2.5 kWh grid is
therefore already at GIRO's SOC granularity — and at 2.5 kWh, `k02_s2` is still
2.1538, not 2. Refining ten times further, to 1.0 kWh, changes nothing
(2.1538). The SOC axis is not what is binding.

If we wrote "matching GIRO's 1% SOC division explains the gap," this table would
contradict us. The correct statement is about **time discretization**.

## 4. What this licenses us to claim

Licensed:

- The exact expanded-network pricer **reproduces GIRO's fleet exactly, with a
  reduced-cost certificate**, on every k2 instance at 1 kWh / 5 min, and on
  7 of 9 k2/k3/k5 instances overall.
- The cost of the primary 15 kWh / 10 min grid is **measurable in whole buses**:
  6 of 10 certified primary-grid cells need exactly one bus more than GIRO;
  4 match exactly.
- Route-space resolution is a **measured axis**, not a hidden assumption.

Not licensed:

- A certified route weight is a bound **for the named discretized model only**.
  The discretization both removes routes and overstates costs (SOC flooring is
  conservative), so it is **not** a lower bound on the true continuous problem.
  "2.1875 proves 3 buses are needed" is false; "on the 15/10 grid, 3 buses are
  needed, and GIRO's real 2-bus schedule shows the grid is discarding a bus" is
  true.
- `route_weight` remains **combined-cost-master route weight**, not a fleet LP
  bound, except via the D0013 bracket: bus cost is 100,000 and charging at the
  optimum is $86-$2,469, so `w - W* <= C/100000 ~ 0.03` buses.
- Uncertified cells give **no** fleet bound in either direction; their route
  weight is an upper bound on the LP value and can only fall.

## 5. The only gap left

`k03_s2` (3.2549) and `k05_s1` (5.1758) never reach their target at any tested
10-minute grid, and **were never run at `block_min = 5`** — the resolution that
resolved both stubborn k2 cells. Until those two cells are run at 1.0/5, the
honest count is **7 of 9 confirmed exact, 2 of 9 untested at the decisive
resolution**.

That is a 2-cell run, minutes to ~1 hour each. It is the last thing standing
between us and an unqualified "the pricer matches GIRO at small scale" claim.
Do not reopen this question without running it.

## 6. Provenance

`route_weight`, `min_rc`, `artificials`, `pool_columns` are the final rows of the
per-cell `*.iters.csv` files (schema:
`elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns`).
Artificials were zero from iteration 1 in all 39 cells: the singleton initial
pool covers every trip. Provenance label:
`ladder_lite_direct_array` (see D0004); tonight's rows bind to a local-only
commit (see D0010).
