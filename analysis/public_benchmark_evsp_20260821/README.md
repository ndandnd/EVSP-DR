# Public E-VSP benchmark comparison

Date: 2026-08-21. All EVSP-DR runs were local.

## Benchmark choice

The real Leuven data used by van Kooten Niekerk, van den Akker, and Hoogeveen
(2017), *Scheduling electric vehicles*, are described as operator-provided and
are not publicly downloadable. They therefore cannot support a referee-facing
reproduction package.

This comparison uses the closest verifiable public successor:

- W. ten Bosch, H. Hoogeveen, M. van Kooten Niekerk, and P. de Bruin,
  *Scheduling electric vehicles by simulated annealing with recombination
  through ILP*, Public Transport 18 (2026), 211–228,
  <https://doi.org/10.1007/s12469-026-00424-2>.
- Public instances: <https://github.com/UtrechtUniversity/evsp-instances>
- Pinned source commit:
  `d1b3b05939720355449cf20d47e2527e9a02e05c`
- Selected instance: `qlink_8`.

The upstream repository declares no license. Its files are therefore **not
redistributed** here. `src/convert_utrecht_evsp.py` consumes a separately
cloned copy and records the commit, byte counts, and SHA256 of all three source
files.

| upstream file | bytes | SHA256 |
|---|---:|---|
| `qlink_8/dhd.txt` | 68,714 | `7d1456a96172f2a6916ee6008f3673861dc1068acf97d2641e3210c8e24a6497` |
| `qlink_8/parameters.txt` | 303 | `fe3466e292cca9f127b8d3ca65222a4fd1ba70462b188d10c21f91ed88a07387` |
| `qlink_8/trips.txt` | 339,988 | `a53c8864be470b68f2604ed6d95e83707f80dbe1df1c1c4670141ab5b75a26f6` |

## Published comparator

The paper's Tables 1–2 report for qlink 8:

| trips | reported buses | CG-LP score | CG-ILP score | SA-ILP score |
|---:|---:|---:|---:|---:|
| 193 | 10 | 9,950 | 9,950 | 9,950 |

The paper says `#Buses` is the number used in its best solution, not a separate
proof that ten is the minimum fleet.

## Converter mapping

The converter performs these literal mappings:

- trip start/end/location: `T` records;
- trip energy: published trip distance × the `U`-row 1.90 kWh/km rate;
- deadhead time: the departure-time version selected from the published `G`
  windows and directional `D` profile;
- deadhead energy: published deadhead distance × 1.90 kWh/km;
- depot and charging locations: published `G` and `E` parameter rows;
- battery capacity: the published 160 kWh.

No 57-minute Partille trip-gap cutoff or 61-minute trip-to-charger cutoff is
applied. The adapter extends `ProblemData` with instance-specific depot,
stations, horizon, and departure-time deadhead lookup.

## Assumption comparison

| feature | public qlink 8 model | EVSP-DR converted run | shared? |
|---|---|---|---|
| Battery | 160 kWh | 160 kWh | yes |
| Initial SOC | full | full | yes |
| Minimum/terminal SOC | SOC must remain nonnegative | reserve 0; terminal ≥ 0 | yes at stated tolerance |
| Trip energy | distance × 1.90 kWh/km | same | yes |
| Deadheads | directional, departure-time profiles; distance energy | same profiles and distance energy | yes |
| Charging curve | nonlinear | constant 450 kW | **no** |
| Charging time | continuous available interval | 8-minute blocks | **no** |
| SOC representation | 0.02 normalized levels (3.2 kWh) in published CG | 20 kWh levels | **no** |
| Station setup | 2 minutes | omitted | **no** |
| Station SOC ceiling | some en-route stations cap buses at 0.9 | charge up to 1.0 | **no** |
| Charger capacity | published `E` capacities are 99 (effectively loose) | unlimited | approximately |
| Charging economics | electricity plus battery degradation | flat zero tariff plus $5/run tie-break | **no** |
| Bus/operating cost | published `U`-row costs | EVSP-DR fixed 100,000/bus | **no** |

The conversion is consequently a test of the EVSP-DR model on public timetable
geometry, not a reproduction of the paper's objective.

## Executed EVSP-DR result

The reportable converted model uses a commensurate 20 kWh / 8 minute grid,
160 kWh battery, 450 kW constant charging, and reserve zero:

| metric | EVSP-DR | published |
|---|---:|---:|
| fleet LP | **10.2272727273** | not reported in fleet units |
| integer fleet | **11, proven** | 10 in best published solution |
| fleet difference | **+1 bus (+10%)** | reference |
| objective | 1,100,215 | 9,950 |
| objective gap | **not comparable** | different objective/model |

G1 matched 10,353 nodes and 598,910 arcs; exact reachability retained 378,219
arcs. The integer flow decomposed into 11 routes covering all 193 trips exactly
once and passed physical replay. The LP lower bound is above ten, so the
11-route witness proves the converted discretized model's fleet optimum.

An exploratory 10 kWh / 4 minute run had fleet LP 9.5 but returned no integer
incumbent within 600 seconds. It is reported separately in `results.csv` and is
not substituted for the proven 20/8 result.

The +1 bus is **not** evidence that EVSP-DR underperforms the published
algorithm. It measures a materially different charging/SOC model on the same
public trips and deadheads.

## Reproduction

```bash
git clone https://github.com/UtrechtUniversity/evsp-instances.git /tmp/evsp-instances
git -C /tmp/evsp-instances checkout d1b3b05939720355449cf20d47e2527e9a02e05c

python3 src/convert_utrecht_evsp.py \
  --source-dir /tmp/evsp-instances/qlink_8 \
  --name utrecht_qlink8 \
  --upstream-commit d1b3b05939720355449cf20d47e2527e9a02e05c \
  --out /tmp/utrecht_qlink8.json

python3 src/run_public_evsp_benchmark.py \
  --problem /tmp/utrecht_qlink8.json \
  --g-kwh 160 --charge-kw 450 --min-soc-frac 0 \
  --soc-step 20 --block-min 8 \
  --published-fleet 10 \
  --published-cg-lp-score 9950 \
  --published-integer-score 9950 \
  --integrality service --time-limit-s 600 --max-fleet 15 \
  --out /tmp/utrecht_qlink8_result.json
```
