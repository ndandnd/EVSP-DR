# Executed local commands

All commands ran from the repository root. No Slurm or cluster command ran.

## k02_s3 performance gate

Producer: `ba9c17c8bb1ba0604bebe6f8f9e0291c1971140d`.

```bash
/usr/bin/python3.12 src/exact_pricer_expanded.py \
  --time-model event --event-arc-mode lazy \
  --csv scale_ladder/instances/Practice_Custom_DutyUnion_k02_r3.csv \
  --prices_csv hourly_prices_flat.csv \
  --g-kwh 240 --charge-kw 240 --min-soc-frac 0 \
  --soc-step 2.5 --block-min 5 \
  --master-sense partition --initial-pool singletons \
  --max-iters 2000 --columns_per_iter 30 --checkpoint-every 25 \
  --wall-limit-s 900 \
  --phase-telemetry /tmp/event-v2-k02s3-ba9c17c.phases.jsonl \
  --out /tmp/event-v2-k02s3-ba9c17c.json
```

## k05_s2 canary

Producer: `f0b482cf0390ae51ea091871785e70e38ff68671`.

```bash
/usr/bin/python3.12 src/exact_pricer_expanded.py \
  --time-model event --event-arc-mode lazy \
  --csv tariff_response/frozen_instances/Practice_Custom_DutyUnion_k05_r2.csv \
  --prices_csv hourly_prices_flat.csv \
  --g-kwh 240 --charge-kw 240 --min-soc-frac 0 \
  --soc-step 2.5 --block-min 5 \
  --master-sense partition --initial-pool singletons \
  --max-iters 2000 --columns_per_iter 30 --checkpoint-every 25 \
  --wall-limit-s 7200 \
  --phase-telemetry /tmp/event-v2-k05s2-f0b482c-v2.phases.jsonl \
  --out /tmp/event-v2-k05s2-f0b482c-v2.json
```

The first k5 command used the nonexistent scale-ladder k05_r2 path and exited
before model construction. The preserved canary uses the frozen input shown
above and reproduces the prior event-lattice hash and 22,161,911 arc count.

## Strict physical audits

Audit revision: `f7baa6221f2ab87074c57964c7f1de4f13d7900a`.

```bash
/usr/bin/python3.12 src/audit_event_pricer_witnesses.py \
  --status /tmp/event-v2-k02s3-ba9c17c.json \
  --out /tmp/event-v2-k02s3-ba9c17c.physical-f7baa62.json

/usr/bin/python3.12 src/audit_event_pricer_witnesses.py \
  --status /tmp/event-v2-k05s2-f0b482c-v2.json \
  --out /tmp/event-v2-k05s2-f0b482c-v2.physical-f7baa62.json
```

The first k5 audit was also retained. Its 52 rejections isolated the
late-horizon tariff-identity mismatch fixed by `f7baa62`; the corrected audit
has zero rejection.

## Tests

```bash
python3 -m pytest -q \
  tests/test_exact_pricer_resume.py \
  tests/test_k2_default_bit_identity.py \
  tests/test_event_pricer_network.py \
  tests/test_expanded_path_realization.py

python3 -m pytest -q \
  tests/test_event_pricer_network.py \
  tests/test_event_pricer_gates.py \
  tests/test_expanded_path_realization.py

python3 -m compileall -q src tests
python3 -m pytest -q
```

The full-suite nonzero exit is classified in `REPORT.md`: 526 tests and 126
subtests passed; two frozen historical-artifact checks detected only the
expected producer-code hash changes and were not regenerated.
