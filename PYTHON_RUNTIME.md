# Python runtime

EVSP-DR supports CPython 3.12.x. The reproducible environment currently pins
Python 3.12.13 and the numerical-library versions in `environment-py312.yml`.

The Git repository contains the same source code on the Mac and Unicorn, but
each computer supplies its own Python interpreter and installed packages. A
`git pull` updates source files; it does not install or change Python.

## Create or validate the environment

From the repository root:

```bash
bash src/bootstrap_python312.sh
```

The default environment path is `../.evspdr-envs/py312`, outside the Git
checkout. Override it when needed:

```bash
EVSP_PY312_ENV_PREFIX=/path/to/evspdr-py312 \
  bash src/bootstrap_python312.sh
```

The bootstrap creates a missing environment or updates the specified environment
from the pinned file without Conda's package-pruning option. It then compiles the
source, runs the unit tests, and executes a data/preflight smoke test. It never
deletes the older Python 3.10 environment. Use a dedicated prefix: updating an
existing environment can still change the versions of packages named here.

Use the environment without relying on shell activation:

```bash
PYTHON=../.evspdr-envs/py312/bin/python
"$PYTHON" --version
"$PYTHON" src/run_ex_unicorn.py --help
```

On Unicorn, the existing `/home/nc437/evsp_env` already reported Python
3.12.13 in saved run provenance. Do not rebuild that shared environment while
jobs are using it. To create the pinned environment in parallel, set a distinct
`EVSP_PY312_ENV_PREFIX`, run the bootstrap once, and pass the resulting path as
`EVSP_CONDA_ENV` to maintained submission scripts.

## Dependency choices

`requirements-unicorn.txt` contains the pinned free SciPy/HiGHS stack.
`requirements-gurobi.txt` adds Gurobi 12.0.3 for the optional Gurobi LP backend
and final integer master. Gurobi is not required for ordinary Goal-1 column
generation.

Changing Python is an environment and reproducibility improvement, not a cure
for pricing-search incompleteness. Compare representative pricing calls using
the same commit, inputs, algorithm settings, and dependency versions before
attributing a runtime difference to the interpreter.

The initial migration tests and same-machine pricing comparison are recorded in
`PYTHON312_MIGRATION_REPORT_20260803.md`.
