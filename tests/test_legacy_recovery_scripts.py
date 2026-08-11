"""Guardrails for the legacy big-tariff recovery launcher and Slurm job.

These scripts encode the requeue, locking, and WAIT-vs-fatal semantics of the
task-22/24/32 recovery.  They previously had no test coverage, so regressions
in the task->cell mapping or the witness-wait classification could only be
noticed on Unicorn.
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from migrate_legacy_exact_pool import MigrationError, discover_witness  # noqa: E402


LAUNCHER = REPO_ROOT / "src" / "launch_legacy_bigtariff_recovery.sh"
JOB = REPO_ROOT / "src" / "submit_legacy_bigtariff_recovery.sub"

# The original 40-task array arithmetic from submit_cg_bigtariffs.sub:
# TASK0 = task - 1; INST = TASK0 % 10 (0-5 -> k30 r1-6, 6-9 -> k40 r1-4);
# TARIFF_IDX = TASK0 // 10 over (peak08, peak12, peak18, sek).
ORIGINAL_TAGS = ("peak08", "peak12", "peak18", "sek")
ORIGINAL_PRICES = (
    "hourly_prices_single_peak_08.csv",
    "hourly_prices_single_peak_12.csv",
    "hourly_prices_single_peak_18.csv",
    "hourly_prices_transdev_sek.csv",
)


def original_array_cell(task: int) -> tuple[str, str, str]:
    task0 = task - 1
    inst = task0 % 10
    tariff_idx = task0 // 10
    if inst <= 5:
        name = f"Practice_Custom_DutyUnion_k30_r{inst + 1}"
    else:
        name = f"Practice_Custom_DutyUnion_k40_r{inst - 5}"
    return name, ORIGINAL_TAGS[tariff_idx], ORIGINAL_PRICES[tariff_idx]


class LegacyRecoveryScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher_text = LAUNCHER.read_text()
        cls.job_text = JOB.read_text()

    def test_scripts_parse_under_bash(self):
        for script in (LAUNCHER, JOB):
            with self.subTest(script=script.name):
                subprocess.run(["/bin/bash", "-n", str(script)], check=True)

    def test_job_is_requeue_safe_and_time_limited(self):
        self.assertIn("#SBATCH --requeue", self.job_text)
        self.assertIn("#SBATCH --open-mode=append", self.job_text)
        self.assertIn("#SBATCH -t 7-00:00:00", self.job_text)
        self.assertIn("#SBATCH --partition=default_partition", self.job_text)
        self.assertIn("--mail-type=REQUEUE,FAIL,TIME_LIMIT", self.job_text)
        self.assertIn("set -euo pipefail", self.job_text)
        self.assertIn("--resume", self.job_text)

    def test_job_lock_fd_uses_append_mode(self):
        # An O_TRUNC open (exec 9>) would wipe the current owner's diagnostic
        # record before the duplicate job loses the flock race.
        self.assertIn('exec 9>>"$OUT.recovery.lock"', self.job_text)
        self.assertNotRegex(self.job_text, r'exec 9>"')

    def test_job_preserves_legacy_identity_parameters(self):
        # The migrated status is identity-checked against these exact model
        # parameters; the job must reproduce the original campaign's values
        # and must not override the defaults the legacy campaign relied on.
        self.assertIn("--soc-step 15", self.job_text)
        self.assertIn("--block-min 10", self.job_text)
        self.assertIn("--rc-eps 0.0001", self.job_text)
        for forbidden in ("--master-sense", "--g-kwh", "--charge-kw",
                          "--min-soc-frac"):
            self.assertNotIn(forbidden, self.job_text)

    def _job_wait_pattern(self) -> str:
        match = re.search(r"grep -Eq '([^']+)'", self.job_text)
        self.assertIsNotNone(match, "job script lost its WAIT grep")
        return match.group(1)

    def test_wait_grep_matches_missing_witness_but_not_conflicts(self):
        pattern = self._job_wait_pattern()
        legacy = {
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300,
            "min_soc_frac": 0.0,
            "master_sense": "partition",
            "trip_ids": [1, 2],
        }
        # A missing witness must be classified as the clean WAIT state.
        with self.assertRaises(MigrationError) as ctx:
            discover_witness(
                [], legacy,
                path_field="csv", hash_field="instance_sha256",
                expected_hash="a" * 64, require_trip_ids=True,
            )
        self.assertRegex(str(ctx.exception), pattern)
        # A conflict must stay fatal: its message must NOT match the WAIT
        # pattern, otherwise conflicting provenance would exit 0 and wait.
        conflict_message = (
            "conflicting instance_sha256 witnesses for instance.csv: "
            "['aa..', 'bb..']"
        )
        self.assertNotRegex(conflict_message, pattern)

    def _launcher_cells(self) -> dict:
        cells = {}
        pattern = re.compile(
            r'(\d+)\)\s+SHORT="([^"]+)";\s*'
            r'CSV_REL="([^"]+)";\s*'
            r'PRICE_REL="([^"]+)";\s*'
            r'SOURCE_CELL="([^"]+)"',
        )
        for match in pattern.finditer(self.launcher_text):
            cells[int(match.group(1))] = {
                "short": match.group(2),
                "csv_rel": match.group(3),
                "price_rel": match.group(4),
                "source_cell": match.group(5),
            }
        return cells

    def _job_cells(self) -> dict:
        cells = {}
        pattern = re.compile(
            r'(\d+)\)\s+NAME="([^"]+)";\s*TAG="([^"]+)";\s*'
            r'PRICE="([^"]+)"',
        )
        for match in pattern.finditer(self.job_text):
            cells[int(match.group(1))] = {
                "name": match.group(2),
                "tag": match.group(3),
                "price": match.group(4),
            }
        return cells

    def test_task_mapping_matches_original_array_in_both_scripts(self):
        launcher_cells = self._launcher_cells()
        job_cells = self._job_cells()
        self.assertEqual(sorted(launcher_cells), [22, 24, 32])
        self.assertEqual(sorted(job_cells), [22, 24, 32])
        for task in (22, 24, 32):
            with self.subTest(task=task):
                name, tag, price = original_array_cell(task)
                job = job_cells[task]
                self.assertEqual(job["name"], name)
                self.assertEqual(job["tag"], tag)
                self.assertEqual(job["price"], price)
                launcher = launcher_cells[task]
                self.assertEqual(
                    launcher["csv_rel"], f"duty_unions_big/{name}.csv"
                )
                self.assertEqual(launcher["price_rel"], price)
                self.assertEqual(launcher["source_cell"], f"{name}_{tag}")

    def _extract_conda_selection_block(self) -> str:
        match = re.search(
            r'\nactivated=""\n.*?\ndone\n', self.job_text, re.DOTALL
        )
        self.assertIsNotNone(
            match, "job script lost its conda environment selection loop"
        )
        return match.group(0)

    def _run_conda_selection(self, *, env_pythons: dict) -> str:
        """Execute the .sub's real selection loop with mocked conda/python.

        ``env_pythons`` maps activatable environment names to the Python
        version their activation would expose.  Returns the selected
        environment name ('' when no candidate was accepted).
        """

        cases = "\n".join(
            f'        {name}) CURRENT_PY={version}; return 0 ;;'
            for name, version in env_pythons.items()
        )
        harness = (
            "set -euo pipefail\n"
            "CURRENT_PY=\"\"\n"
            "conda() {\n"
            "  case \"$1\" in\n"
            "    activate)\n"
            "      case \"$2\" in\n"
            f"{cases}\n"
            "        *) return 1 ;;\n"
            "      esac ;;\n"
            "    deactivate) CURRENT_PY=\"\"; return 0 ;;\n"
            "  esac\n"
            "}\n"
            "python() { [ \"$CURRENT_PY\" = 3.12 ]; }\n"
            + self._extract_conda_selection_block()
            + "echo \"SELECTED=$activated\"\n"
        )
        completed = subprocess.run(
            ["/bin/bash", "-c", harness], text=True, capture_output=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        match = re.search(r"SELECTED=(.*)", completed.stdout)
        self.assertIsNotNone(match, completed.stdout)
        return match.group(1).strip()

    def test_conda_fallback_skips_activatable_non_python312_environment(self):
        # A legacy 3.10 environment that activates first must not stop the
        # search before the validated 3.12 candidate is tried.
        selected = self._run_conda_selection(env_pythons={
            "/home/nc437/evsp_env": "3.10",
            "evsp_env312": "3.12",
        })
        self.assertEqual(selected, "evsp_env312")

    def test_conda_fallback_selects_nothing_without_a_python312_candidate(self):
        # With only non-3.12 environments activatable, the loop must leave
        # $activated empty so the job fatals instead of running mismatched
        # Python against the recovery artifacts.
        selected = self._run_conda_selection(env_pythons={
            "/home/nc437/evsp_env": "3.10",
        })
        self.assertEqual(selected, "")

    def test_launcher_defaults_to_dry_run_and_pins_both_commits(self):
        self.assertIn("SUBMIT=0", self.launcher_text)
        self.assertIn("--submit) SUBMIT=1", self.launcher_text)
        self.assertIn(
            'SOURCE_HEAD" = "$LEGACY_COMMIT"', self.launcher_text.replace(
                '[ "$', '')
        )
        self.assertIn("EVSP_EXPECTED_COMMIT=$COMMIT", self.launcher_text)
        # The compute job re-verifies both checkouts on every (re)start.
        self.assertIn('"$actual_commit" = "$EXPECTED_COMMIT"', self.job_text)
        self.assertIn('"$source_commit" = "$LEGACY_COMMIT"', self.job_text)


if __name__ == "__main__":
    unittest.main()
