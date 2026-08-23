import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_exact_pool_mip_highs import _incidence, _native_milp  # noqa: E402


class HighsPoolMipTests(unittest.TestCase):
    def test_incidence_preserves_exact_partition_rows(self):
        routes = [
            {"trips": [10, 30]},
            {"trips": [20]},
            {"trips": [10, 20, 30]},
        ]
        matrix = _incidence(routes, [10, 20, 30]).toarray()
        np.testing.assert_array_equal(
            matrix,
            np.asarray([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]),
        )

    def test_native_highs_accepts_integer_warm_start_and_threads(self):
        solved = _native_milp(
            objective=np.ones(2),
            matrix=csr_matrix(np.eye(2)),
            row_lower=np.ones(2),
            row_upper=np.ones(2),
            time_limit=30,
            mip_gap=0.0,
            threads=2,
            start_indices=[0, 1],
        )
        self.assertEqual(solved.status, 0)
        np.testing.assert_allclose(solved.x, [1.0, 1.0])
        self.assertAlmostEqual(solved.mip_dual_bound, 2.0)


if __name__ == "__main__":
    unittest.main()
