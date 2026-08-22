import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_exact_pool_mip_highs import _incidence  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
