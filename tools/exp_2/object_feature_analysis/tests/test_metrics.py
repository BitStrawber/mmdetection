import unittest

import numpy as np

from tools.exp_2.object_feature_analysis.metrics import (
    fisher_discriminant_ratio,
    l2_normalize,
)


class MetricTests(unittest.TestCase):
    def test_fisher_ratio_is_positive(self):
        features = np.asarray(
            [[1.0, 0.0], [1.1, 0.1], [0.0, 1.0], [0.1, 1.1]]
        )
        labels = np.asarray([0, 0, 1, 1])
        metrics = fisher_discriminant_ratio(features, labels)
        self.assertGreater(metrics["d_inter"], 0.0)
        self.assertGreater(metrics["fdr"], 0.0)

    def test_l2_normalization(self):
        values = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
