import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.exp_2.object_feature_analysis.feature_io import (
    assert_aligned_bundles,
    load_feature_bundle,
    save_feature_bundle,
)


class FeatureIoTests(unittest.TestCase):
    def test_save_load_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "labels": np.asarray([1, 2], dtype=np.int64),
                "instance_ids": ["a", "b"],
                "image_ids": np.asarray([10, 20]),
                "group_ids": ["val:10", "val:20"],
                "bboxes": np.asarray([[0, 0, 1, 1], [1, 1, 3, 3]]),
                "split": ["val", "val"],
            }
            first_path = save_feature_bundle(
                root / "first.npz", metadata={"model": "a"}, **common
            )
            second_path = save_feature_bundle(
                root / "second.npz", metadata={"model": "b"}, **common
            )
            first = load_feature_bundle(first_path, verify_array_hashes=True)
            second = load_feature_bundle(second_path, verify_array_hashes=True)
            assert_aligned_bundles({"first": first, "second": second})


if __name__ == "__main__":
    unittest.main()
