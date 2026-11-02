import unittest

import numpy as np

from tools.exp_2.object_feature_analysis.bbox_mapping import (
    coordinate_feature_map,
    normalized_box_centers,
    project_boxes_to_feature,
    roundtrip_error,
    unproject_boxes_from_feature,
)


class BboxMappingTests(unittest.TestCase):
    def test_projection_and_roundtrip(self):
        boxes = np.asarray(
            [[0.0, 0.0, 250.0, 125.0], [100.0, 50.0, 900.0, 450.0]]
        )
        projected = project_boxes_to_feature(boxes, (500, 1000), (32, 64))
        np.testing.assert_allclose(
            projected,
            [[0.0, 0.0, 16.0, 8.0], [6.4, 3.2, 57.6, 28.8]],
        )
        restored = unproject_boxes_from_feature(
            projected, (500, 1000), (32, 64)
        )
        np.testing.assert_allclose(restored, boxes)
        self.assertLess(float(roundtrip_error(boxes, (500, 1000), (32, 64)).max()), 1e-10)

    def test_coordinate_feature_map_shape(self):
        values = coordinate_feature_map((8, 16))
        self.assertEqual(values.shape, (2, 8, 16))
        self.assertEqual(float(values[0, 0, 0]), 0.0)
        self.assertEqual(float(values[0, 0, -1]), 1.0)
        self.assertEqual(float(values[1, -1, 0]), 1.0)

    def test_box_centers(self):
        boxes = np.asarray([[0.0, 0.0, 8.0, 4.0]])
        centers = normalized_box_centers(boxes, (8, 16))
        np.testing.assert_allclose(centers, [[0.25, 0.25]])


if __name__ == "__main__":
    unittest.main()
