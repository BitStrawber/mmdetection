import unittest

from tools.exp_2.object_feature_analysis.extract_roi_features import (
    _resolve_token_grid,
)


class ExtractHelperTests(unittest.TestCase):
    def test_token_grid_does_not_drop_patch_when_cls_is_absent(self):
        height, width, drop = _resolve_token_grid(4200, (800, 1344), 16, 1)
        self.assertEqual((height, width, drop), (50, 84, 0))

    def test_token_grid_drops_configured_cls_token(self):
        height, width, drop = _resolve_token_grid(4201, (800, 1344), 16, 1)
        self.assertEqual((height, width, drop), (50, 84, 1))


if __name__ == "__main__":
    unittest.main()

