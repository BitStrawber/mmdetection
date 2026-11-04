import unittest

from tools.exp_2.frozen_backbone_training.analyze_backbone_updates import parse_models


class ModelArgumentTests(unittest.TestCase):
    def test_parse_models(self):
        values = parse_models(["DFUI=/tmp/dfui.pth", "SFUI=/tmp/sfui.pth"])
        self.assertEqual(set(values), {"DFUI", "SFUI"})

    def test_duplicate_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_models(["DFUI=/tmp/a.pth", "DFUI=/tmp/b.pth"])


if __name__ == "__main__":
    unittest.main()
