import unittest

from tools.exp_2.frozen_backbone_training.prepare_frozen_training import (
    build_derived_config,
)


class FrozenConfigTests(unittest.TestCase):
    def test_resnet_freezes_all_four_stages(self):
        text = build_derived_config(
            "/tmp/base.py",
            "/tmp/backbone.pth",
            "resnet50",
            "coco/bbox_mAP",
        )
        self.assertIn("frozen_stages=4", text)
        self.assertIn("norm_eval=True", text)
        self.assertIn("save_best='coco/bbox_mAP'", text)

    def test_vit_freezes_all_twelve_blocks(self):
        text = build_derived_config(
            "/tmp/base.py",
            "/tmp/backbone.pth",
            "vit_small",
            "coco/segm_mAP",
        )
        self.assertIn("frozen_stages=12", text)
        self.assertNotIn("norm_eval=", text)
        self.assertIn("save_best='coco/segm_mAP'", text)


if __name__ == "__main__":
    unittest.main()

