import unittest

from tools.exp_2.object_feature_analysis.sampling import (
    allocate_quota,
    select_balanced_instances,
    summarize_manifest,
)


class SamplingTests(unittest.TestCase):
    def test_allocate_quota_exact_and_respects_capacity(self):
        counts = allocate_quota({("a", 0): 2, ("b", 0): 9, ("b", 1): 4}, 10)
        self.assertEqual(sum(counts.values()), 10)
        self.assertLessEqual(counts[("a", 0)], 2)
        self.assertLessEqual(counts[("b", 0)], 9)
        self.assertLessEqual(counts[("b", 1)], 4)

    def test_balanced_selection_is_deterministic(self):
        candidates = []
        instance = 0
        for category in (1, 2, 3):
            for split in ("train", "val"):
                for size_bin in range(4):
                    for repeat in range(5):
                        instance += 1
                        candidates.append(
                            {
                                "instance_id": f"{category}:{split}:{size_bin}:{repeat}",
                                "image_id": instance,
                                "category_id": category,
                                "split": split,
                                "size_bin": size_bin,
                                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                            }
                        )
        first = select_balanced_instances(candidates, 12, 42)
        second = select_balanced_instances(candidates, 12, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        for category in (1, 2, 3):
            self.assertEqual(
                sum(row["category_id"] == category for row in first), 12
            )

    def test_summary_keeps_split_image_groups_distinct(self):
        rows = [
            {
                "category_id": 1,
                "image_id": 10,
                "split": "train",
                "size_bin": 0,
            },
            {
                "category_id": 2,
                "image_id": 10,
                "split": "val",
                "size_bin": 1,
            },
        ]
        summary = summarize_manifest(rows, {1: "a", 2: "b"}, [0.1, 0.2, 0.3])
        self.assertEqual(summary["image_id_values"], 1)
        self.assertEqual(summary["split_image_groups"], 2)
        self.assertEqual(summary["images"], 2)


if __name__ == "__main__":
    unittest.main()
