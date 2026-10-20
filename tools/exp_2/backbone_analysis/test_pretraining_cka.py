import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .common import write_json
from .compare_pretraining_cka import GROUPS, validate_provenance
from .compute_cka import centered_gram


class PretrainingCkaTests(unittest.TestCase):
    def test_centered_gram_self_similarity(self):
        features = np.array([[1., 2.], [3., 5.], [5., 4.]])
        gram, norm = centered_gram(features, 1e-12)
        self.assertAlmostEqual(float(np.sum(gram * gram) / (norm * norm)), 1.)

    def test_provenance_checks_all_six_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = {}
            for backbone in GROUPS:
                mapping[backbone] = {}
                for source in ('ImageNet', 'Synthetic5', 'RealUW'):
                    model_id = f'{backbone}_{source}'
                    mapping[backbone][source] = model_id
                    model_root = root / 'metadata' / model_id
                    write_json(model_root / 'clean.json', [
                        {'sample_index': index, 'image_id': image_id}
                        for index, image_id in enumerate((10, 20, 30))
                    ])
                    allowed = ([f'blocks.{index}.attn.rel_pos_{axis}'
                                for index in range(12) for axis in ('h', 'w')]
                               if backbone == 'vit_small' else [])
                    write_json(model_root / 'extraction_provenance.json', {
                        'manifest_sha256': 'same-manifest',
                        'config_sha256': backbone,
                        'pooling': 'avg',
                        'model_load_report': {
                            'checkpoint_epoch': 100,
                            'required_match_ratio': 1.0,
                            'initialization_seed': 42 if allowed else None,
                            'allowed_missing_keys': allowed,
                            'allowed_missing_sha256': 'same-init' if allowed else '',
                        },
                    })
            self.assertEqual(
                validate_provenance(root, mapping, 'clean')['image_count'], 3)

            broken = root / 'metadata' / mapping['vit_small']['RealUW']
            record = json.loads(
                (broken / 'extraction_provenance.json').read_text())
            record['model_load_report']['allowed_missing_sha256'] = 'different'
            write_json(broken / 'extraction_provenance.json', record)
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                validate_provenance(root, mapping, 'clean')


if __name__ == '__main__':
    unittest.main()
