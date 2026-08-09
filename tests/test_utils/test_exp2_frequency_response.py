from pathlib import Path

import numpy as np
from PIL import Image

from tools.exp_2.backbone_analysis.compute_frequency_response import (
    centered,
    cosine_similarity,
    input_metrics,
    variant_kind,
)
from tools.exp_2.backbone_analysis.compute_fourier_basis_sensitivity import (
    perturbation,
)


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode='RGB').save(path)


def test_variant_kind() -> None:
    assert variant_kind('low') == ('band_pass', 'low')
    assert variant_kind('remove_high') == ('band_stop', 'high')


def test_cosine_similarity() -> None:
    first = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    second = np.asarray([[1.0, 0.0], [-1.0, -1.0]])
    actual = cosine_similarity(first, second, 1e-12)
    np.testing.assert_allclose(actual, [1.0, -1.0])


def test_input_metrics_use_centered_model_inputs(tmp_path: Path) -> None:
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    clean[:, 2:, :] = 200
    low = np.full((4, 4, 3), 100, dtype=np.uint8)
    low[:, 2:, :] = 150
    clean_path = tmp_path / 'clean.png'
    low_path = tmp_path / 'low.png'
    save_rgb(clean_path, clean)
    save_rgb(low_path, low)
    rows = [{
        'sample_index': 0,
        'variants': {
            'clean': {'image_path': str(clean_path)},
            'low': {'image_path': str(low_path)},
        },
    }]

    metrics = input_metrics(rows, 'clean', ['low'], 1e-12)['low']
    assert metrics['clean_centered_rms'][0] > 0
    assert metrics['variant_centered_rms'][0] > 0
    np.testing.assert_allclose(
        metrics['input_centered_rms_ratio'][0], 0.25, rtol=1e-6)
    np.testing.assert_allclose(
        metrics['input_relative_shift'][0], 0.75, rtol=1e-6)
    np.testing.assert_allclose(centered(np.ones((2, 2, 3))), 0.0)


def test_fourier_perturbation_has_requested_shape_and_range() -> None:
    value = perturbation(17, 23, fx=0.125, fy=0.25, phase=0.0)

    assert value.shape == (17, 23)
    assert value.dtype == np.float32
    assert float(value.min()) >= -1.0
    assert float(value.max()) <= 1.0
