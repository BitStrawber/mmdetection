import numpy as np

from tools.exp_2.backbone_analysis.generate_frequency_bands import (
    decompose_soft,
    parse_bands,
    parse_energy_quantiles,
    quantile_from_histogram,
    soft_partition_masks,
)


def test_soft_masks_form_partition_of_unity():
    bands = parse_bands('', 'soft-cpp')
    masks = soft_partition_masks(127, 193, bands, transition_ratio=0.25)
    stack = np.stack([masks[band.name] for band in bands])

    assert np.all(stack >= 0)
    assert np.all(stack <= 1)
    np.testing.assert_allclose(stack.sum(axis=0), 1.0, atol=1e-6)


def test_signed_bands_reconstruct_image_after_reflect_padding():
    rng = np.random.default_rng(2026)
    image = rng.random((79, 113, 3), dtype=np.float32)
    bands = parse_bands('', 'soft-cpp')

    outputs, _ = decompose_soft(
        image, bands, transition_ratio=0.25, pad_fraction=0.05)
    reconstructed = sum(outputs.values())

    np.testing.assert_allclose(reconstructed, image, atol=1e-6)
    assert any(float(value.min()) < 0 for value in outputs.values())


def test_default_cutoffs_are_cycles_per_pixel():
    low, mid, high = parse_bands('', 'soft-cpp')

    assert low.low == 0
    assert low.high == 1 / 32
    assert mid.low == 1 / 32
    assert mid.high == 1 / 8
    assert high.low == 1 / 8
    assert high.high is None


def test_energy_quantiles_support_fraction_syntax():
    assert parse_energy_quantiles('1/3,2/3') == (1 / 3, 2 / 3)


def test_histogram_quantile_interpolates_inside_bin():
    edges = np.asarray([0.0, 0.1, 0.2, 0.3])
    cumulative = np.asarray([0.2, 0.6, 1.0])

    np.testing.assert_allclose(
        quantile_from_histogram(edges, cumulative, 0.4), 0.15)
