import pytest

torch = pytest.importorskip("torch")

from training import evaluation


def test_psnr_and_ssim_identical_inputs():
    img = torch.rand(1, 1, 8, 8)
    assert torch.isinf(evaluation.psnr(img, img))
    ssim_val = evaluation.ssim(img, img)
    assert torch.isclose(ssim_val, torch.tensor(1.0))


def test_psnr_degrades_with_error():
    img = torch.zeros(1, 1, 4, 4)
    noisy = img + 0.5
    psnr_val = evaluation.psnr(img, noisy)
    assert psnr_val < 5


def test_per_cfa_error_histogram_counts_bins_per_channel():
    residual = torch.tensor(
        [
            [[[0.1, -0.1, 0.1, -0.1], [0.0, 0.0, 0.0, 0.0], [0.05, -0.05, 0.05, -0.05], [0.0, 0.0, 0.0, 0.0]]]
        ]
    )

    hist = evaluation.per_cfa_error_histogram(residual, num_bins=10, value_range=(-0.1, 0.1))

    assert set(hist.keys()) == {"r", "g", "b"}
    assert hist["r"].sum() > 0
    assert hist["g"].sum() > 0
    assert hist["b"].sum() > 0


def test_gradient_consistency_metrics_tracks_direction_and_halo():
    base = torch.zeros(1, 3, 4, 4)
    pred = base.clone()
    pred[:, :, :, 1:] = 1.0

    ref = base.clone()
    ref[:, :, 1:, :] = 1.0

    metrics = evaluation.gradient_consistency_metrics(pred, ref)

    assert metrics["corr_x"] < 0.5  # Gradients differ in direction
    assert metrics["corr_y"] < 0.5
    assert metrics["halo_l1"] > 0

