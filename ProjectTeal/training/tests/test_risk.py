import torch

from training.risk import (
    detect_misalignment,
    estimate_white_balance_neutral,
    quad_binner_residual,
    white_balance_consistency,
)


def test_detect_misalignment_flags_shifted_anchor():
    anchor = torch.rand(1, 1, 6, 6)
    simulated_aligned = anchor.clone()
    simulated_shifted = torch.roll(anchor, shifts=(1, 0), dims=(-2, -1))

    aligned_diag = detect_misalignment(anchor, simulated_aligned, psnr_threshold=50.0)
    misaligned_diag = detect_misalignment(anchor, simulated_shifted, psnr_threshold=50.0)

    assert aligned_diag["is_aligned"] is True
    assert misaligned_diag["is_aligned"] is False
    assert misaligned_diag["best_shift"] != (0, 0)
    assert misaligned_diag["best_psnr"] > misaligned_diag["reference_psnr"]


def test_quad_binner_residual_matches_manual_rggb():
    rgb = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0],
                    [14.0, 15.0, 16.0, 17.0],
                ],
                [
                    [3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0],
                ],
            ]
        ]
    )

    residual = quad_binner_residual(rgb, channel_scale=torch.tensor([1.0, 2.0, 0.5]))
    assert torch.isclose(residual, torch.tensor(0.0))


def test_white_balance_estimation_and_consistency_checks():
    # Construct a 4x4 RGGB mosaic with controlled per-channel means.
    r_value = 0.5
    g_value = 1.0
    b_value = 0.8

    mosaic = torch.tensor(
        [
            [
                [r_value, g_value, r_value, g_value],
                [g_value, b_value, g_value, b_value],
                [r_value, g_value, r_value, g_value],
                [g_value, b_value, g_value, b_value],
            ]
        ]
    ).unsqueeze(0)

    neutral = estimate_white_balance_neutral(mosaic)
    expected_neutral = torch.tensor([2.0, 1.0, 1.25])
    assert torch.allclose(neutral, expected_neutral.view(1, 3), atol=1e-3)

    within, diag = white_balance_consistency(mosaic, expected_neutral, tolerance=0.01)
    assert within is True
    assert torch.all(diag["deviation"] < 1e-3)

    mismatched_neutral = torch.tensor([1.0, 1.0, 1.0])
    within, diag = white_balance_consistency(mosaic, mismatched_neutral, tolerance=0.01)
    assert within is False
    assert torch.any(diag["deviation"] > 0.1)
