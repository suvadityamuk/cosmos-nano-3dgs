import pytest

from cosmos3_gsplat.splat_trainer import _l1_loss, _ssim_loss

torch = pytest.importorskip("torch")
functional = torch.nn.functional


def _reference_window(window_size, channels, device, dtype):
    positions = torch.arange(window_size, device=device, dtype=torch.float32)
    gaussian = torch.exp(-((positions - window_size // 2) ** 2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = gaussian.unsqueeze(1).mm(gaussian.unsqueeze(0)).float().unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous().to(dtype)


def _reference_ssim_loss(image1, image2, window_size=11):
    channels = image1.shape[1]
    window = _reference_window(window_size, channels, image1.device, image1.dtype)
    padding = window_size // 2
    mean1 = functional.conv2d(image1, window, padding=padding, groups=channels)
    mean2 = functional.conv2d(image2, window, padding=padding, groups=channels)
    mean1_squared = mean1.pow(2)
    mean2_squared = mean2.pow(2)
    mean_product = mean1 * mean2
    variance1 = functional.conv2d(image1 * image1, window, padding=padding, groups=channels) - mean1_squared
    variance2 = functional.conv2d(image2 * image2, window, padding=padding, groups=channels) - mean2_squared
    covariance = functional.conv2d(image1 * image2, window, padding=padding, groups=channels) - mean_product
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mean_product + c1) * (2 * covariance + c2)) / (
        (mean1_squared + mean2_squared + c1) * (variance1 + variance2 + c2)
    )
    return 1.0 - ssim_map.mean()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_l1_and_ssim_are_exactly_equal_to_reference(dtype) -> None:
    generator = torch.Generator().manual_seed(2026)
    predicted = torch.rand((2, 3, 32, 37), generator=generator, dtype=dtype)
    target = torch.rand((2, 3, 32, 37), generator=generator, dtype=dtype)

    assert torch.equal(_l1_loss(predicted, target), functional.l1_loss(predicted, target, reduction="none"))
    assert torch.equal(_ssim_loss(predicted, target), _reference_ssim_loss(predicted, target))


def test_local_ssim_gradient_is_exactly_equal_to_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.rand((1, 3, 24, 29), generator=generator, dtype=torch.float64)
    target = torch.rand((1, 3, 24, 29), generator=generator, dtype=torch.float64)
    local_input = base.clone().requires_grad_(True)
    reference_input = base.clone().requires_grad_(True)

    _ssim_loss(local_input, target).backward()
    _reference_ssim_loss(reference_input, target).backward()

    assert torch.equal(local_input.grad, reference_input.grad)


@pytest.mark.parametrize(
    "predicted,target",
    [
        (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16)),
        (torch.ones(1, 3, 16, 16), torch.ones(1, 3, 16, 16)),
        (torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16)),
        (torch.full((1, 3, 16, 16), 1e-7), torch.full((1, 3, 16, 16), 1e-7)),
    ],
)
def test_local_losses_are_finite_for_constant_and_extreme_inputs(predicted, target) -> None:
    l1 = _l1_loss(predicted, target).mean()
    ssim = _ssim_loss(predicted, target)
    assert torch.isfinite(l1)
    assert torch.isfinite(ssim)
    assert torch.equal(ssim, _reference_ssim_loss(predicted, target))
