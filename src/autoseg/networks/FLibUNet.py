from funlib.learn.torch.models import UNet


def setup_unet(
    in_channels: int = 1,
    num_fmaps: float = 12,
    fmap_inc_factor: int = 3,
    downsample_factors: list = [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    num_heads: int = 2,
    padding: str = "valid",
) -> UNet:
    return UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        constant_upsample=True,
        num_heads=num_heads,
        padding=padding,
    )
