"""
Model components for HipMRI prostate segmentation (PyTorch only, no NumPy).

  * ImprovedUNet3D: 3D Improved U-Net (Isensee et al., 2018), the proposed model
  * UNet2D:         standard 2D U-Net (Ronneberger et al., 2015), the slice-wise baseline
  * DiceCELoss:     soft multi-class Dice + cross-entropy, shared by both models

Both networks return raw logits of shape (B, num_classes, *spatial).

Usage (shape and parameter smoke test):
  python modules.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Improved U-Net (Isensee et al., 2018)
# ---------------------------------------------------------------------------

def conv3d(in_ch, out_ch, kernel=3, stride=1):
    """3D convolution with 'same' padding for odd kernels. No bias: a norm layer always follows."""
    return nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)


def norm_act(ch):
    """Instance norm + leaky ReLU, as in the paper. Instance norm suits the batch sizes of 1-2 that 3D forces."""
    return nn.Sequential(nn.InstanceNorm3d(ch, affine=True), nn.LeakyReLU(1e-2, inplace=True))


class ContextModule(nn.Module):
    """
    Pre-activation residual block: two 3x3x3 convs with dropout between them.
    Output = input + block(input), so channels and spatial size are unchanged.
    """

    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            norm_act(ch), conv3d(ch, ch),
            nn.Dropout3d(dropout),
            norm_act(ch), conv3d(ch, ch),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsamplingModule(nn.Module):
    """Double the spatial size (nearest upsampling) then a 3x3x3 conv that sets the channel count."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(conv3d(in_ch, out_ch), norm_act(out_ch))

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LocalizationModule(nn.Module):
    """
    Merge concatenated skip + upsampled features: a 3x3x3 conv then a 1x1x1 conv.
    Takes 2C channels (after concatenation) and returns C.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3d(in_ch, out_ch), norm_act(out_ch),
            conv3d(out_ch, out_ch, kernel=1), norm_act(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    """
    3D Improved U-Net with 5 resolution levels and deep supervision.

    Encoder level k: conv (stride 2 for k > 1) then a context module, base * 2^(k-1) channels.
    Decoder: upsampling module -> concatenate skip -> localization module.
    Deep supervision: 1x1x1 segmentation layers on decoder levels 3, 2 and 1 are upsampled
    and summed, so coarse levels also receive gradient towards the final segmentation.

    Input spatial size must be divisible by 16 (four stride-2 downsamplings).
    """

    def __init__(self, in_ch=1, num_classes=6, base=16, dropout=0.3):
        super().__init__()
        chs = [base * 2 ** k for k in range(5)]  # 16, 32, 64, 128, 256

        # Encoder
        self.enc_convs = nn.ModuleList(
            [conv3d(in_ch, chs[0])] + [conv3d(chs[k - 1], chs[k], stride=2) for k in range(1, 5)]
        )
        self.contexts = nn.ModuleList([ContextModule(c, dropout) for c in chs])

        # Decoder, ordered from the bottleneck upwards
        self.ups = nn.ModuleList([UpsamplingModule(chs[k], chs[k - 1]) for k in range(4, 0, -1)])
        self.locs = nn.ModuleList([LocalizationModule(2 * chs[k], chs[k]) for k in range(3, 0, -1)])
        self.final = nn.Sequential(conv3d(2 * chs[0], 2 * chs[0]), norm_act(2 * chs[0]))

        # Segmentation heads for deep supervision (decoder levels 3, 2, 1)
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(chs[2], num_classes, 1),
            nn.Conv3d(chs[1], num_classes, 1),
            nn.Conv3d(2 * chs[0], num_classes, 1),
        ])

    def forward(self, x):
        skips = []
        for conv, context in zip(self.enc_convs, self.contexts):
            x = context(conv(x))
            skips.append(x)

        x = skips.pop()  # bottleneck
        seg = None
        for k, up in enumerate(self.ups):
            x = torch.cat([skips.pop(), up(x)], dim=1)
            x = self.locs[k](x) if k < 3 else self.final(x)
            if k >= 1:  # decoder levels 3, 2, 1
                level_seg = self.seg_heads[k - 1](x)
                seg = level_seg if seg is None else level_seg + F.interpolate(seg, scale_factor=2, mode="nearest")
        return seg


# ---------------------------------------------------------------------------
# 2D U-Net baseline (Ronneberger et al., 2015)
# ---------------------------------------------------------------------------

class DoubleConv2D(nn.Module):
    """Two 3x3 conv -> batch norm -> ReLU layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """
    Standard 2D U-Net: 4 max-pool downsamplings, transposed-conv upsampling, skip concatenation.
    Applied slice by slice; 3D volumes are rebuilt by stacking slice predictions (see segment_volume_2d).
    Input height and width must be divisible by 16.
    """

    def __init__(self, in_ch=1, num_classes=6, base=32):
        super().__init__()
        chs = [base * 2 ** k for k in range(5)]  # 32, 64, 128, 256, 512
        self.downs = nn.ModuleList([DoubleConv2D(in_ch, chs[0])] +
                                   [DoubleConv2D(chs[k - 1], chs[k]) for k in range(1, 5)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[k], chs[k - 1], 2, stride=2) for k in range(4, 0, -1)])
        self.dec = nn.ModuleList([DoubleConv2D(2 * chs[k - 1], chs[k - 1]) for k in range(4, 0, -1)])
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for k, down in enumerate(self.downs):
            x = down(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)


@torch.no_grad()
def segment_volume_2d(model, volume, batch_size=32):
    """
    Run a 2D model over every axial slice of a (B, C, D, H, W) volume and return
    (B, num_classes, D, H, W) logits, so the baseline is scored exactly like the 3D model.
    """
    b, c, d, h, w = volume.shape
    slices = volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    logits = torch.cat([model(slices[i:i + batch_size]) for i in range(0, b * d, batch_size)])
    return logits.reshape(b, d, -1, h, w).permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """
    Soft multi-class Dice loss plus cross-entropy, for logits of shape (B, C, *spatial).

    Dice is computed per class over the whole batch, then averaged over classes, so the
    small prostate class counts as much as the background. Cross-entropy keeps gradients
    stable early in training, when soft Dice is noisy.
    """

    def __init__(self, dice_weight=1.0, ce_weight=1.0, smooth=1e-5):
        super().__init__()
        self.dice_weight, self.ce_weight, self.smooth = dice_weight, ce_weight, smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes).movedim(-1, 1).to(probs.dtype)
        dims = (0,) + tuple(range(2, logits.ndim))  # sum over batch and spatial axes
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return self.dice_weight * (1 - dice.mean()) + self.ce_weight * F.cross_entropy(logits, target)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    loss_fn = DiceCELoss()

    model3d = ImprovedUNet3D()
    x = torch.randn(1, 1, 32, 64, 64)
    y = torch.randint(0, 6, (1, 32, 64, 64))
    out = model3d(x)
    loss = loss_fn(out, y)
    loss.backward()
    print(f"ImprovedUNet3D: {count_parameters(model3d):,} params, {tuple(x.shape)} -> {tuple(out.shape)}, "
          f"loss {loss.item():.3f}")

    model2d = UNet2D()
    out2d = model2d(x[:, :, 0])
    loss2d = loss_fn(out2d, y[:, 0])
    loss2d.backward()
    print(f"UNet2D:         {count_parameters(model2d):,} params, {tuple(x[:, :, 0].shape)} -> {tuple(out2d.shape)}, "
          f"loss {loss2d.item():.3f}")

    model2d.eval()
    vol = segment_volume_2d(model2d, x)
    print(f"segment_volume_2d: {tuple(x.shape)} -> {tuple(vol.shape)}")


if __name__ == "__main__":
    main()
