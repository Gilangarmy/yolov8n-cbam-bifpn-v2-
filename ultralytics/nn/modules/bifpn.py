import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


def fast_norm(weights, eps=1e-4):
    return F.relu(weights) / (torch.sum(F.relu(weights), dim=0) + eps)


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C3, C4, C5 = channels

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
        self.w4 = nn.Parameter(torch.ones(2))

        self.P3_conv = SeparableConvBlock(C3, C3)
        self.P4_conv = SeparableConvBlock(C4, C4)
        self.P5_conv = SeparableConvBlock(C5, C5)

        self.P4_down = SeparableConvBlock(C4, C4)
        self.P5_down = SeparableConvBlock(C5, C5)

    def forward(self, P3, P4, P5):
        # Top-down
        w1 = fast_norm(self.w1)
        P4_up = w1[0] * P4 + w1[1] * F.interpolate(P5, size=P4.shape[-2:], mode="nearest")
        P4_up = self.P4_conv(P4_up)

        w2 = fast_norm(self.w2)
        P3_up = w2[0] * P3 + w2[1] * F.interpolate(P4_up, size=P3.shape[-2:], mode="nearest")
        P3_up = self.P3_conv(P3_up)

        # Bottom-up
        w3 = fast_norm(self.w3)
        P4_down = w3[0] * P4_up + w3[1] * F.max_pool2d(P3_up, 2)
        P4_down = self.P4_down(P4_down)

        w4 = fast_norm(self.w4)
        P5_down = w4[0] * P5 + w4[1] * F.max_pool2d(P4_down, 2)
        P5_down = self.P5_down(P5_down)

        return P3_up, P4_down, P5_down


class BiFPN(nn.Module):
    """
    Full BiFPN Stack (num_layers times)
    """
    def __init__(self, channels=(256, 512, 1024), num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(channels) for _ in range(num_layers)
        ])

    def forward(self, inputs):
        """
        inputs: list berisi [P3, P4, P5]
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                f"BiFPN expects 3 feature maps (P3,P4,P5), but got: {type(inputs)} "
                f"with len={len(inputs) if hasattr(inputs,'__len__') else 'N/A'}"
            )

        P3, P4, P5 = inputs

        for layer in self.layers:
            P3, P4, P5 = layer(P3, P4, P5)

        return [P3, P4, P5]
