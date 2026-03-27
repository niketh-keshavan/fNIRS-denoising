import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding so output length == input length."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class DenoiseTCN(nn.Module):
    """1D Temporal Convolutional Network for fNIRS denoising.

    Architecture (from PLAN.md):
        Input:  [batch, 2, 128]   (HbO + HbR, 128-sample window at 10 Hz)
        4x dilated causal conv layers (d=1,2,4,8) with residual connections
        1x pointwise projection
        Output: [batch, 2, 128]   (denoised HbO + HbR)

    Receptive field: 1 + (kernel-1) * sum(dilations) = 91 samples = 9.1s at 10 Hz
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        layers = []
        ch_in = in_channels
        for d in dilations:
            layers.append(CausalConv1d(ch_in, hidden_channels, kernel_size, dilation=d))
            layers.append(nn.ReLU())
            ch_in = hidden_channels
        self.backbone = nn.ModuleList(layers)

        self.projection = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        # 1x1 adapters for residual connections where channel dims differ
        self._skip_proj = nn.Conv1d(in_channels, hidden_channels, 1) if in_channels != hidden_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(0, len(self.backbone), 2):
            conv = self.backbone[i]
            relu = self.backbone[i + 1]
            out = relu(conv(h))
            # residual: project h if channel mismatch, else identity
            if h.shape[1] != out.shape[1]:
                h = out + self._skip_proj(h)
            else:
                h = out + h
        return self.projection(h)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_kb(self, dtype_bytes: int = 4) -> float:
        return self.param_count() * dtype_bytes / 1024
