import torch
import torch.nn as nn


class DenoiseLoss(nn.Module):
    """Combined time-domain + frequency-domain MSE loss.

    L = MSE(y_hat, y) + lambda_freq * MSE(|FFT(y_hat)|, |FFT(y)|)

    The frequency term prevents spectral over-smoothing and preserves HRF shape.
    """

    def __init__(self, lambda_freq: float = 0.1):
        super().__init__()
        self.lambda_freq = lambda_freq

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time_loss = nn.functional.mse_loss(y_hat, y)

        # Magnitude spectrum along time axis (dim=-1)
        mag_hat = torch.abs(torch.fft.rfft(y_hat, dim=-1))
        mag_y = torch.abs(torch.fft.rfft(y, dim=-1))
        freq_loss = nn.functional.mse_loss(mag_hat, mag_y)

        return time_loss + self.lambda_freq * freq_loss
