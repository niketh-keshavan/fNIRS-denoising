import numpy as np
from scipy import signal as sp_signal

ALL_NOISE_TYPES = ["mayer", "respiratory", "cardiac", "motion", "electronic"]


def synthesize_mayer_wave(
    n_samples: int, sfreq: float = 10.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples) / sfreq
    freq = rng.uniform(0.08, 0.12)
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0.5, 1.5)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def synthesize_respiratory(
    n_samples: int, sfreq: float = 10.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    white = rng.standard_normal(n_samples)
    nyq = sfreq / 2.0
    low, high = 0.1 / nyq, 0.3 / nyq
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)
    if low >= high:
        return white
    sos = sp_signal.butter(4, [low, high], btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, white)


def synthesize_cardiac(
    n_samples: int, sfreq: float = 10.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    hr = rng.uniform(50, 100)
    beat_interval = 60.0 / hr
    out = np.zeros(n_samples)
    t = 0.0
    while t < n_samples / sfreq:
        jitter = rng.normal(0, 0.05 * beat_interval)
        beat_t = t + jitter
        beat_idx = int(beat_t * sfreq)
        if 0 <= beat_idx < n_samples:
            template_len = int(0.6 * sfreq)
            template = np.zeros(template_len)
            rise = max(1, int(0.1 * sfreq))
            decay = max(1, int(0.3 * sfreq))
            peak = min(rise, template_len)
            template[:peak] = np.linspace(0, 1, peak)
            decay_end = min(peak + decay, template_len)
            template[peak:decay_end] = np.exp(
                -np.arange(decay_end - peak) / (0.1 * sfreq)
            )
            end_idx = min(beat_idx + template_len, n_samples)
            copy_len = end_idx - beat_idx
            out[beat_idx:end_idx] += template[:copy_len]
        t += beat_interval
    return out


def synthesize_motion_spikes(
    n_samples: int, sfreq: float = 10.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    duration = n_samples / sfreq
    n_spikes = rng.poisson(0.05 * duration)
    out = np.zeros(n_samples)
    if n_spikes == 0:
        return out
    locations = rng.integers(0, n_samples, size=n_spikes)
    amplitudes = rng.lognormal(0, 1, size=n_spikes)
    sigma = 0.1 * sfreq
    kernel_half = int(4 * sigma)
    kernel_t = np.arange(-kernel_half, kernel_half + 1)
    gaussian_kernel = np.exp(-0.5 * (kernel_t / sigma) ** 2)
    tau = 0.5 * sfreq
    decay_len = int(4 * tau)
    decay_kernel = np.exp(-np.arange(decay_len) / tau)
    spike_kernel = np.convolve(gaussian_kernel, decay_kernel, mode="full")
    spike_kernel /= spike_kernel.max() + 1e-12
    half = kernel_half
    for loc, amp in zip(locations, amplitudes):
        start_in_kernel = 0
        start_in_out = loc - half
        if start_in_out < 0:
            start_in_kernel = -start_in_out
            start_in_out = 0
        end_in_out = min(loc - half + len(spike_kernel), n_samples)
        end_in_kernel = start_in_kernel + (end_in_out - start_in_out)
        out[start_in_out:end_in_out] += amp * spike_kernel[start_in_kernel:end_in_kernel]
    return out


def synthesize_electronic(
    n_samples: int, sfreq: float = 10.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    white = rng.standard_normal(n_samples)
    if sfreq <= 4.0:
        return white
    nyq = sfreq / 2.0
    cutoff = 2.0 / nyq
    cutoff = min(cutoff, 1.0 - 1e-6)
    sos = sp_signal.butter(4, cutoff, btype="highpass", output="sos")
    return sp_signal.sosfiltfilt(sos, white)


def inject_noise(
    clean: np.ndarray,
    snr_db: float,
    noise_types: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    if rng is None:
        rng = np.random.default_rng()
    if noise_types is None:
        noise_types = ALL_NOISE_TYPES

    _synthesizers = {
        "mayer": synthesize_mayer_wave,
        "respiratory": synthesize_respiratory,
        "cardiac": synthesize_cardiac,
        "motion": synthesize_motion_spikes,
        "electronic": synthesize_electronic,
    }

    n_channels, n_samples = clean.shape
    sfreq = 10.0

    enabled = {nt: bool(rng.random() < 0.8) for nt in noise_types}
    active_types = [nt for nt, on in enabled.items() if on]

    signal_power = np.mean(clean ** 2)
    if signal_power == 0:
        signal_power = 1.0
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))

    noisy = clean.copy()
    for ch in range(n_channels):
        if not active_types:
            break
        combined = np.zeros(n_samples)
        for nt in active_types:
            combined += _synthesizers[nt](n_samples, sfreq=sfreq, rng=rng)
        current_power = np.mean(combined ** 2)
        if current_power > 0:
            combined *= np.sqrt(target_noise_power / current_power)
        noisy[ch] += combined

    noise_info = {nt: enabled.get(nt, False) for nt in noise_types}
    noise_info["snr_db"] = snr_db
    return noisy, noise_info
