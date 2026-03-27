import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def compute_band_power_envelope(
    raw_eeg: mne.io.Raw,
    band: tuple[float, float] = (30.0, 100.0),
    lp_freq: float = 0.5,
) -> mne.io.Raw:
    raw = raw_eeg.copy().pick_types(eeg=True)
    raw.filter(band[0], band[1])
    raw.apply_hilbert(envelope=True)
    raw.filter(None, lp_freq)
    return raw


def get_fnirs_channel_positions(raw_fnirs: mne.io.Raw) -> dict[str, np.ndarray]:
    hbo_indices = mne.pick_types(raw_fnirs.info, fnirs="hbo")
    positions = {}
    for idx in hbo_indices:
        ch = raw_fnirs.info["chs"][idx]
        src = ch["loc"][:3]
        det = ch["loc"][3:6]
        if np.all(src == 0) and np.all(det == 0):
            continue
        positions[ch["ch_name"]] = (src + det) / 2.0
    return positions


def match_eeg_to_fnirs(
    raw_eeg: mne.io.Raw,
    raw_fnirs: mne.io.Raw,
    radius_mm: float = 30.0,
) -> dict[str, list[str]]:
    fnirs_positions = get_fnirs_channel_positions(raw_fnirs)
    radius_m = radius_mm / 1000.0

    montage = raw_eeg.get_montage()
    if montage is not None:
        ch_pos = montage.get_positions()["ch_pos"]
        eeg_positions = {name: pos for name, pos in ch_pos.items() if name in raw_eeg.ch_names}
    else:
        eeg_positions = {}
        for ch in raw_eeg.info["chs"]:
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
                eeg_positions[ch["ch_name"]] = ch["loc"][:3]

    eeg_names = list(eeg_positions.keys())
    eeg_coords = np.array([eeg_positions[n] for n in eeg_names])

    matching: dict[str, list[str]] = {}
    for fnirs_ch, fnirs_pos in fnirs_positions.items():
        dists = np.linalg.norm(eeg_coords - fnirs_pos, axis=1)
        within = [eeg_names[i] for i, d in enumerate(dists) if d <= radius_m]
        if within:
            matching[fnirs_ch] = within
        else:
            nearest = eeg_names[int(np.argmin(dists))]
            logger.warning(
                "No EEG electrodes within %.1f mm of fNIRS channel %s; "
                "falling back to nearest electrode %s (%.1f mm)",
                radius_mm,
                fnirs_ch,
                nearest,
                np.min(dists) * 1000.0,
            )
            matching[fnirs_ch] = [nearest]

    return matching


def average_matched_envelopes(
    envelope_data: np.ndarray,
    eeg_ch_names: list[str],
    matching: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    name_to_idx = {name: i for i, name in enumerate(eeg_ch_names)}
    fnirs_ch_names = list(matching.keys())
    averaged = np.zeros((len(fnirs_ch_names), envelope_data.shape[1]), dtype=envelope_data.dtype)
    for row, fnirs_ch in enumerate(fnirs_ch_names):
        indices = [name_to_idx[n] for n in matching[fnirs_ch] if n in name_to_idx]
        if indices:
            averaged[row] = envelope_data[indices].mean(axis=0)
    return averaged, fnirs_ch_names
