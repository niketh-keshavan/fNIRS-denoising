import argparse
import logging
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing.nirs import beer_lambert_law, optical_density
from scipy import signal as sp_signal
from scipy.stats import pearsonr

from src.data.preprocess_train import (
    _drop_bad_channels,
    _drop_nan_inf_channels,
    _subject_id_from_path,
)
from src.data.eeg_correspondence import (
    average_matched_envelopes,
    compute_band_power_envelope,
    match_eeg_to_fnirs,
)

logger = logging.getLogger(__name__)

_MIN_OVERLAP_S = 60.0
_MIN_GOOD_CH = 2


def _fnirs_minimal_preprocess(raw: mne.io.Raw) -> tuple[mne.io.Raw, mne.io.Raw]:
    raw_od = optical_density(raw)
    raw_haemo = beer_lambert_law(raw_od)
    return raw_od, raw_haemo


def _align_signals(
    fnirs_data: np.ndarray,
    eeg_data: np.ndarray,
    fnirs_sfreq: float,
    eeg_sfreq: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    overlap_s = min(fnirs_data.shape[1] / fnirs_sfreq, eeg_data.shape[1] / eeg_sfreq)
    if overlap_s < _MIN_OVERLAP_S:
        raise ValueError(
            f"Overlap {overlap_s:.1f}s < minimum {_MIN_OVERLAP_S}s"
        )

    target_n_samples = int(overlap_s * fnirs_sfreq)
    eeg_resampled = sp_signal.resample(eeg_data, target_n_samples, axis=1)
    fnirs_trimmed = fnirs_data[:, :target_n_samples]
    return fnirs_trimmed, eeg_resampled, fnirs_sfreq


def _compute_raw_correlations(
    hbo: np.ndarray,
    hbr: np.ndarray,
    eeg_env: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_ch = hbo.shape[0]
    r_hbo = np.zeros(n_ch, dtype=np.float32)
    r_hbr = np.zeros(n_ch, dtype=np.float32)

    for i in range(n_ch):
        eeg_ch = eeg_env[i] if i < eeg_env.shape[0] else eeg_env[-1]

        if np.std(hbo[i]) < 1e-12 or np.std(eeg_ch) < 1e-12:
            r_hbo[i] = 0.0
        else:
            r_hbo[i] = pearsonr(hbo[i], eeg_ch)[0]

        if np.std(hbr[i]) < 1e-12 or np.std(eeg_ch) < 1e-12:
            r_hbr[i] = 0.0
        else:
            r_hbr[i] = pearsonr(hbr[i], eeg_ch)[0]

    return r_hbo, r_hbr


def preprocess_test_subject(nirs_path, eeg_path, is_bids: bool = True) -> dict | None:
    try:
        if is_bids:
            from mne_bids import read_raw_bids
            raw_fnirs = read_raw_bids(nirs_path, verbose=False)
        else:
            raw_fnirs = mne.io.read_raw_snirf(str(nirs_path), verbose=False)
        raw_fnirs.load_data()
    except Exception as exc:
        logger.warning("Failed to load fNIRS %s: %s", nirs_path, exc)
        return None

    raw_od, raw_haemo = _fnirs_minimal_preprocess(raw_fnirs)
    raw_haemo = _drop_bad_channels(raw_od, raw_haemo)
    raw_haemo = _drop_nan_inf_channels(raw_haemo)

    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    if len(hbo_picks) < _MIN_GOOD_CH:
        logger.warning(
            "Skipping %s: only %d HbO channels after quality gating", nirs_path, len(hbo_picks)
        )
        return None

    try:
        if is_bids:
            from mne_bids import read_raw_bids
            raw_eeg = read_raw_bids(eeg_path, verbose=False)
        else:
            raw_eeg = mne.io.read_raw_bdf(str(eeg_path), verbose=False, preload=False)
        raw_eeg.load_data()
    except Exception as exc:
        logger.warning("Failed to load EEG %s: %s", eeg_path, exc)
        return None

    envelope_raw = compute_band_power_envelope(raw_eeg)
    eeg_sfreq = envelope_raw.info["sfreq"]
    eeg_data = envelope_raw.get_data()
    eeg_ch_names = envelope_raw.ch_names

    matching = match_eeg_to_fnirs(raw_eeg, raw_haemo)
    eeg_env_matched, matched_fnirs_names = average_matched_envelopes(
        eeg_data, eeg_ch_names, matching
    )

    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    hbo = raw_haemo.get_data(picks=hbo_picks)
    hbr = raw_haemo.get_data(picks=hbr_picks)
    ch_names_hbo = [raw_haemo.ch_names[i] for i in hbo_picks]
    fnirs_sfreq = raw_haemo.info["sfreq"]

    fnirs_ch_to_idx = {name: i for i, name in enumerate(ch_names_hbo)}
    ordered_indices = []
    ordered_fnirs_names = []
    for name in matched_fnirs_names:
        base = name.replace(" hbo", "").replace(" hbr", "")
        hbo_name = f"{base} hbo"
        if hbo_name in fnirs_ch_to_idx:
            ordered_indices.append(fnirs_ch_to_idx[hbo_name])
            ordered_fnirs_names.append(hbo_name)

    if not ordered_indices:
        ordered_indices = list(range(len(ch_names_hbo)))
        ordered_fnirs_names = ch_names_hbo
        eeg_env_matched = eeg_env_matched[: len(ordered_indices)]

    hbo_ordered = hbo[ordered_indices]
    hbr_picks_names = [raw_haemo.ch_names[i] for i in hbr_picks]
    hbr_reordered = []
    for hbo_name in ordered_fnirs_names:
        hbr_name = hbo_name.replace(" hbo", " hbr")
        if hbr_name in hbr_picks_names:
            hbr_reordered.append(hbr[hbr_picks_names.index(hbr_name)])
        else:
            hbr_reordered.append(np.zeros(hbr.shape[1]))
    hbr_ordered = np.array(hbr_reordered)

    n_ch = min(hbo_ordered.shape[0], eeg_env_matched.shape[0])
    hbo_ordered = hbo_ordered[:n_ch]
    hbr_ordered = hbr_ordered[:n_ch]
    eeg_env_matched = eeg_env_matched[:n_ch]

    try:
        hbo_aligned, eeg_aligned, sfreq = _align_signals(
            hbo_ordered, eeg_env_matched, fnirs_sfreq, eeg_sfreq
        )
    except ValueError as exc:
        logger.warning("Alignment failed for %s: %s", nirs_path, exc)
        return None

    hbr_aligned = hbr_ordered[:, : hbo_aligned.shape[1]]

    r_hbo, r_hbr = _compute_raw_correlations(hbo_aligned, hbr_aligned, eeg_aligned)

    subject_id = _subject_id_from_path(nirs_path)

    return {
        "hbo": hbo_aligned,
        "hbr": hbr_aligned,
        "eeg_envelope": eeg_aligned,
        "sfreq": sfreq,
        "ch_names": ordered_fnirs_names,
        "r_raw_hbo": r_hbo,
        "r_raw_hbr": r_hbr,
        "subject_id": subject_id,
    }


def process_test_dataset(dataset_dir: str, output_dir: str) -> None:
    from mne_bids import BIDSPath, get_entity_vals

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    subjects = get_entity_vals(Path(dataset_dir), "subject")
    if not subjects:
        logger.warning("No subjects found in %s", dataset_dir)
        return

    for sub in subjects:
        nirs_bp = BIDSPath(subject=sub, root=dataset_dir, suffix="nirs", datatype="nirs")
        eeg_bp = BIDSPath(subject=sub, root=dataset_dir, suffix="eeg", datatype="eeg")

        nirs_matches = nirs_bp.match()
        eeg_matches = eeg_bp.match()

        if not nirs_matches:
            logger.warning("No fNIRS files found for sub-%s", sub)
            continue
        if not eeg_matches:
            logger.warning("No EEG files found for sub-%s", sub)
            continue

        nirs_path = nirs_matches[0]
        eeg_path = eeg_matches[0]

        result = preprocess_test_subject(nirs_path, eeg_path, is_bids=True)
        if result is None:
            continue

        subject_id = result["subject_id"]
        npz_path = out_path / f"{subject_id}.npz"
        np.savez(
            npz_path,
            hbo=result["hbo"],
            hbr=result["hbr"],
            eeg_envelope=result["eeg_envelope"],
            sfreq=result["sfreq"],
            ch_names=result["ch_names"],
            r_raw_hbo=result["r_raw_hbo"],
            r_raw_hbr=result["r_raw_hbr"],
            subject_id=subject_id,
        )
        logger.info("Saved %s → %s", subject_id, npz_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess fNIRS+EEG test data (ds004514)")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    process_test_dataset(args.dataset_dir, args.output_dir)
