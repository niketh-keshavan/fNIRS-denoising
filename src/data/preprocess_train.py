import argparse
import logging
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
    source_detector_distances,
    temporal_derivative_distribution_repair as tddr,
)

logger = logging.getLogger(__name__)

_MIN_SCI = 0.5
_MIN_SD_DIST_M = 0.01
_MIN_GOOD_CH = 4
_MIN_WINDOWS = 3
_RESAMPLE_FREQ = 10.0
_LP_FREQ = 0.5
_HP_FREQ = 0.01


def _drop_bad_channels(raw_od, raw_haemo):
    sci = scalp_coupling_index(raw_od)
    bad_sci = [ch for ch, s in zip(raw_od.ch_names, sci) if s < _MIN_SCI]
    if bad_sci:
        logger.warning("Dropping %d channels with SCI < %.2f: %s", len(bad_sci), _MIN_SCI, bad_sci)

    dists = source_detector_distances(raw_haemo.info)
    bad_dist = [raw_haemo.ch_names[i] for i, d in enumerate(dists) if d < _MIN_SD_DIST_M]
    if bad_dist:
        logger.warning("Dropping %d short-separation channels: %s", len(bad_dist), bad_dist)

    bad_haemo_from_sci = []
    for ch in bad_sci:
        base = ch.rsplit(" ", 1)[0]
        matches = [c for c in raw_haemo.ch_names if c.startswith(base)]
        bad_haemo_from_sci.extend(matches)

    all_bad = list(set(bad_haemo_from_sci + bad_dist))
    raw_haemo.drop_channels([c for c in all_bad if c in raw_haemo.ch_names])
    return raw_haemo


def _drop_nan_inf_channels(raw_haemo):
    data = raw_haemo.get_data()
    bad_idx = [i for i in range(data.shape[0]) if not np.isfinite(data[i]).all()]
    if bad_idx:
        names = [raw_haemo.ch_names[i] for i in bad_idx]
        logger.warning("Dropping %d channels with NaN/Inf after Beer-Lambert: %s", len(names), names)
        raw_haemo.drop_channels(names)
    return raw_haemo


def preprocess_subject(snirf_path: str, is_bids: bool = False) -> dict | None:
    try:
        if is_bids:
            from mne_bids import read_raw_bids
            raw = read_raw_bids(snirf_path, verbose=False)
        else:
            raw = mne.io.read_raw_snirf(snirf_path, verbose=False)
        raw.load_data()
    except Exception as exc:
        logger.warning("Failed to load %s: %s", snirf_path, exc)
        return None

    raw_od = optical_density(raw)
    raw_od = tddr(raw_od)
    raw_haemo = beer_lambert_law(raw_od)

    raw_haemo = _drop_bad_channels(raw_od, raw_haemo)
    raw_haemo = _drop_nan_inf_channels(raw_haemo)

    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    n_good = min(len(hbo_picks), len(hbr_picks))

    if n_good < _MIN_GOOD_CH:
        logger.warning("Skipping %s: only %d good channel pairs after gating", snirf_path, n_good)
        return None

    raw_haemo.filter(_HP_FREQ, _LP_FREQ, verbose=False)
    raw_haemo.resample(_RESAMPLE_FREQ, verbose=False)

    hbo = raw_haemo.get_data(picks=hbo_picks)
    hbr = raw_haemo.get_data(picks=hbr_picks)
    ch_names_hbo = [raw_haemo.ch_names[i] for i in hbo_picks]

    return {
        "hbo": hbo,
        "hbr": hbr,
        "sfreq": _RESAMPLE_FREQ,
        "ch_names": ch_names_hbo,
    }


def window_signals(hbo: np.ndarray, hbr: np.ndarray, win_size: int = 128, stride: int = 64) -> np.ndarray:
    n_ch, n_samples = hbo.shape
    windows = []

    for ch in range(n_ch):
        start = 0
        while start + win_size <= n_samples:
            h_win = hbo[ch, start:start + win_size]
            r_win = hbr[ch, start:start + win_size]
            if h_win.var() >= 1e-12 and r_win.var() >= 1e-12:
                windows.append(np.stack([h_win, r_win], axis=0))
            start += stride

    if not windows:
        return np.empty((0, 2, win_size), dtype=np.float32)

    return np.stack(windows, axis=0).astype(np.float32)


def _ch_indices_for_windows(hbo: np.ndarray, hbr: np.ndarray, win_size: int = 128, stride: int = 64) -> np.ndarray:
    n_ch, n_samples = hbo.shape
    indices = []
    for ch in range(n_ch):
        start = 0
        while start + win_size <= n_samples:
            h_win = hbo[ch, start:start + win_size]
            r_win = hbr[ch, start:start + win_size]
            if h_win.var() >= 1e-12 and r_win.var() >= 1e-12:
                indices.append(ch)
            start += stride
    return np.array(indices, dtype=np.int32)


def _subject_id_from_path(path) -> str:
    p = Path(str(path))
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return p.stem


def process_dataset(dataset_dir: str, output_dir: str, dataset_type: str = "snirf") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if dataset_type == "bids":
        from mne_bids import BIDSPath, get_entity_vals
        bids_root = Path(dataset_dir)
        subjects = get_entity_vals(bids_root, "subject")
        paths = []
        for sub in subjects:
            bp = BIDSPath(subject=sub, root=bids_root, suffix="nirs", extension=".snirf")
            matches = bp.match()
            paths.extend(matches)
        is_bids = True
    else:
        bids_root = Path(dataset_dir)
        paths = list(bids_root.rglob("*.snirf"))
        is_bids = False

    if not paths:
        logger.warning("No files found in %s for dataset_type=%s", dataset_dir, dataset_type)
        return

    for path in paths:
        subject_id = _subject_id_from_path(path)
        result = preprocess_subject(path if is_bids else str(path), is_bids=is_bids)

        if result is None:
            continue

        windows = window_signals(result["hbo"], result["hbr"])

        if len(windows) < _MIN_WINDOWS:
            logger.warning("Skipping %s: only %d windows produced", subject_id, len(windows))
            continue

        ch_indices = _ch_indices_for_windows(result["hbo"], result["hbr"])

        npz_path = out_path / f"{subject_id}.npz"
        np.savez(
            npz_path,
            clean=windows,
            subject_id=subject_id,
            ch_indices=ch_indices,
        )
        logger.info("Saved %s: %d windows → %s", subject_id, len(windows), npz_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess fNIRS training data")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-type", choices=["snirf", "bids"], default="snirf")
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_dir, args.dataset_type)
