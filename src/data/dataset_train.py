import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.data.noise_synthesis import inject_noise


class FNIRSDenoiseDataset(Dataset):
    """Loads pre-computed clean windows from .npz files, injects noise on-the-fly.

    Args:
        npz_dir: path to directory of per-subject .npz files
        subject_ids: list of subject IDs to include (for train/val/test splitting)
        snr_range: (min_db, max_db) for uniform SNR randomization
        seed: random seed for reproducibility

    Each .npz has:
        'clean': shape (n_windows, 2, 128)
        'subject_id': str
        'ch_indices': array

    __getitem__ returns dict with:
        'noisy': torch.Tensor (2, 128)
        'clean': torch.Tensor (2, 128)
        'subject_id': str
        'snr_db': float
    """

    def __init__(
        self,
        npz_dir: str | Path,
        subject_ids: list[str],
        snr_range: tuple[float, float] = (-5.0, 20.0),
        seed: int = 42,
    ):
        self.snr_range = snr_range
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        npz_dir = Path(npz_dir)
        subject_set = set(subject_ids)

        self._data: list[np.ndarray] = []
        self._subject_labels: list[str] = []
        self._index: list[tuple[int, int]] = []

        for npz_path in sorted(npz_dir.glob("*.npz")):
            arr = np.load(npz_path, allow_pickle=True)
            sid = str(arr["subject_id"])
            if sid not in subject_set:
                continue
            clean = arr["clean"]
            subj_idx = len(self._data)
            self._data.append(clean)
            self._subject_labels.append(sid)
            for w in range(clean.shape[0]):
                self._index.append((subj_idx, w))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        subj_idx, win_idx = self._index[idx]
        clean = self._data[subj_idx][win_idx].copy().astype(np.float32)

        for c in range(clean.shape[0]):
            var = clean[c].var()
            if var >= 1e-12:
                clean[c] = (clean[c] - clean[c].mean()) / np.sqrt(var)

        snr_db = float(self._rng.uniform(self.snr_range[0], self.snr_range[1]))
        noisy, _ = inject_noise(clean, snr_db, rng=self._rng)

        return {
            "noisy": torch.from_numpy(noisy.astype(np.float32)),
            "clean": torch.from_numpy(clean),
            "subject_id": self._subject_labels[subj_idx],
            "snr_db": snr_db,
        }


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    ds = torch.utils.data.get_worker_info().dataset
    ds._rng = np.random.default_rng(worker_seed)


def build_subject_split(
    subject_ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Shuffle subjects with fixed seed, split into train/val/test.
    Returns {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    if save_path is not None:
        Path(save_path).write_text(json.dumps(split, indent=2))

    return split


def create_dataloaders(
    data_dir: str,
    split_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Load split JSON, create Dataset and DataLoader for each split.
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    split: dict[str, list[str]] = json.loads(Path(split_path).read_text())

    def _make_loader(split_name: str) -> DataLoader:
        ds = FNIRSDenoiseDataset(
            npz_dir=data_dir,
            subject_ids=split[split_name],
            seed=seed,
        )
        shuffle = split_name == "train"
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=lambda wid: _worker_init_fn(wid, seed),
            persistent_workers=num_workers > 0,
        )

    return {name: _make_loader(name) for name in ("train", "val", "test")}
