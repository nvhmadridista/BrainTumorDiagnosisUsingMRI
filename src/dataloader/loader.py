from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


NPY_EXTENSIONS: Tuple[str, ...] = (".npy",)


def _is_npy_file(path: Path) -> bool:
	return path.suffix.lower() in NPY_EXTENSIONS


def _default_numpy_loader(path: Path) -> np.ndarray:
	arr = np.load(str(path), allow_pickle=False)
	if not isinstance(arr, np.ndarray):
		raise TypeError(f"Loaded object is not a numpy array: {path}")
	return arr


def _to_channels_first(arr: np.ndarray) -> np.ndarray:
	# Accept (H, W), (H, W, C) or (C, H, W)
	if arr.ndim == 2:
		arr = arr[None, :, :]  # (1, H, W)
	elif arr.ndim == 3:
		c_first_likely = arr.shape[0] in (1, 3) and arr.shape[1] > 8 and arr.shape[2] > 8
		if c_first_likely:
			pass  # already (C, H, W)
		elif arr.shape[-1] in (1, 3):
			arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
		else:
			pass
	else:
		raise ValueError(
			f"Unsupported array shape {arr.shape}. Expected 2D or 3D arrays."
		)

	if arr.dtype != np.float32:
		arr = arr.astype(np.float32, copy=False)
	return arr


class NpyFolderDataset(Dataset):
	"""A dataset that mirrors torchvision's ImageFolder for .npy files.
	"""

	def __init__(
		self,
		root: Path | str,
		transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		target_transform: Optional[Callable[[int], int]] = None,
		loader: Callable[[Path], np.ndarray] = _default_numpy_loader,
	) -> None:
		super().__init__()
		self.root = Path(root)
		if not self.root.exists():
			raise FileNotFoundError(f"Dataset root not found: {self.root}")

		classes = [d.name for d in self.root.iterdir() if d.is_dir()]
		if not classes:
			raise RuntimeError(f"No class subfolders found under: {self.root}")
		classes.sort()
		self.classes: List[str] = classes
		self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

		samples: List[Tuple[Path, int]] = []
		for cls in self.classes:
			cls_dir = self.root / cls
			for p in cls_dir.rglob("*"):
				if p.is_file() and _is_npy_file(p):
					samples.append((p, self.class_to_idx[cls]))

		if not samples:
			raise RuntimeError(f"No .npy files found under: {self.root}")

		self.samples = samples
		self.loader = loader
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self) -> int:  
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  
		path, target = self.samples[index]
		arr = self.loader(path)
		arr = _to_channels_first(arr)
		tensor = torch.from_numpy(arr)  

		if self.transform is not None:
			tensor = self.transform(tensor)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return tensor, target


def _default_data_root() -> Path:
	here = Path(__file__).resolve()
	repo_root = here.parents[2]
	return repo_root / "dataset_imagenet"


def get_data_loaders(
	data_root: Optional[Path | str] = None,
	batch_size: int = 32,
	num_workers: Optional[int] = None,
	shuffle_train: bool = True,
	pin_memory: Optional[bool] = None,
	persistent_workers: Optional[bool] = None,
	drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
	"""Create train/val/test loaders and return class names.

	Returns (train_loader, val_loader, test_loader, classes)
	"""

	root = Path(data_root) if data_root is not None else _default_data_root()

	train_dir = root / "train"
	val_dir = root / "val"
	test_dir = root / "test"

	if num_workers is None:
		try:
			cpus = os.cpu_count() or 2
		except Exception:
			cpus = 2
		num_workers = max(2, cpus // 2)

	if pin_memory is None:
		pin_memory = torch.cuda.is_available()

	if persistent_workers is None:
		persistent_workers = num_workers > 0

	train_dataset = NpyFolderDataset(train_dir)
	val_dataset = NpyFolderDataset(val_dir)
	test_dataset = NpyFolderDataset(test_dir)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle_train,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=drop_last,
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=False,
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=False,
	)

	return train_loader, val_loader, test_loader, train_dataset.classes