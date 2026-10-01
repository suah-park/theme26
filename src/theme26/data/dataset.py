"""Memory-mapped train/evaluation crop generation under the frozen data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from theme26.contracts import CropSpec, SamplingSpec
from theme26.data.crop_sampler import extract_aligned_crop
from theme26.data.raw_volume import normalize_grayscale, to_pore_mask
from theme26.data.sparse_sampling import SparseSample, linear_interpolate_z, make_sparse_sample


RAW_SHAPE_XYZ = (1000, 1000, 1000)


@dataclass(frozen=True)
class CoreSource:
    core_id: str
    split: str
    grayscale_path: Path
    binary_path: Path
    source_pore_value: int = 0
    shape_xyz: tuple[int, int, int] = RAW_SHAPE_XYZ


@dataclass(frozen=True)
class PreprocessedCrop:
    core_id: str
    split: str
    crop: CropSpec
    grayscale: np.ndarray
    pore_mask: np.ndarray
    sparse: SparseSample
    e0_linear: np.ndarray


def _open_uint8_memmap(path: Path, shape_xyz: tuple[int, int, int]) -> np.memmap:
    expected = int(np.prod(shape_xyz, dtype=np.int64))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} uint8 bytes, found {path.stat().st_size}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape_xyz, order="C")


def open_core_memmaps(source: CoreSource) -> tuple[np.memmap, np.memmap]:
    """Open source RAW arrays read-only; no source copy is created."""

    return _open_uint8_memmap(source.grayscale_path, source.shape_xyz), _open_uint8_memmap(source.binary_path, source.shape_xyz)


def preprocess_crop(
    source: CoreSource,
    *,
    crop: CropSpec,
    normalization_lower: float,
    normalization_upper: float,
    sampling: SamplingSpec = SamplingSpec(),
) -> PreprocessedCrop:
    """Materialize exactly one aligned crop from read-only RAW memory maps."""

    crop.validate_inside(source.shape_xyz)
    raw_gray, raw_binary = open_core_memmaps(source)
    x0, y0, z0 = crop.origin_xyz
    sx, sy, sz = crop.shape_xyz
    region = np.s_[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz]
    grayscale = normalize_grayscale(np.asarray(raw_gray[region]), lower=normalization_lower, upper=normalization_upper)
    pore = to_pore_mask(np.asarray(raw_binary[region]), source_pore_value=source.source_pore_value)
    grayscale, pore = extract_aligned_crop(grayscale, pore, crop=CropSpec((0, 0, 0), crop.shape_xyz))
    sparse = make_sparse_sample(grayscale, sampling=sampling)
    return PreprocessedCrop(source.core_id, source.split, crop, grayscale, pore, sparse, linear_interpolate_z(sparse))


class TrainRandomCropDataset:
    """On-demand random aligned crops from train parent cores only."""

    def __init__(
        self,
        sources: Sequence[CoreSource],
        *,
        normalization_lower: float,
        normalization_upper: float,
        crop_shape_xyz: tuple[int, int, int] = (256, 256, 256),
        sampling: SamplingSpec = SamplingSpec(),
    ) -> None:
        if not sources or any(source.split != "train" for source in sources):
            raise ValueError("TrainRandomCropDataset requires one or more train-only sources")
        self.sources = tuple(sources)
        self.normalization_lower = normalization_lower
        self.normalization_upper = normalization_upper
        self.crop_shape_xyz = crop_shape_xyz
        self.sampling = sampling

    def sample(self, rng: np.random.Generator) -> PreprocessedCrop:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        source = self.sources[int(rng.integers(len(self.sources)))]
        origin = tuple(int(rng.integers(0, limit - size + 1)) for size, limit in zip(self.crop_shape_xyz, source.shape_xyz))
        return preprocess_crop(
            source, crop=CropSpec(origin, self.crop_shape_xyz), normalization_lower=self.normalization_lower,
            normalization_upper=self.normalization_upper, sampling=self.sampling,
        )
