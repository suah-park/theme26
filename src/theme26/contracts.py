"""Dependency-free shared contracts used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ShapeXYZ: TypeAlias = tuple[int, int, int]
SpacingXYZ: TypeAlias = tuple[float, float, float]
AxisOrder: TypeAlias = Literal["XYZ"]
StorageOrder: TypeAlias = Literal["XYZ", "ZYX"]

AXIS_ORDER: AxisOrder = "XYZ"
SPARSE_AXIS = 2
PORE_VALUE = 1
DEFAULT_CROP_SHAPE: ShapeXYZ = (256, 256, 256)


def _positive_triplet(name: str, value: tuple[int | float, int | float, int | float]) -> None:
    if len(value) != 3 or any(item <= 0 for item in value):
        raise ValueError(f"{name} must contain three positive values, got {value!r}")


@dataclass(frozen=True)
class VolumeMetadata:
    """Metadata required to interpret one raw volume without guessing."""

    sample_id: str
    source_core_id: str
    shape_xyz: ShapeXYZ
    voxel_spacing_um_xyz: SpacingXYZ
    dtype: str
    endianness: Literal["little", "big", "not-applicable"]
    normalization_id: str | None = None
    axis_order: AxisOrder = AXIS_ORDER
    storage_order: StorageOrder = AXIS_ORDER

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source_core_id:
            raise ValueError("sample_id and source_core_id must be non-empty")
        _positive_triplet("shape_xyz", self.shape_xyz)
        _positive_triplet("voxel_spacing_um_xyz", self.voxel_spacing_um_xyz)
        if self.endianness not in ("little", "big", "not-applicable"):
            raise ValueError("endianness must be 'little', 'big', or 'not-applicable'")
        if self.axis_order != AXIS_ORDER:
            raise ValueError(f"axis_order must be {AXIS_ORDER!r}")
        if self.storage_order not in ("XYZ", "ZYX"):
            raise ValueError("storage_order must be 'XYZ' or 'ZYX'")


@dataclass(frozen=True)
class CropSpec:
    """Aligned spatial crop in canonical XYZ coordinates."""

    origin_xyz: tuple[int, int, int]
    shape_xyz: ShapeXYZ = DEFAULT_CROP_SHAPE

    def __post_init__(self) -> None:
        if len(self.origin_xyz) != 3 or any(item < 0 for item in self.origin_xyz):
            raise ValueError(f"origin_xyz must contain three non-negative values, got {self.origin_xyz!r}")
        _positive_triplet("shape_xyz", self.shape_xyz)

    def validate_inside(self, volume_shape_xyz: ShapeXYZ) -> None:
        """Raise when this crop does not fit inside the given volume."""

        _positive_triplet("volume_shape_xyz", volume_shape_xyz)
        if any(origin + size > limit for origin, size, limit in zip(self.origin_xyz, self.shape_xyz, volume_shape_xyz)):
            raise ValueError(
                f"crop origin={self.origin_xyz}, shape={self.shape_xyz} exceeds volume={volume_shape_xyz}"
            )


@dataclass(frozen=True)
class SamplingSpec:
    """Sparse-z sampling rule shared by preprocessing, models and metrics."""

    factor: int = 4
    offset: int = 0
    axis: int = SPARSE_AXIS
    boundary_mode: Literal["replicate"] = "replicate"

    def __post_init__(self) -> None:
        if self.factor < 2:
            raise ValueError("factor must be at least 2")
        if not 0 <= self.offset < self.factor:
            raise ValueError("offset must satisfy 0 <= offset < factor")
        if self.axis != SPARSE_AXIS:
            raise ValueError(f"primary protocol requires sparse axis {SPARSE_AXIS} (Z)")

    def observed_indices(self, z_size: int) -> tuple[int, ...]:
        """Return HR-grid indices retained by this sampling rule."""

        if z_size <= self.offset:
            raise ValueError("z_size must be greater than offset")
        return tuple(range(self.offset, z_size, self.factor))

    def bracketed_missing_indices(self, z_size: int) -> tuple[int, ...]:
        """Return missing indices that have observed slices on both sides."""

        observed = self.observed_indices(z_size)
        return tuple(
            index
            for left, right in zip(observed, observed[1:])
            for index in range(left + 1, right)
        )
