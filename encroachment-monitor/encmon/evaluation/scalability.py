"""Scalability analysis from district to national coverage (Table VIII).

Coverage is estimated from the *measured* per-km² processing cost of the
pipeline on the synthetic 5.12 km evaluation scene, extrapolated over
6.55 km² Sentinel-2 tiles with tile-parallel cloud execution, plus the
~4.85 MB per-tile multi-temporal archive record.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScaleRow:
    scope: str
    area_km2: int
    tiles: int
    core_hours: float
    wall_hours: float
    storage: str


def _fmt_bytes(gb: float) -> str:
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def scalability_table(sec_per_km2: float, workers: int, tile_km2: float = 6.55,
                      mb_per_tile: float = 4.85) -> list[ScaleRow]:
    """Extrapolate measured throughput to district / state / nation."""
    rows = []
    for scope, area in (
        ("District", 3_000),
        ("State", 130_000),
        ("Nation", 3_287_000),
    ):
        import math

        tiles = math.ceil(area / tile_km2)
        core_h = tiles * tile_km2 * sec_per_km2 / 3600.0
        wall_h = core_h / max(1, workers)
        storage_gb = tiles * mb_per_tile / 1024.0
        rows.append(
            ScaleRow(
                scope=scope,
                area_km2=area,
                tiles=tiles,
                core_hours=core_h,
                wall_hours=wall_h,
                storage=_fmt_bytes(storage_gb),
            )
        )
    return rows
