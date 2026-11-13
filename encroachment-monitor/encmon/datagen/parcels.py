"""Cadastral parcel layer (the GIS protected-zone prior, paper Table IV).

A demo AOI near Coimbatore is laid out as a jittered grid of rectangular
parcels in a local UTM-aligned metric frame (x: easting m, y: northing m).
A random subset is marked protected (government land).  Parcels export to
GeoJSON in WGS-84 for the dashboard.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import (
    AOI_CENTER_LATLON,
    PARCEL_GRID,
    PARCEL_SIZE_M,
    PROTECTED_FRACTION,
)


def meters_per_degree(lat: float) -> tuple[float, float]:
    """Local scale factors (m/deg lon, m/deg lat) at a given latitude."""
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lon, m_per_deg_lat


class Aoi:
    """Local metric frame anchored at the AOI centre (WGS-84 <-> metres)."""

    def __init__(self, lat: float, lon: float, side_m: float):
        self.lat, self.lon = lat, lon
        self.side_m = side_m
        self.mx, self.my = meters_per_degree(lat)
        self.x0 = -side_m / 2.0   # metric frame spans [-side/2, side/2]^2
        self.y0 = -side_m / 2.0

    def to_latlon(self, x: np.ndarray, y: np.ndarray):
        lat = self.lat + (y - self.y0) / self.my
        lon = self.lon + (x - self.x0) / self.mx
        return lat, lon

    @property
    def side_deg(self) -> tuple[float, float]:
        return self.side_m / self.mx, self.side_m / self.my


def _rect_polygon(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def generate_parcels(seed: int, aoi: Aoi | None = None):
    """Generate the parcel cadastre.

    Returns (parcels, aoi) where each parcel is a dict:
      id, name, protected (bool), metric ring [[x, y], ...], area_m2.
    """
    rng = np.random.default_rng(seed + 7)
    side = PARCEL_GRID * PARCEL_SIZE_M
    if aoi is None:
        aoi = Aoi(*AOI_CENTER_LATLON, side)
    parcels = []
    k = 0
    for iy in range(PARCEL_GRID):
        for ix in range(PARCEL_GRID):
            # jittered rectangle inside its grid cell (cadastral irregularity)
            jw = PARCEL_SIZE_M * float(rng.uniform(0.82, 0.96))
            jh = PARCEL_SIZE_M * float(rng.uniform(0.82, 0.96))
            jx = ix * PARCEL_SIZE_M + rng.uniform(0, PARCEL_SIZE_M - jw) + aoi.x0
            jy = iy * PARCEL_SIZE_M + rng.uniform(0, PARCEL_SIZE_M - jh) + aoi.y0
            ring = _rect_polygon(jx, jy, jw, jh)
            k += 1
            parcels.append(
                dict(
                    id=f"P-{k:04d}",
                    name=f"Parcel P-{k:04d}",
                    protected=bool(rng.random() < PROTECTED_FRACTION),
                    ring=ring,
                    area_m2=jw * jh,
                )
            )
    return parcels, aoi


def point_in_ring(px: float, py: float, ring) -> bool:
    """Ray-casting point-in-polygon (pure Python, no dependencies)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > py) != (yj > py):
            x_int = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_int:
                inside = not inside
        j = i
    return inside


def enclosing_parcel(parcels, px: float, py: float):
    """Return the parcel containing (px, py) or None."""
    for p in parcels:
        if point_in_ring(px, py, p["ring"]):
            return p
    return None


def protected_rings(parcels):
    return [p["ring"] for p in parcels if p["protected"]]
