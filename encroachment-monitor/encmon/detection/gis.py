"""GIS cadastral overlay (paper Sec. Data/GIS Integration, Table IV).

The overlay intersects the new-construction mask with the protected-parcel
layer -- the step that converts "new construction" into "encroachment on
government land".  Geometry is computed with shapely when available and
falls back to a pure-Python ray-casting engine otherwise.
"""
from __future__ import annotations

from ..datagen import parcels as pgeo  # pure-python geometry helpers

try:  # optional acceleration
    from shapely.geometry import Point as _ShPoint
    from shapely.geometry import Polygon as _ShPolygon
    from shapely.prepared import prep as _prep

    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    HAS_SHAPELY = False


class ProtectedLayer:
    """Protected-parcel layer with point-in-polygon and raster overlay."""

    def __init__(self, parcels: list[dict]):
        self.parcels = [p for p in parcels if p["protected"]]
        self._prep = None
        if HAS_SHAPELY:
            self._geoms = [(p["id"], _ShPolygon(p["ring"])) for p in self.parcels]

    # -- point queries ---------------------------------------------------
    def contains_point(self, x: float, y: float):
        """Return (is_protected, parcel_id | None) for a metric point."""
        if HAS_SHAPELY:
            pt = _ShPoint(x, y)
            for pid, g in self._geoms:
                if g.covers(pt):
                    return True, pid
            return False, None
        for p in self.parcels:
            if pgeo.point_in_ring(x, y, p["ring"]):
                return True, p["id"]
        return False, None

    # -- raster overlay ----------------------------------------------------
    def raster_mask(self, xs, ys) -> "np.ndarray":
        """Boolean mask over the pixel-centre coordinate grids."""
        import numpy as np

        mask = np.zeros(xs.shape, dtype=bool)
        if HAS_SHAPELY:
            minx, miny, maxx, maxy = (
                min(c[0] for g in self._geoms for c in g[1].exterior.coords),
                min(c[1] for g in self._geoms for c in g[1].exterior.coords),
                max(c[0] for g in self._geoms for c in g[1].exterior.coords),
                max(c[1] for g in self._geoms for c in g[1].exterior.coords),
            )
            cand = (ys >= miny) & (ys <= maxy) & (xs >= minx) & (xs <= maxx)
            idx = np.argwhere(cand)
            for i, j in idx:
                pt = _ShPoint(xs[i, j], ys[i, j])
                for pid, g in self._geoms:
                    if g.covers(pt):
                        mask[i, j] = True
                        break
            return mask
        for p in self.parcels:
            ring = p["ring"]
            xs_r = [c[0] for c in ring]
            ys_r = [c[1] for c in ring]
            cand = (
                (xs >= min(xs_r)) & (xs <= max(xs_r)) & (ys >= min(ys_r)) & (ys <= max(ys_r))
            )
            idx = np.argwhere(cand)
            for i, j in idx:
                if pgeo.point_in_ring(xs[i, j], ys[i, j], ring):
                    mask[i, j] = True
        return mask

    def total_area_m2(self) -> float:
        return sum(p["area_m2"] for p in self.parcels)
