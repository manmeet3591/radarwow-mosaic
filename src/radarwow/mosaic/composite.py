"""Multi-radar compositing of per-radar rain fields on a common latitude/longitude grid, and radar-to-radar
reflectivity comparison in overlap areas.

Each radar first goes through the full single-radar chain (radarwow.pipeline.hourly_totals): QPE per radar
*before* mosaicking, as in the radar-wow build plan. The composite then picks or blends radars by a quality index:

- `quality_index`: Q = Q_height * Q_blockage. Q_blockage follows Zhang et al. 2011 (BAMS,
  doi:10.1175/2011BAMS-D-11-00047.1) as used by Crisologo et al. 2018: 1 for blockage <= 0.1, falling linearly to
  0 at 0.5. Q_height = exp(-(h / H)^2) for beam-centre height h above the radar, a height weighting of the kind
  used by Zhang, Howard & Gourley 2005 (JTECH, doi:10.1175/JTECH-1689.1); H = 1.5 km is a radar-wow choice.
- `composite(..., mode="max")`: the radar with the highest quality at each cell (BALTRAD's "max quality"
  composite, Michelson et al. 2018, J. Open Res. Softw., doi:10.5334/jors.193; in spirit, MRMS's lowest-beam
  hybrid scan, Qi & Zhang 2017).
- `composite(..., mode="weighted")`: quality-weighted mean of all radars that see the cell (BALTRAD's weighted
  composite), which avoids seams where the best radar changes.

Georeferencing: each radar's grid is in km east/north of the radar, with latitude/longitude from the same local
approximation as radarwow.pipeline (lat = lat0 + y / 111.2, lon = lon0 + x / (111.2 cos lat0)).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def mosaic_grid(radars, margin_km: float = 155.0, res_km: float = 1.0) -> xr.Dataset:
    """Regular latitude/longitude grid (about ``res_km`` spacing) covering every radar's disk."""
    lat0 = np.array([r.attrs["radar_lat"] for r in radars])
    lon0 = np.array([r.attrs["radar_lon"] for r in radars])
    clat = float(np.mean(lat0))
    dlat = res_km / 111.2
    dlon = res_km / (111.2 * np.cos(np.deg2rad(clat)))
    lats = np.arange(lat0.min() - margin_km / 111.2, lat0.max() + margin_km / 111.2, dlat)[::-1]
    lons = np.arange(lon0.min() - margin_km / (111.2 * np.cos(np.deg2rad(lat0.max()))),
                     lon0.max() + margin_km / (111.2 * np.cos(np.deg2rad(lat0.max()))), dlon)
    return xr.Dataset(coords={"lat": lats, "lon": lons})


def to_grid(field, rad: xr.Dataset, grid: xr.Dataset):
    """Nearest-cell sample of a radar-centric (y, x) field onto the lat/lon ``grid`` (NaN outside the radar's grid)."""
    lat0, lon0 = rad.attrs["radar_lat"], rad.attrs["radar_lon"]
    lon2, lat2 = np.meshgrid(grid["lon"].values, grid["lat"].values)
    xk = (lon2 - lon0) * 111.2 * np.cos(np.deg2rad(lat0))
    yk = (lat2 - lat0) * 111.2
    x, y = rad["x"].values, rad["y"].values
    j = np.round((xk - x[0]) / (x[1] - x[0])).astype(int)
    i = np.round((yk - y[0]) / (y[1] - y[0])).astype(int)
    ok = (i >= 0) & (i < y.size) & (j >= 0) & (j < x.size)
    out = np.full(lat2.shape, np.nan)
    out[ok] = np.asarray(field)[i[ok], j[ok]]
    return out


def quality_index(height_km, blockage, h_scale_km: float = 1.5):
    """Q in [0, 1] from beam-centre height above the radar and cumulative blockage (see module doc)."""
    b = np.nan_to_num(np.asarray(blockage, float), nan=1.0)
    qb = np.clip(1 - (b - 0.1) / 0.4, 0, 1)
    qh = np.exp(-(np.asarray(height_km, float) / h_scale_km) ** 2)
    return np.where(np.isfinite(height_km), qb * qh, 0.0)


def composite(fields, qualities, mode: str = "max"):
    """Combine co-located fields; returns (field, index of the best radar or -1, total quality)."""
    f = np.stack(fields)
    q = np.where(np.isfinite(f), np.stack(qualities), 0.0)
    best = np.where(q.max(axis=0) > 0, np.argmax(q, axis=0), -1)
    if mode == "max":
        out = np.take_along_axis(f, np.clip(best, 0, None)[None], axis=0)[0]
    elif mode == "weighted":
        out = np.nansum(np.nan_to_num(f) * q, axis=0) / np.where(q.sum(axis=0) > 0, q.sum(axis=0), np.nan)
    else:
        raise ValueError(mode)
    return np.where(best >= 0, out, np.nan), best, q.sum(axis=0)


def mosaic_hourly(radars, mode: str = "max", res_km: float = 1.0, h_scale_km: float = 1.5) -> xr.Dataset:
    """Composite of several `hourly_totals` outputs (same hours) on a common lat/lon grid."""
    grid = mosaic_grid(radars, res_km=res_km)
    q = [to_grid(quality_index(r["HEIGHT"].values, r["BLOCKAGE"].values, h_scale_km), r, grid) for r in radars]
    times = radars[0]["time"].values
    rain, best = [], None
    for t in times:
        f = [to_grid(r["RAIN"].sel(time=t).values, r, grid) for r in radars]
        m, best, _ = composite(f, [np.nan_to_num(x) for x in q], mode)
        rain.append(m)
    return xr.Dataset({"RAIN": (("time", "lat", "lon"), np.array(rain, dtype="float32"), {"units": "mm"}),
                       "SOURCE": (("lat", "lon"), best.astype("int8"), {"flag_meanings": " ".join(r.attrs["site"] for r in radars)})},
                      coords={"time": times, "lat": grid["lat"].values, "lon": grid["lon"].values},
                      attrs={"mode": mode, "radars": " ".join(r.attrs["site"] for r in radars)})


def overlap_difference(z_a, z_b, h_a, h_b, valid, max_dh_km: float = 0.5, z_min: float = 15.0):
    """Reflectivity difference a - b (dB) over co-located cells where both see rain at similar heights.

    Returns (median difference, interquartile range, number of cells). ``valid`` is an extra mask (e.g. below the
    melting layer, little attenuation on both paths)."""
    m = valid & np.isfinite(z_a) & np.isfinite(z_b) & (z_a >= z_min) & (z_b >= z_min) & (np.abs(h_a - h_b) <= max_dh_km)
    if m.sum() < 30:
        return np.nan, np.nan, int(m.sum())
    d = z_a[m] - z_b[m]
    q1, q3 = np.percentile(d, [25, 75])
    return float(np.median(d)), float(q3 - q1), int(m.sum())
