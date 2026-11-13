import numpy as np
import xarray as xr

from radarwow.mosaic import composite, mosaic_grid, overlap_difference, quality_index, to_grid


def _radar(lat0, lon0, value):
    x = np.arange(-150, 150) + 0.5
    xx, yy = np.meshgrid(x, x[::-1])
    d = np.hypot(xx, yy)
    rain = np.where(d <= 150, value, np.nan)
    return xr.Dataset({"RAIN": (("time", "y", "x"), rain[None].astype("float32")), "HEIGHT": (("y", "x"), np.where(d <= 150, d / 100, np.nan)),
                       "BLOCKAGE": (("y", "x"), np.zeros_like(d))}, coords={"time": [np.datetime64("2023-08-30T12:00")], "y": x[::-1], "x": x},
                      attrs={"radar_lat": lat0, "radar_lon": lon0, "site": f"r{value}"})


def test_quality_index():
    assert quality_index(0.0, 0.0) == 1.0
    assert np.isclose(quality_index(0.0, 0.3), 0.5)
    assert quality_index(0.0, 0.6) == 0.0 and quality_index(np.nan, 0.0) == 0.0
    assert quality_index(1.0, 0.0) > quality_index(2.0, 0.0)


def test_max_composite_takes_nearest_radar():
    a, b = _radar(60.0, 25.0, 1.0), _radar(60.0, 27.0, 2.0)  # ~111 km apart
    g = mosaic_grid([a, b])
    qa = to_grid(quality_index(a.HEIGHT.values, a.BLOCKAGE.values), a, g)
    qb = to_grid(quality_index(b.HEIGHT.values, b.BLOCKAGE.values), b, g)
    fa = to_grid(a.RAIN.values[0], a, g)
    fb = to_grid(b.RAIN.values[0], b, g)
    m, best, _ = composite([fa, fb], [np.nan_to_num(qa), np.nan_to_num(qb)], "max")
    j25 = np.argmin(np.abs(g.lon.values - 25.2))
    j27 = np.argmin(np.abs(g.lon.values - 26.8))
    i = np.argmin(np.abs(g.lat.values - 60.0))
    assert m[i, j25] == 1.0 and m[i, j27] == 2.0
    w, _, _ = composite([fa, fb], [np.nan_to_num(qa), np.nan_to_num(qb)], "weighted")
    jmid = np.argmin(np.abs(g.lon.values - 26.0))
    assert 1.3 < w[i, jmid] < 1.7  # halfway: equal weights


def test_overlap_difference():
    rng = np.random.default_rng(0)
    za = rng.uniform(20, 40, 500)
    d, iqr, n = overlap_difference(za, za - 6 + rng.normal(0, 1, 500), np.zeros(500), np.zeros(500), np.ones(500, bool))
    assert abs(d - 6) < 0.3 and n > 400
