"""Writes the explainer's data: a tiny, real slice of ARCO-ERA5.

The explainer shows every row of its table and every partition, so it
needs a grid small enough to see: 2 m temperature at 12:00 UTC on eight
days, at six latitudes and eight longitudes. The selection is the code the
explainer shows. Run from the repository root:

    python docs/explainer/make_snapshot.py
"""

import json
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

era5 = xr.open_zarr(ARCO_ERA5, chunks=None, storage_options={"token": "anon"})
ds = (
    era5[["2m_temperature"]]
    .rename({"2m_temperature": "t2m"})
    .sel(
        time=pd.date_range("2024-07-01T12", periods=8, freq="D"),
        latitude=[75, 45, 15, -15, -45, -75],
        longitude=range(0, 360, 45),
    )
    .load()
)

snapshot = {
    "source": ARCO_ERA5,
    "variable": "t2m",
    "long_name": ds.t2m.attrs.get("long_name", "2 metre temperature"),
    "units": ds.t2m.attrs.get("units", "K"),
    "time": [str(t)[:10] for t in ds.time.values.astype("datetime64[D]")],
    "latitude": ds.latitude.values.tolist(),
    "longitude": ds.longitude.values.tolist(),
    # Rounded to 0.1 K, in (time, latitude, longitude) order.
    "values": np.round(ds.t2m.values.astype(float), 1).tolist(),
}
out = pathlib.Path(__file__).with_name("era5.json")
out.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n")
print(f"wrote {out} ({out.stat().st_size} bytes)")
