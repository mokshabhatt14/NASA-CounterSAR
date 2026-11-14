import re                                  # regular expressions, used to pull the date out of the file name
from pathlib import Path                   # clean way to handle file paths on Windows

import h5py                                # reads the NISAR .h5 files
import numpy as np                         # array maths
import pandas as pd                        # date handling
import xarray as xr                        # builds the NetCDF cube

RAW_DIR = Path("data/raw/nisar")                   # where the downloaded scenes live
OUT_PATH = Path("data/processed/nisar_cube.nc")    # where the finished cube is written
BASE = "science/LSAR/GCOV/grids/frequencyA"        # path inside the .h5 to the 10 m layers

# The team's locked 100 m grid, copied from test_step2_dummy.py (pixel centres, metres, EPSG:32643)
X_OUT = np.arange(642700, 664700, 100, dtype=np.float64)    # 220 columns, west to east
Y_OUT = np.arange(1028400, 1050500, 100, dtype=np.float64)  # 221 rows, south to north

BLOCK = 10        # one 100 m cell = 10 x 10 pixels of 10 m
MIN_VALID = 50    # a 100 m cell needs at least 50 of its 100 pixels to be valid, else it becomes NaN


def window_indices(f):
    """Find which rows and columns of the big grid cover our 100 m grid."""
    x = f[BASE + "/xCoordinates"][:]            # 1-D list of x (pixel centres), west to east
    y = f[BASE + "/yCoordinates"][:]            # 1-D list of y (pixel centres), north to south
    cols = np.where((x > X_OUT[0] - 50) & (x < X_OUT[-1] + 50))[0]   # columns inside the box edges (centre -/+ 50 m)
    rows = np.where((y > Y_OUT[0] - 50) & (y < Y_OUT[-1] + 50))[0]   # rows inside the box edges
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1    # +1 because Python slices exclude the end
    expected = (len(Y_OUT) * BLOCK, len(X_OUT) * BLOCK)              # should be 2210 rows x 2200 columns
    assert (r1 - r0, c1 - c0) == expected, f"scene does not fully cover the grid: got {(r1 - r0, c1 - c0)}, need {expected}"
    return r0, r1, c0, c1


def block_mean_db(window):
    """Turn a 10 m linear-power window into a 100 m dB grid, rows ordered south to north."""
    window = window[::-1, :]                              # flip rows so y ascends, as the team's cube does
    valid = np.isfinite(window) & (window > 0)            # True where the pixel is a usable positive power value
    values = np.where(valid, window, 0.0)                 # unusable pixels become 0 so they add nothing to the sum
    shape = (len(Y_OUT), BLOCK, len(X_OUT), BLOCK)        # view the array as (cell row, 10 rows, cell col, 10 cols)
    total = values.reshape(shape).sum(axis=(1, 3), dtype=np.float64)   # sum of power inside each 10 x 10 block
    count = valid.reshape(shape).sum(axis=(1, 3))         # how many valid pixels each block has
    mean = np.where(count >= MIN_VALID, total / np.maximum(count, 1), np.nan)  # average power, NaN if too few valid pixels
    return (10.0 * np.log10(mean)).astype(np.float32)     # dB conversion AFTER averaging in linear power


def process_scene(path):
    """Read one NISAR file and return (date, HH_dB, HV_dB) on the 100 m grid."""
    date = re.search(r"_(\d{8})T\d{6}_", path.name).group(1)   # first 8-digit date in the file name, e.g. 20260821
    with h5py.File(path, "r") as f:                            # open the file read-only
        r0, r1, c0, c1 = window_indices(f)                     # rows and columns that cover Kuttanad
        hh = f[BASE + "/HHHH"][r0:r1, c0:c1]                   # read ONLY that window of the HH layer
        hv = f[BASE + "/HVHV"][r0:r1, c0:c1]                   # and of the HV layer
    return pd.Timestamp(date), block_mean_db(hh), block_mean_db(hv)


def build_cube():
    files = sorted(RAW_DIR.glob("NISAR_L2_PR_GCOV_*_113_A_006_*.h5"))   # ascending track 113, frame 006 only
    assert files, f"no matching .h5 files in {RAW_DIR}"
    results = sorted((process_scene(p) for p in files), key=lambda r: r[0])   # process each file, order by date

    ds = xr.Dataset(
        data_vars={
            "HH": (("time", "y", "x"), np.stack([r[1] for r in results]),
                   {"units": "dB", "crs": "EPSG:32643"}),     # same attrs as the team's "VV"
            "HV": (("time", "y", "x"), np.stack([r[2] for r in results]),
                   {"units": "dB", "crs": "EPSG:32643"}),
        },
        coords={
            "time": pd.DatetimeIndex([r[0] for r in results]),
            "y": ("y", Y_OUT, {"units": "m", "standard_name": "projection_y_coordinate"}),
            "x": ("x", X_OUT, {"units": "m", "standard_name": "projection_x_coordinate"}),
        },
        attrs={"crs": "EPSG:32643", "resolution_m": 100, "study_area": "Kuttanad, Kerala",
               "sensor": "NISAR-L", "product": "L2 GCOV frequencyA", "orbit": "ascending track 113 frame 006"},
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)         # create data/processed if missing
    ds.to_netcdf(OUT_PATH)                                     # write the cube
    return ds


if __name__ == "__main__":
    ds = build_cube()
    print(ds)
    for name in ("HH", "HV"):
        arr = ds[name]
        print(name, "NaN fraction:", float(arr.isnull().mean()),
              "| min/mean/max dB:", float(arr.min()), float(arr.mean()), float(arr.max()))