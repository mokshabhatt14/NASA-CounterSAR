import xarray as xr                                   # opens the NetCDF cube
import matplotlib.pyplot as plt                       # draws the figure
from pathlib import Path                              # path handling for the output folder

ds = xr.open_dataset("data/processed/nisar_cube.nc")  # load the cube (run from the repo root)
day = ds.isel(time=0)                                 # first date, 21 Aug 2026 (isel picks by position)
limits = {"HH": (-25, 5), "HV": (-30, -5)}            # dB colour range per channel: water dark, land with detail

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)  # constrained_layout spaces panels, labels and colour bars so nothing overlaps
for ax, pol in zip(axes, ["HH", "HV"]):               # pair each panel with a channel
    vmin, vmax = limits[pol]                          # colour range for this channel
    day[pol].plot(ax=ax, cmap="gray", vmin=vmin, vmax=vmax,   # draw in grey with the fixed range
                  cbar_kwargs={"label": f"{pol} backscatter (dB)"})  # label the colour bar
    ax.set_title(f"NISAR L-band {pol}")               # panel title
    ax.set_xlabel("Easting (m, EPSG:32643)")          # x axis label
    ax.set_ylabel("Northing (m)" if pol == "HH" else "")  # only the left panel needs the y label, which removes the collision
    ax.ticklabel_format(style="plain", useOffset=False)   # show full numbers instead of the "1e6" shorthand
    ax.set_aspect("equal")                            # 1 m east = 1 m north, so the map isn't stretched
fig.suptitle(f"Kuttanad, NISAR ascending track 113, {str(day.time.values)[:10]} (100 m)")  # title with the date

Path("src/nisar/figures").mkdir(parents=True, exist_ok=True)   # create the output folder if missing
fig.savefig("src/nisar/figures/nisar_hh_hv.png", dpi=200, bbox_inches="tight")  # dpi = sharpness; tight trims white borders