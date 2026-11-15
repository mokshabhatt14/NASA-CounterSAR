"""CounterSAR explorer for the Kuttanad SAR study area."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = (
    ROOT / "results" / "countersar_outputs.nc",
    ROOT / "results" / "dummy_output.nc",
    ROOT / "results" / "dummy_step2_outputs.nc",
)
CARDS_PATH = ROOT / "results" / "anomaly_cards.json"
PIXEL_AREA_KM2 = 0.01
BOUNDS = [[9.30, 76.30], [9.50, 76.50]]
LAYERS = ["Verdict Map", "Bidirectional Z-Score Map", "Observed SAR Backscatter (dB)", "Expected SAR Backscatter (dB)"]
VERDICT_COLORS = {
    "Expected": "#218765",
    "Unusual but Explained": "#d69a19",
    "Unexplained Candidate Event": "#d94b4b",
}


@st.cache_data(show_spinner="Loading CounterSAR NetCDF…")
def load_pipeline_data(path: str, modified_time: float, cards_path: str = "", cards_modified_time: float = 0.0) -> dict:
    """Load NetCDF variables and normalize backend naming into the UI contract."""
    with xr.open_dataset(path) as source:
        ds = source.load()
    aliases = {
        "observed": ("observed", "observed_vv"),
        "predicted": ("predicted", "expected_mean", "expected_mean_vv"),
        "z_scores": ("z_scores", "z_score_clim", "z_score"),
    }
    payload = {}
    payload["crs"] = str(ds.attrs.get("crs", "EPSG:32643"))
    for target, names in aliases.items():
        found = next((name for name in names if name in ds), None)
        if found is None:
            raise ValueError(f"NetCDF is missing {target}; expected one of {names}.")
        payload[target] = ds[found].rename(target)
    sigma_name = next((name for name in ("expected_sigma", "expected_sigma_vv", "sigma") if name in ds), None)
    payload["expected_sigma"] = ds[sigma_name].rename("expected_sigma") if sigma_name else None
    if "verdicts" in ds:
        payload["verdicts"] = ds["verdicts"].rename("verdicts").astype("int8")
    else:
        z = payload["z_scores"]
        # Candidate thresholds follow the requested three-tier display convention.
        payload["verdicts"] = xr.where(np.abs(z) > 2.5, 2, xr.where(np.abs(z) > 2.0, 1, 0)).astype("int8")
        payload["verdicts"].name = "verdicts"
    cards = ds.attrs.get("cards", {}) or {}
    if cards_path and Path(cards_path).exists():
        with open(cards_path, "r", encoding="utf-8") as cards_file:
            external_cards = json.load(cards_file)
        if isinstance(external_cards, list):
            cards = {item["date"]: item for item in external_cards if item.get("date")}
        elif isinstance(external_cards, dict):
            cards = external_cards
    payload["cards"] = cards
    # Optional per-date environmental context can be stored as dataset variables.
    payload["environment"] = {}
    dates = _date_strings(payload["observed"].time.values)
    for i, date in enumerate(dates):
        env = {}
        for key in ("rainfall_mm", "soil_moisture", "crop_phase", "sensor"):
            if key in ds:
                val = ds[key].isel(time=i).item()
                env[key] = val.item() if isinstance(val, np.generic) else val
        payload["environment"][date] = env
    # Normalize the pipeline's anomaly_cards.json schema for the dashboard.
    for date, card in payload["cards"].items():
        drivers = card.get("environmental_drivers", {}) or {}
        card.setdefault("env_summary", {
            "rainfall_mm": drivers.get("rainfall", drivers.get("rainfall_mm")),
            "soil_moisture": drivers.get("soil_moisture"),
            "crop_phase": drivers.get("crop_phase"),
            "sensor": drivers.get("sensor", "Sentinel-1 C-band"),
        })
        payload["environment"][date] = card["env_summary"] or payload["environment"].get(date, {})
    return payload


def _date_strings(values) -> list[str]:
    return [np.datetime_as_string(np.datetime64(value, "D"), unit="D") for value in values]


def _warp_to_geographic(array: xr.DataArray, crs: str, categorical: bool = False) -> tuple[np.ndarray, list[list[float]]]:
    """Resample an ascending projected (y, x) grid to a north-up geographic image."""
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates

    source = np.asarray(array.values, dtype=np.float32)
    x = np.asarray(array.x.values, dtype=np.float64)
    y = np.asarray(array.y.values, dtype=np.float64)
    if source.ndim != 2 or len(x) < 2 or len(y) < 2:
        raise ValueError("Map rasters must be 2D with at least two x and y coordinates.")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.allclose(np.diff(x), dx, rtol=1e-5, atol=1e-5) or not np.allclose(np.diff(y), dy, rtol=1e-5, atol=1e-5):
        raise ValueError("The map reprojection expects a regularly spaced x/y grid.")

    x_min, x_max = min(x[0], x[-1]) - abs(dx) / 2, max(x[0], x[-1]) + abs(dx) / 2
    y_min, y_max = min(y[0], y[-1]) - abs(dy) / 2, max(y[0], y[-1]) + abs(dy) / 2
    edge_x = np.concatenate((np.linspace(x_min, x_max, 512), np.full(256, x_max),
                             np.linspace(x_max, x_min, 512), np.full(256, x_min)))
    edge_y = np.concatenate((np.full(512, y_min), np.linspace(y_min, y_max, 256),
                             np.full(512, y_max), np.linspace(y_max, y_min, 256)))
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    edge_lon, edge_lat = to_geo.transform(edge_x, edge_y)
    west, east = float(np.min(edge_lon)), float(np.max(edge_lon))
    south, north = float(np.min(edge_lat)), float(np.max(edge_lat))
    bounds = [[south, west], [north, east]]

    rows, cols = source.shape
    lon_centers = west + (np.arange(cols) + 0.5) * (east - west) / cols
    lat_centers = north - (np.arange(rows) + 0.5) * (north - south) / rows
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    to_projected = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected_x, projected_y = to_projected.transform(lon_grid, lat_grid)
    source_cols = (projected_x - x[0]) / dx
    source_rows = (projected_y - y[0]) / dy
    warped = map_coordinates(
        source,
        [source_rows, source_cols],
        order=0 if categorical else 1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return warped.astype(np.float32), bounds


def _card_for(data: dict, date: str, index: int) -> tuple[dict, dict]:
    card = data["cards"].get(date, {})
    env = card.get("env_summary", {}) or data["environment"].get(date, {})
    verdict_slice = np.asarray(data["verdicts"].isel(time=index).values)
    count = int(np.count_nonzero(verdict_slice == 2))
    if not card.get("verdict"):
        card = dict(card)
        card["verdict"] = ("Unexplained Candidate Event" if count else
                           "Unusual but Explained" if np.any(verdict_slice == 1) else "Expected")
    if card.get("estimated_affected_area_km2") is None:
        card["estimated_affected_area_km2"] = count * PIXEL_AREA_KM2
    return card, env


def _as_png_data_uri(values: np.ndarray, layer: str) -> str:
    """Colorize a 2D raster, serialize PNG bytes, and return a Folium-safe URI."""
    from matplotlib import colormaps

    values = np.asarray(values)
    valid = np.isfinite(values)
    if layer == "Verdict Map":
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[values == 0] = (33, 135, 101, 70)
        rgba[values == 1] = (245, 190, 45, 220)
        rgba[values == 2] = (220, 61, 69, 240)
    elif layer == "Bidirectional Z-Score Map":
        scale = np.clip((np.nan_to_num(values, nan=0.0) + 4.0) / 8.0, 0, 1)
        rgba = (colormaps["RdBu_r"](scale) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid & (np.abs(values) >= 0.25), 225, 0).astype(np.uint8)
    else:
        low, high = -22.0, -4.0
        scale = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
        gray = (scale * 255).astype(np.uint8)
        rgba = np.stack((gray, gray, gray, np.where(valid, 220, 0).astype(np.uint8)), axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_map(data: dict, index: int, layer: str) -> None:
    try:
        import folium
        from streamlit_folium import st_folium

        variable = {
            "Verdict Map": "verdicts",
            "Bidirectional Z-Score Map": "z_scores",
            "Observed SAR Backscatter (dB)": "observed",
            "Expected SAR Backscatter (dB)": "predicted",
        }[layer]
        raster, raster_bounds = _warp_to_geographic(
            data[variable].isel(time=index),
            data["crs"],
            categorical=(layer == "Verdict Map"),
        )
        image_uri = _as_png_data_uri(raster, layer)
        # OpenStreetMap provides geographic context without a key-protected tile service.
        # Keep the view fixed to the full study area so the raster and boundaries stay in frame.
        fmap = folium.Map(
            location=[9.40, 76.40],
            zoom_start=10,
            tiles="OpenStreetMap",
            control_scale=True,
        )
        folium.raster_layers.ImageOverlay(
            image=image_uri,
            bounds=raster_bounds,
            opacity=0.9,
            interactive=True,
            name=layer,
            zindex=2,
        ).add_to(fmap)
        # Draw a contrasting double outline above the raster so it remains easy to see.
        folium.Rectangle(
            bounds=BOUNDS,
            color="#102f35",
            weight=8,
            opacity=0.95,
            fill=False,
            tooltip="Kuttanad study area boundary",
        ).add_to(fmap)
        folium.Rectangle(
            bounds=BOUNDS,
            color="#38f2d0",
            weight=4,
            opacity=1,
            fill=False,
            tooltip="Kuttanad study area: 9.30–9.50°N, 76.30–76.50°E",
        ).add_to(fmap)
        folium.Marker(
            location=[9.40, 76.40],
            icon=folium.DivIcon(
                html=(
                    '<div style="white-space:nowrap;background:rgba(12,38,43,.82);'
                    'color:white;border:1px solid #38f2d0;border-radius:5px;'
                    'padding:4px 8px;font-size:13px;font-weight:700;">Kuttanad study area</div>'
                )
            ),
        ).add_to(fmap)
        fmap.fit_bounds(raster_bounds, padding=(24, 24))
        folium.LayerControl().add_to(fmap)
        st_folium(fmap, height=677, use_container_width=True, returned_objects=[])
    except Exception as exc:
        st.error(f"Map could not be rendered: {exc}")


def _rgb(values: np.ndarray) -> Image.Image:
    from matplotlib import colormaps

    scale = np.clip((np.nan_to_num(values, nan=-22) + 22) / 18, 0, 1)
    rgba = (colormaps["gray"](scale) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(values), 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def main() -> None:
    st.set_page_config(page_title="CounterSAR | Kuttanad", page_icon="🌾", layout="wide")
    background_path = Path(__file__).resolve().parent / "assets" / "earth-horizon.jpg"
    try:
        background_b64 = base64.b64encode(background_path.read_bytes()).decode("ascii")
        background_css = f'url("data:image/jpeg;base64,{background_b64}")'
    except OSError:
        background_css = "linear-gradient(135deg,#07111d,#15354d)"
    styles = """<style>
    :root{--ink:#f3f7fb;--muted:#bac8d5;--mint:#63e0c1;--gold:#f0bd70;--glass:rgba(8,18,31,.76);--line:rgba(210,231,245,.17)}
    .stApp{
      color:var(--ink);font-family:"Aptos","Segoe UI","Helvetica Neue",sans-serif;background-color:#07111d;
      background-image:linear-gradient(90deg,rgba(4,10,19,.91) 0%,rgba(4,12,23,.67) 45%,rgba(4,12,23,.35) 100%),
        linear-gradient(0deg,rgba(4,10,18,.78),rgba(4,10,18,.12) 60%,rgba(4,10,18,.42)),
        __EARTH_BACKGROUND__;
      background-size:cover;background-position:center 42%;background-attachment:fixed;
    }
    [data-testid="stHeader"]{background:rgba(3,10,18,.18)}
    [data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent}
    .block-container{padding:1.4rem 2.5vw 3.5rem;max-width:2200px;width:100%}
    h1,h2,h3,p,label,.stMarkdown{color:var(--ink)}
    [data-testid="stMarkdownContainer"] p{font-size:1.08rem;line-height:1.72}
    [data-testid="stCaptionContainer"]{color:var(--muted);font-size:.95rem}
    .masthead{position:relative;min-height:clamp(230px,31vh,390px);display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-end;
      padding:2.5rem 0 2rem;margin:0 0 1.15rem;border:0;border-radius:0;background:none;box-shadow:none;backdrop-filter:none}
    .eyebrow{font-size:.9rem;letter-spacing:.29em;text-transform:uppercase;color:var(--mint);font-weight:700}
    .masthead h1{margin:.72rem 0 .3rem;color:#fff;font-size:clamp(4rem,9vw,8rem);line-height:.94;letter-spacing:.075em;text-transform:uppercase;font-weight:350;text-shadow:0 4px 35px rgba(0,0,0,.5)}
    .masthead p{margin:.75rem 0 0;color:#e2edf5;font-size:clamp(1.2rem,1.8vw,1.7rem);line-height:1.55;letter-spacing:.015em;max-width:1040px;text-shadow:0 2px 18px rgba(0,0,0,.6)}
    .source-chip{position:absolute;right:0;bottom:2.2rem;border:1px solid rgba(193,231,245,.34);border-radius:99px;padding:.65rem 1rem;color:#f2fbff;
      background:rgba(182,223,239,.12);box-shadow:inset 0 1px 0 rgba(255,255,255,.27),0 8px 32px rgba(0,0,0,.22);backdrop-filter:blur(18px) saturate(155%);
      font-size:.95rem;letter-spacing:.08em}
    h2{font-size:2.25rem!important;line-height:1.18!important;font-weight:450!important;letter-spacing:.015em!important}
    h3{font-size:1.55rem!important;line-height:1.3!important;font-weight:550!important}
    [data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.65rem;border:1px solid rgba(226,244,252,.22);background:linear-gradient(135deg,rgba(220,242,249,.16),rgba(118,165,185,.07));
      padding:.55rem .7rem;border-radius:18px;backdrop-filter:blur(24px) saturate(165%);box-shadow:inset 0 1px 0 rgba(255,255,255,.24),0 14px 38px rgba(0,0,0,.2)}
    [data-testid="stTabs"] button[role="tab"]{height:66px;padding:0 1.6rem;border-radius:14px;color:#d4e1ea;font-size:1.265rem;letter-spacing:.055em;transition:all .2s ease}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:rgba(99,224,193,.14);color:#eafff9}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:#eafff9}
    [data-testid="stWidgetLabel"] p{font-size:1.05rem;line-height:1.4;color:#eaf1f7}
    [data-testid="stMetric"]{background:linear-gradient(135deg,rgba(226,244,250,.18),rgba(154,197,215,.07));border:1px solid rgba(225,244,252,.25);
      padding:1.5rem 1.55rem;border-radius:20px;backdrop-filter:blur(24px) saturate(155%);box-shadow:inset 0 1px 0 rgba(255,255,255,.23),0 14px 36px rgba(0,0,0,.2)}
    [data-testid="stMetricLabel"] p{color:#d0dee8;font-size:1.12rem;line-height:1.45}
    [data-testid="stMetricValue"]{color:#fff;font-size:2.35rem;line-height:1.2}
    [data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(135deg,rgba(221,242,249,.14),rgba(108,152,176,.055));border-color:rgba(225,244,252,.2);border-radius:20px;backdrop-filter:blur(22px) saturate(150%);box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 16px 40px rgba(0,0,0,.16)}
    [data-testid="stPlotlyChart"]{padding:1rem;border:1px solid rgba(225,244,252,.19);border-radius:22px;background:linear-gradient(135deg,rgba(221,242,249,.13),rgba(108,152,176,.045));backdrop-filter:blur(22px) saturate(145%);box-shadow:inset 0 1px 0 rgba(255,255,255,.17),0 16px 40px rgba(0,0,0,.18)}
    .tag{display:inline-block;padding:.35rem .8rem;border-radius:30px;font-weight:750;color:white;box-shadow:0 5px 18px rgba(0,0,0,.22)}
    .notice{padding:1rem 1.2rem;border-left:4px solid var(--gold);background:rgba(46,34,20,.75);border-radius:8px;color:#fff0d7}
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p{color:#eaf1f7}
    [data-testid="stSelectSlider"] p{color:#eaf1f7}
    [data-testid="stImage"]{padding:.5rem;border:1px solid rgba(225,244,252,.19);border-radius:21px;background:linear-gradient(135deg,rgba(221,242,249,.13),rgba(108,152,176,.045));backdrop-filter:blur(18px);box-shadow:0 16px 40px rgba(0,0,0,.2)}
    [data-testid="stImage"] img{border-radius:14px;border:1px solid rgba(215,237,248,.2);box-shadow:0 14px 38px rgba(0,0,0,.28)}
    .st-key-home_page,.st-key-assessment_page,.st-key-spatial_page{--page-scale:1.15}
    .st-key-time_machine_page{--page-scale:1.25}
    .st-key-home_page [data-testid="stMarkdownContainer"] p,.st-key-home_page [data-testid="stMarkdownContainer"] li,
    .st-key-assessment_page [data-testid="stMarkdownContainer"] p,.st-key-assessment_page [data-testid="stMarkdownContainer"] li,
    .st-key-spatial_page [data-testid="stMarkdownContainer"] p,.st-key-spatial_page [data-testid="stMarkdownContainer"] li,
    .st-key-time_machine_page [data-testid="stMarkdownContainer"] p,.st-key-time_machine_page [data-testid="stMarkdownContainer"] li{font-size:calc(1.08rem * var(--page-scale));line-height:1.72}
    .st-key-home_page h1,.st-key-home_page h2,.st-key-home_page h3,
    .st-key-assessment_page h1,.st-key-assessment_page h2,.st-key-assessment_page h3,
    .st-key-spatial_page h1,.st-key-spatial_page h2,.st-key-spatial_page h3,
    .st-key-time_machine_page h1,.st-key-time_machine_page h2,.st-key-time_machine_page h3{font-size:calc(1em * var(--page-scale))!important}
    .st-key-home_page h2,.st-key-assessment_page h2,.st-key-spatial_page h2,.st-key-time_machine_page h2{font-size:calc(2.25rem * var(--page-scale))!important}
    .st-key-home_page h3,.st-key-assessment_page h3,.st-key-spatial_page h3,.st-key-time_machine_page h3{font-size:calc(1.55rem * var(--page-scale))!important}
    .st-key-home_page [data-testid="stCaptionContainer"],.st-key-assessment_page [data-testid="stCaptionContainer"],.st-key-spatial_page [data-testid="stCaptionContainer"],.st-key-time_machine_page [data-testid="stCaptionContainer"]{font-size:calc(.95rem * var(--page-scale))}
    .st-key-home_page [data-testid="stWidgetLabel"] p,.st-key-assessment_page [data-testid="stWidgetLabel"] p,.st-key-spatial_page [data-testid="stWidgetLabel"] p,.st-key-time_machine_page [data-testid="stWidgetLabel"] p{font-size:calc(1.05rem * var(--page-scale))}
    .st-key-assessment_page [data-testid="stMetricLabel"] p,.st-key-home_page [data-testid="stMetricLabel"] p,.st-key-spatial_page [data-testid="stMetricLabel"] p,.st-key-time_machine_page [data-testid="stMetricLabel"] p{font-size:calc(1.12rem * var(--page-scale))}
    .st-key-assessment_page [data-testid="stMetricValue"],.st-key-home_page [data-testid="stMetricValue"],.st-key-spatial_page [data-testid="stMetricValue"],.st-key-time_machine_page [data-testid="stMetricValue"]{font-size:calc(2.35rem * var(--page-scale))}
    .st-key-assessment_page .tag,.st-key-home_page .tag,.st-key-spatial_page .tag,.st-key-time_machine_page .tag{font-size:calc(1rem * var(--page-scale))}
    .st-key-home_page [data-testid="stVerticalBlockBorderWrapper"],.st-key-assessment_page [data-testid="stVerticalBlockBorderWrapper"]{padding:1.1rem}
    .verdict-card{height:190px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;padding:1.4rem 1.3rem;
      border:1px solid rgba(225,244,252,.25);border-radius:16px;background:linear-gradient(135deg,rgba(226,244,250,.16),rgba(154,197,215,.055));
      backdrop-filter:blur(22px) saturate(150%);box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 14px 34px rgba(0,0,0,.18)}
    .verdict-card h3{font-size:1.78rem!important;line-height:1.25!important;margin:0 0 .7rem!important}
    .verdict-card p{font-size:1.242rem!important;line-height:1.5!important;margin:.15rem 0!important}
    a{color:#83e6d0}
    @media(max-width:800px){.block-container{padding:1rem .8rem 2rem}.masthead{min-height:230px;padding:1.4rem 0}.masthead h1{font-size:clamp(2.8rem,13vw,5rem);letter-spacing:.035em}.eyebrow{font-size:.75rem;letter-spacing:.18em}.source-chip{position:static;display:inline-block;margin-top:1rem}h2{font-size:1.8rem!important}[data-testid="stMetricValue"]{font-size:1.8rem}[data-testid="stTabs"] button[role="tab"]{font-size:1rem;padding:0 .7rem}.verdict-card{height:190px}.verdict-card h3{font-size:1.35rem!important}.verdict-card p{font-size:1.05rem!important}}
    </style>"""
    st.markdown(styles.replace("__EARTH_BACKGROUND__", background_css), unsafe_allow_html=True)

    path = next((candidate for candidate in DATA_PATHS if candidate.exists()), None)
    if path is None:
        st.error(f"NetCDF not found. Looked for: {', '.join(str(p) for p in DATA_PATHS)}")
        st.stop()
    try:
        # File modification times in the cache key make freshly pulled outputs visible immediately.
        cards_path = str(CARDS_PATH) if CARDS_PATH.exists() else ""
        cards_mtime = CARDS_PATH.stat().st_mtime if cards_path else 0.0
        data = load_pipeline_data(str(path), path.stat().st_mtime, cards_path, cards_mtime)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        st.stop()

    dates = _date_strings(data["observed"].time.values)
    if not dates:
        st.error("The NetCDF contains no time steps.")
        st.stop()
    st.markdown(
        f"<div class='masthead'><div class='eyebrow'>NASA Space Apps Challenge &nbsp; / &nbsp; Kuttanad, Kerala</div>"
        f"<span class='source-chip'>DATA &nbsp;·&nbsp; {path.name}</span>"
        "<h1>CounterSAR</h1><p>Is it unusual, or just the season? Read flood signals against the rhythm of rain, crops, and radar.</p></div>",
        unsafe_allow_html=True,
    )
    home_tab, time_tab, assessment_tab, spatial_tab = st.tabs(["Home", "Time Machine", "Daily Anomaly Assessment", "Spatial Visualization"])

    with time_tab, st.container(key="time_machine_page"):
        st.subheader("Time Machine")
        st.caption("Track the seasonal signal across every acquisition. Select a date to synchronize the assessment and spatial view.")
        selected_date = st.select_slider("Observation date", options=dates, value=dates[0], key="selected_date")
        index = dates.index(selected_date)
        observed_mean = np.array([float(data["observed"].isel(time=i).mean(skipna=True)) for i in range(len(dates))])
        predicted_mean = np.array([float(data["predicted"].isel(time=i).mean(skipna=True)) for i in range(len(dates))])
        if data["expected_sigma"] is not None:
            sigma = np.array([float(data["expected_sigma"].isel(time=i).mean(skipna=True)) for i in range(len(dates))])
        else:
            # Keep a usable band for older files that do not contain baseline sigma.
            sigma = np.array([float((data["observed"].isel(time=i) - data["predicted"].isel(time=i)).std(skipna=True)) for i in range(len(dates))])
        rainfall = []
        for i, date in enumerate(dates):
            card, env = _card_for(data, date, i)
            rainfall.append(env.get("rainfall_mm"))
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=predicted_mean + sigma, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=predicted_mean - sigma, line=dict(width=0), fill="tonexty", fillcolor="rgba(99,224,193,.20)", name="Expected ± 1σ"))
        fig.add_trace(go.Scatter(x=dates, y=predicted_mean, name="Predicted dB", line=dict(color="#55DDF5", dash="dash", width=3)))
        fig.add_trace(go.Scatter(x=dates, y=observed_mean, name="Observed dB", mode="lines+markers", line=dict(color="#FFD166", width=3), marker=dict(size=9)))
        if any(value is not None for value in rainfall):
            fig.add_trace(go.Bar(x=dates, y=[value or 0 for value in rainfall], name="Rainfall (mm)", yaxis="y2", marker_color="rgba(63,133,184,.32)"))
        fig.add_vline(x=selected_date, line_dash="dot", line_color="#FFFFFF", line_width=2)
        fig.update_layout(height=594, margin=dict(l=20, r=24, t=52, b=20), hovermode="x unified",
                          legend=dict(orientation="h", y=1.12, font=dict(size=19, color="#FFFFFF"),
                                      bgcolor="rgba(4,12,22,.58)", bordercolor="rgba(225,244,252,.2)", borderwidth=1),
                          yaxis=dict(title=dict(text="Backscatter (dB)", font=dict(size=21, color="#F4F8FC")),
                                     tickfont=dict(size=21, color="#F4F8FC"), gridcolor="rgba(226,240,250,.15)"),
                          yaxis2=dict(title=dict(text="Rainfall (mm)", font=dict(size=21, color="#E0D5FF")),
                                      tickfont=dict(size=21, color="#E0D5FF"), overlaying="y", side="right", showgrid=False),
                          xaxis=dict(tickfont=dict(size=20, color="#F4F8FC"), gridcolor="rgba(226,240,250,.08)"),
                          barmode="overlay", font=dict(family="Aptos, Segoe UI, sans-serif", size=21, color="#eef4f8"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,14,25,.62)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if not any(value is not None for value in rainfall):
            st.caption("Rainfall data is not included in this NetCDF.")

    selected_date = st.session_state.get("selected_date", dates[0])
    index = dates.index(selected_date) if selected_date in dates else 0
    card, env = _card_for(data, selected_date, index)
    with home_tab, st.container(key="home_page"):
        st.subheader("The signal, in context")
        with st.container(border=True):
            st.markdown("### CounterSAR: Is it unusual, or just the season?")
            st.write("CounterSAR compares observed radar backscatter with an expected baseline informed by seasonal and environmental conditions. Wet fields can change a radar signal as part of the monsoon and crop cycle; the system highlights departures that are not explained by that expected context.")
        st.markdown("#### Three-tier verdict system")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='verdict-card'><h3>🟢 Expected</h3><p>Radar behavior fits the expected seasonal baseline.</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='verdict-card'><h3>🟡 Unusual but Explained</h3><p>A change is visible, with a plausible weather or seasonal explanation.</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='verdict-card'><h3>🔴 Unexplained Candidate Event</h3><p>A statistical departure worth investigation; not a confirmed disaster.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='notice'><b>How to read findings:</b> Observed measurements, contextual weather evidence, and physical-driver hypotheses are distinct. A radar anomaly is a statistical deviation, not a confirmed disaster.</div>", unsafe_allow_html=True)
        st.caption(f"Loaded {len(dates)} acquisitions from {path.name} · Selected date: {selected_date}")

    with assessment_tab, st.container(key="assessment_page"):
        st.subheader(f"Daily Anomaly Assessment · {selected_date}")
        raw_verdict = str(card.get("verdict", "Expected"))
        verdict = next((name for name in VERDICT_COLORS if name.lower() in raw_verdict.lower()), "Expected")
        event_pixels = int(np.count_nonzero(data["verdicts"].isel(time=index).values == 2))
        area = float(card.get("estimated_affected_area_km2", event_pixels * PIXEL_AREA_KM2) or 0)
        if event_pixels and not area:
            area = event_pixels * PIXEL_AREA_KM2
        st.markdown(f"<span class='tag' style='background:{VERDICT_COLORS[verdict]}'>SYSTEM VERDICT · {verdict}</span>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Candidate affected area", f"{area:.2f} km²")
        k2.metric("Confidence / persistence", str(card.get("confidence", "Not provided")))
        k3.metric("Active sensor", str(env.get("sensor", "Not provided")))
        k4.metric("Candidate pixels", f"{event_pixels:,}")
        e1, e2, e3 = st.columns(3)
        rainfall_value = env.get("rainfall_mm")
        e1.metric("Rainfall", f"{rainfall_value} mm" if rainfall_value is not None else "Unavailable")
        soil_moisture = env.get("soil_moisture")
        crop_phase = env.get("crop_phase")
        e2.metric("Soil moisture", str(soil_moisture) if soil_moisture is not None else "Unavailable")
        e3.metric("Crop phase", str(crop_phase) if crop_phase is not None else "Unavailable")
        if rainfall_value is None or env.get("soil_moisture") is None:
            st.warning("Weather context is missing from this file. Mode B fallback: interpret the radar signal without full weather corroboration.")
        z = data["z_scores"].isel(time=index).values
        v = data["verdicts"].isel(time=index).values
        negative = int(np.count_nonzero((z < -2.0) & (v == 2)))
        positive = int(np.count_nonzero((z > 2.0) & (v == 2)))
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown(f"### 🔵 Negative anomaly · {negative:,} pixels")
                st.write("Z < −2.0")
                st.write("**Hypothesis:** specular reflection from open water or inundated bare paddy.")
        with right:
            with st.container(border=True):
                st.markdown(f"### 🟠 Positive anomaly · {positive:,} pixels")
                st.write("Z > +2.0")
                st.write("**Hypothesis:** double-bounce scattering from flooded vegetation or canopy-water interaction.")
        st.markdown("<div class='notice'><b>Scientific disclaimer:</b> Every finding is labelled Observed, Contextual, or Hypothesis. A radar anomaly is a statistical deviation, not a confirmed disaster.</div>", unsafe_allow_html=True)

    with spatial_tab, st.container(key="spatial_page"):
        st.subheader(f"Spatial Visualization · {selected_date}")
        map_frame = st.columns([1, 6, 1])
        with map_frame[1]:
            layer = st.selectbox("Map layer", LAYERS, key="map_layer")
            _render_map(data, index, layer)
        if layer == "Bidirectional Z-Score Map":
            st.caption("Blue = negative Z · white = near expected · red = positive Z. Bounds: 9.30–9.50°N, 76.30–76.50°E.")
        elif "Backscatter" in layer:
            st.caption("Backscatter intensity uses a shared −22 to −4 dB scale.")

        st.markdown("#### Verdict Map legend")
        st.write("Each pixel is classified for the selected date. The map summarizes whether its radar signal fits the seasonal expectation, has a contextual explanation, or is a candidate anomaly.")
        legend_cols = st.columns(3)
        legend_cols[0].markdown("🟢 **Expected**  \nRadar behavior fits the expected seasonal baseline.")
        legend_cols[1].markdown("🟡 **Unusual but Explained**  \nA change is visible and may be explained by weather or seasonal conditions.")
        legend_cols[2].markdown("🔴 **Unexplained Candidate Event**  \nA statistical departure that may warrant investigation; it is not confirmed damage.")

        st.markdown("#### Expected vs. actual backscatter")
        st.caption("Both images use the same scale (−22 to −4 dB) for direct visual comparison.")
        expected_values, image_bounds = _warp_to_geographic(data["predicted"].isel(time=index), data["crs"])
        actual_values, actual_bounds = _warp_to_geographic(data["observed"].isel(time=index), data["crs"])
        if not np.allclose(image_bounds, actual_bounds, rtol=0, atol=1e-10):
            st.error("Expected and observed layers do not share the same geographic footprint.")
        expected = _rgb(expected_values)
        actual = _rgb(actual_values)
        st.caption(
            f"North-up footprint · {image_bounds[0][0]:.4f}–{image_bounds[1][0]:.4f}°N · "
            f"{image_bounds[0][1]:.4f}–{image_bounds[1][1]:.4f}°E"
        )
        before, after = st.columns(2, gap="large")
        before.image(expected, caption="Expected backscatter (dB)", use_container_width=True)
        after.image(actual, caption="Observed backscatter (dB)", use_container_width=True)

    st.markdown(
        "<div style='text-align:right;color:rgba(226,238,247,.72);font-size:.75rem;padding-top:1rem;'>"
        "Earth horizon photograph: NASA / ISS crew · "
        "<a href='https://commons.wikimedia.org/wiki/File:Earth%27s_thin_blue_atmosphere_traces_the_planet%27s_horizon_above_the_Pacific_Ocean_(iss074e0089803).jpg' "
        "target='_blank'>image source</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
