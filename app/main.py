"""CounterSAR explorer for the Kuttanad SAR study area."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "dummy_output.nc"
# The repository currently carries this step-output filename; dummy_output.nc is preferred.
FALLBACK_DATA_PATH = ROOT / "results" / "dummy_step2_outputs.nc"
PIXEL_AREA_KM2 = 0.01
BOUNDS = [[9.30, 76.30], [9.50, 76.50]]
LAYERS = ["Verdict Map", "Bidirectional Z-Score Map", "Observed SAR Backscatter (dB)", "Expected SAR Backscatter (dB)"]
VERDICT_COLORS = {
    "Expected": "#218765",
    "Unusual but Explained": "#d69a19",
    "Unexplained Candidate Event": "#d94b4b",
}


@st.cache_data(show_spinner="Loading CounterSAR NetCDF…")
def load_pipeline_data(path: str, modified_time: float) -> dict:
    """Load NetCDF variables and normalize backend naming into the UI contract."""
    with xr.open_dataset(path) as source:
        ds = source.load()
    aliases = {
        "observed": ("observed", "observed_vv"),
        "predicted": ("predicted", "expected_mean", "expected_mean_vv"),
        "z_scores": ("z_scores", "z_score_clim", "z_score"),
    }
    payload = {}
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
    payload["cards"] = ds.attrs.get("cards", {}) or {}
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
    return payload


def _date_strings(values) -> list[str]:
    return [np.datetime_as_string(np.datetime64(value, "D"), unit="D") for value in values]


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
    if layer == "Verdict Map":
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[values == 0] = (33, 135, 101, 70)
        rgba[values == 1] = (245, 190, 45, 220)
        rgba[values == 2] = (220, 61, 69, 240)
    elif layer == "Bidirectional Z-Score Map":
        scale = np.clip((np.nan_to_num(values, nan=0.0) + 4.0) / 8.0, 0, 1)
        rgba = (colormaps["RdBu_r"](scale) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(np.abs(values) < 0.25, 0, 225).astype(np.uint8)
    else:
        low, high = -22.0, -4.0
        scale = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
        gray = (scale * 255).astype(np.uint8)
        rgba = np.stack((gray, gray, gray, np.full_like(gray, 220)), axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_map(data: dict, index: int, layer: str, opacity: float) -> None:
    try:
        import folium
        from streamlit_folium import st_folium

        variable = {
            "Verdict Map": "verdicts",
            "Bidirectional Z-Score Map": "z_scores",
            "Observed SAR Backscatter (dB)": "observed",
            "Expected SAR Backscatter (dB)": "predicted",
        }[layer]
        raster = data[variable].isel(time=index).values
        image_uri = _as_png_data_uri(raster, layer)
        # A tile-free canvas avoids third-party basemap keys and keeps the raster visible offline.
        fmap = folium.Map(location=[9.40, 76.40], zoom_start=12, tiles=None, control_scale=True)
        folium.Rectangle(bounds=BOUNDS, color="#277d85", weight=2, fill=False, tooltip="Kuttanad study area").add_to(fmap)
        folium.raster_layers.ImageOverlay(
            image=image_uri,
            bounds=BOUNDS,
            opacity=opacity,
            interactive=True,
            name=layer,
            zindex=2,
        ).add_to(fmap)
        folium.LayerControl().add_to(fmap)
        st_folium(fmap, height=700, use_container_width=True, returned_objects=[])
    except Exception as exc:
        st.error(f"Map could not be rendered: {exc}")


def _rgb(values: np.ndarray) -> Image.Image:
    from matplotlib import colormaps

    scale = np.clip((np.nan_to_num(values, nan=-22) + 22) / 18, 0, 1)
    return Image.fromarray((colormaps["gray"](scale)[..., :3] * 255).astype(np.uint8))


def main() -> None:
    st.set_page_config(page_title="CounterSAR | Kuttanad", page_icon="🌾", layout="wide")
    st.markdown("""<style>
    .stApp{background:linear-gradient(145deg,#f5f8f6,#eef4f5 58%,#f8f4e9)}
    .block-container{padding-top:1.2rem;max-width:1800px}
    .hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#123c43,#17676a);color:white;margin-bottom:1rem}
    .hero h1{margin:0;color:white;font-size:2.35rem}.hero p{margin:.35rem 0 0;color:#d5eeea}
    .tag{display:inline-block;padding:.3rem .7rem;border-radius:30px;font-weight:700;color:white}
    .notice{padding:1rem 1.2rem;border-left:5px solid #d69a19;background:#fff8e4;border-radius:8px;color:#55451c}
    </style>""", unsafe_allow_html=True)

    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    if not path.exists():
        st.error(f"NetCDF not found. Expected: {DATA_PATH}")
        st.stop()
    try:
        # Include the file mtime in the cache key so a refreshed pull is loaded immediately.
        data = load_pipeline_data(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        st.stop()

    dates = _date_strings(data["observed"].time.values)
    if not dates:
        st.error("The NetCDF contains no time steps.")
        st.stop()
    with st.sidebar:
        st.title("🌾 CounterSAR")
        st.caption(f"Data source · {path.name}")
        st.caption("Kuttanad, Kerala · 76.30–76.50°E · 9.30–9.50°N")

    st.markdown("<div class='hero'><h1>CounterSAR</h1><p>Is it unusual, or just the season? · Flood signals with environmental context</p></div>", unsafe_allow_html=True)
    home_tab, time_tab, assessment_tab, spatial_tab = st.tabs(["Home", "Time Machine", "Daily Anomaly Assessment", "Spatial Visualization"])

    with time_tab:
        st.subheader("Time Machine")
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
        fig.add_trace(go.Scatter(x=dates, y=predicted_mean - sigma, line=dict(width=0), fill="tonexty", fillcolor="rgba(27,132,133,.16)", name="Expected ± 1σ"))
        fig.add_trace(go.Scatter(x=dates, y=predicted_mean, name="Predicted dB", line=dict(color="#16817c", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=observed_mean, name="Observed dB", mode="lines+markers", line=dict(color="#d9693b", width=2)))
        if any(value is not None for value in rainfall):
            fig.add_trace(go.Bar(x=dates, y=[value or 0 for value in rainfall], name="Rainfall (mm)", yaxis="y2", marker_color="rgba(63,133,184,.32)"))
        fig.add_vline(x=selected_date, line_dash="dot", line_color="#333")
        fig.update_layout(height=420, margin=dict(l=10, r=15, t=35, b=10), hovermode="x unified",
                          legend=dict(orientation="h", y=1.12), yaxis=dict(title="Backscatter (dB)"),
                          yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", showgrid=False),
                          barmode="overlay", plot_bgcolor="rgba(255,255,255,.6)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if not any(value is not None for value in rainfall):
            st.caption("Rainfall data is not included in this NetCDF.")

    selected_date = st.session_state.get("selected_date", dates[0])
    index = dates.index(selected_date) if selected_date in dates else 0
    card, env = _card_for(data, selected_date, index)
    with home_tab:
        st.subheader("Welcome to CounterSAR")
        st.markdown("### CounterSAR: Is it unusual, or just the season?")
        st.write("CounterSAR compares observed radar backscatter with an expected baseline informed by seasonal and environmental conditions. Wet fields can change a radar signal as part of the monsoon and crop cycle; the system highlights departures that are not explained by that expected context.")
        st.markdown("#### Three-tier verdict system")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**🟢 Expected**  \nRadar behavior fits the expected seasonal baseline.")
        c2.markdown("**🟡 Unusual but Explained**  \nA change is visible, with a plausible weather or seasonal explanation.")
        c3.markdown("**🔴 Unexplained Candidate Event**  \nA statistical departure worth investigation; not a confirmed disaster.")
        st.markdown("<div class='notice'><b>How to read findings:</b> Observed measurements, contextual weather evidence, and physical-driver hypotheses are distinct. A radar anomaly is a statistical deviation, not a confirmed disaster.</div>", unsafe_allow_html=True)
        st.caption(f"Loaded {len(dates)} acquisitions from {path.name} · Selected date: {selected_date}")

    with assessment_tab:
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
        e2.metric("Soil moisture", str(env.get("soil_moisture", "Unavailable")))
        e3.metric("Crop phase", str(env.get("crop_phase", "Unavailable")))
        if rainfall_value is None or env.get("soil_moisture") is None:
            st.warning("Weather context is missing from this file. Mode B fallback: interpret the radar signal without full weather corroboration.")
        z = data["z_scores"].isel(time=index).values
        v = data["verdicts"].isel(time=index).values
        negative = int(np.count_nonzero((z < -2.0) & (v == 2)))
        positive = int(np.count_nonzero((z > 2.0) & (v == 2)))
        left, right = st.columns(2)
        left.markdown(f"### 🔵 Negative anomaly · {negative:,} pixels\nZ < −2.0  \n**Hypothesis:** specular reflection from open water or inundated bare paddy.")
        right.markdown(f"### 🟠 Positive anomaly · {positive:,} pixels\nZ > +2.0  \n**Hypothesis:** double-bounce scattering from flooded vegetation or canopy-water interaction.")
        st.markdown("<div class='notice'><b>Scientific disclaimer:</b> Every finding is labelled Observed, Contextual, or Hypothesis. A radar anomaly is a statistical deviation, not a confirmed disaster.</div>", unsafe_allow_html=True)

    with spatial_tab:
        st.subheader(f"Spatial Visualization · {selected_date}")
        controls = st.columns([1.6, 1])
        with controls[0]:
            layer = st.selectbox("Map layer", LAYERS, key="map_layer")
        with controls[1]:
            opacity = st.slider("Map opacity", 0.1, 1.0, 0.9, 0.05, key="map_opacity")
        map_col, compare_col = st.columns([1.15, .85], gap="large")
        with map_col:
            _render_map(data, index, layer, opacity)
            if layer == "Verdict Map":
                st.caption("Green = expected · yellow = unusual but explained · red = unexplained candidate event")
            elif layer == "Bidirectional Z-Score Map":
                st.caption("Blue = negative Z · white = near expected · red = positive Z. Bounds: 9.30–9.50°N, 76.30–76.50°E.")
            else:
                st.caption("Backscatter intensity uses a shared −22 to −4 dB scale.")
        with compare_col:
            st.markdown("#### Expected vs. actual")
            st.caption("Shared scale: −22 to −4 dB")
            expected = _rgb(data["predicted"].isel(time=index).values)
            actual = _rgb(data["observed"].isel(time=index).values)
            try:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=expected,
                    img2=actual,
                    label1="Expected dB",
                    label2="Observed dB",
                    width=780,
                    in_memory=True,
                )
            except Exception:
                a, b = st.columns(2)
                a.image(expected, caption="Expected", use_container_width=True)
                b.image(actual, caption="Actual", use_container_width=True)


if __name__ == "__main__":
    main()
