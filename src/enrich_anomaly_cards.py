import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "results" / "anomaly_cards.json"
CONTEXT_PATH = ROOT / "config" / "env_context.json"
ERA5_SOIL_PATH = ROOT / "data" / "processed" / "era5_land_soil_moisture_kuttanad.json"

# EDIT if your code computes confidence differently
CONFIDENCE_DEFINITION = (
    "Score from 0 to 1 summarising how strongly the evidence supports the verdict "
    "(based on |Z-score| and data quality). Higher = more reliable. "
    "0.8-1.0 high, 0.5-0.8 moderate, below 0.5 low. Not a probability."
)


def parse(d):
    return date.fromisoformat(str(d)[:10])


def crop_phase(d, calendar):
    for p in calendar:
        if "start_month" in p and "end_month" in p:
            start_month = int(p["start_month"])
            end_month = int(p["end_month"])
            in_window = (
                start_month <= d.month <= end_month
                if start_month <= end_month
                else d.month >= start_month or d.month <= end_month
            )
            if in_window:
                return p["phase"]
        elif parse(p["start"]) <= d <= parse(p["end"]):
            return p["phase"]
    return None


def main():
    raw = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    cards = raw["cards"] if isinstance(raw, dict) and "cards" in raw else raw
    ctx = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    era5_soil = {}
    if ERA5_SOIL_PATH.exists():
        era5_soil = json.loads(ERA5_SOIL_PATH.read_text(encoding="utf-8")).get("dates", {})

    window = ctx.get("rainfall_window_days")      # null if unverified
    default_sensor = ctx.get("sensor_default") or {}

    for c in cards:
        d = parse(c["date"])
        key = d.isoformat()
        drivers = c.get("environmental_drivers", {})

        # rainfall may be a number (original) or a dict (already enriched)
        rain_val = drivers.get("rainfall", c.get("rainfall_mm", c.get("rainfall")))
        if isinstance(rain_val, dict):
            rain_val = rain_val.get("value")

        sensor = ctx.get("sensor_by_date", {}).get(key, default_sensor)
        phase = crop_phase(d, ctx.get("crop_calendar", []))
        soil_entry = era5_soil.get(key, {})
        soil_value = ctx.get("soil_moisture_by_date", {}).get(key)
        if soil_value is None:
            soil_value = soil_entry.get("soil_moisture_m3_m3")

        c["environmental_drivers"] = {
            **drivers,
            "rainfall": {
                "value": rain_val,
                "units": "mm",
                "accumulation_window_days": window,
                "window_description": (
                    f"Total rainfall over the {window} days preceding this acquisition"
                    if window else None
                ),
                "source": ctx.get("rainfall_source"),
            },
            "soil_moisture": {
                "value": soil_value,
                "units": "m3/m3",
                "description": "ERA5-Land daily mean volumetric soil water, layer 1 (0-7 cm)",
            },
            "soil_moisture_status": soil_entry.get("status"),
            "crop_phase": phase,
            "crop_phase_status": (
                "Estimated from typical Kuttanad crop-season calendar"
                if phase else None
            ),
            "crop_phase_source": ctx.get("crop_calendar_source_url"),
        }

        c["sensor"] = {
            "name": sensor.get("name"),
            "band": sensor.get("band"),
            "orbit": sensor.get("orbit"),
        }

        # counts already present on the card; "expected" stays null until total pixels are known
        c["class_pixel_counts"] = {
            "expected": None,
            "unusual_but_explained": c.get("explained_pixels"),
            "unexplained_candidate_event": c.get("unexplained_pixels"),
            "pixel_area_m2": ctx.get("pixel_area_m2"),
        }

        c["confidence_definition"] = CONFIDENCE_DEFINITION

    CARDS_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"Enriched {len(cards)} cards -> {CARDS_PATH}")


if __name__ == "__main__":
    main()
