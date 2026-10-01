"""Police <-> Waze accident matching: annotation tool (public demo).

A trimmed copy of the tool used to build ground truth for matching police
accident records with Waze accident reports. Every record here is SYNTHETIC:
generated at random on start, nothing comes from real police or Waze data.

Run locally:  streamlit run app.py
In the browser it runs through stlite (see index.html).
"""
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Police ↔ Waze matching · demo", layout="wide")

# Brno-sized box; points are random, not real places of accidents
LAT0, LAT1, LON0, LON1 = 49.165, 49.235, 16.56, 16.67
MONTH_START = datetime(2024, 10, 1)
N_POLICE = 40
N_NOISE = 320


# =====================
# SYNTHETIC DATA
# =====================
@st.cache_data
def make_data(seed: int):
    rng = np.random.default_rng(seed)

    minutes = rng.integers(0, 31 * 24 * 60, N_POLICE)
    police = pd.DataFrame({
        "accident_id": [f"P-{seed:02d}-{i:03d}" for i in range(N_POLICE)],
        "lat": rng.uniform(LAT0, LAT1, N_POLICE),
        "lon": rng.uniform(LON0, LON1, N_POLICE),
        "datetime": [MONTH_START + timedelta(minutes=int(m)) for m in minutes],
        "injuries": rng.choice(["none", "light", "serious"], N_POLICE, p=[.6, .32, .08]),
        "vehicles": rng.integers(1, 4, N_POLICE),
    })
    # police records sometimes carry an impossible time such as 25:60
    bad_time = rng.random(N_POLICE) < 0.15
    police["time"] = np.where(bad_time, "25:60", police["datetime"].dt.strftime("%H:%M"))
    police.loc[bad_time, "datetime"] = police.loc[bad_time, "datetime"].dt.normalize()

    rows = []
    for _, p in police.iterrows():
        if rng.random() > 0.65:          # not every accident gets reported in Waze
            continue
        for _ in range(rng.integers(1, 4)):
            dist_km = min(rng.exponential(0.25), 0.95)
            angle = rng.uniform(0, 2 * np.pi)
            rows.append({
                "lat": p.lat + dist_km * np.cos(angle) / 111.2,
                "lon": p.lon + dist_km * np.sin(angle) / (111.2 * np.cos(np.radians(p.lat))),
                "datetime": p.datetime + timedelta(minutes=float(rng.normal(25, 40))),
            })
    noise_min = rng.integers(0, 31 * 24 * 60, N_NOISE)
    rows += [{
        "lat": rng.uniform(LAT0, LAT1),
        "lon": rng.uniform(LON0, LON1),
        "datetime": MONTH_START + timedelta(minutes=int(m)),
    } for m in noise_min]

    waze = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    waze.insert(0, "uuid", [f"W-{seed:02d}-{i:04d}" for i in range(len(waze))])
    return police, waze


def haversine_km(lat, lon, lat2, lon2):
    lat, lon, lat2, lon2 = map(np.radians, (lat, lon, lat2, lon2))
    a = np.sin((lat2 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat2) * np.sin((lon2 - lon) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# =====================
# TIME MATCH LOGIC
# =====================
def is_whole_day(police_row):
    return police_row["time"] == "25:60"


def candidates(police_row, waze, max_km, window_h):
    out = waze.copy()
    out["distance_km"] = haversine_km(police_row["lat"], police_row["lon"], out["lat"], out["lon"])
    out = out[out["distance_km"] <= max_km]
    if is_whole_day(police_row):
        # no usable time: accept the day before, the day itself and the day after
        day = police_row["datetime"].date()
        start = datetime.combine(day - timedelta(days=1), time.min)
        end = datetime.combine(day + timedelta(days=1), time.max)
        out = out[(out["datetime"] >= start) & (out["datetime"] <= end)]
    else:
        out = out[(out["datetime"] - police_row["datetime"]).abs() <= timedelta(hours=window_h)]
    return out.sort_values("distance_km").reset_index(drop=True)


# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.header("Matching rules")
    max_km = st.slider("Max distance (km)", 0.2, 2.0, 1.0, 0.1)
    window_h = st.slider("Time window (± hours)", 1, 8, 5)
    seed = st.number_input("Synthetic dataset", 1, 99, 7,
                           help="Each number generates a different random month of accidents.")
    st.caption("Police records with an impossible time (25:60) are matched "
               "against a whole three-day window instead.")

police, waze = make_data(int(seed))

ss = st.session_state
if ss.get("seed") != seed:
    # open on the first accident that has something to inspect
    first = next((i for i in range(len(police)) if not is_whole_day(police.iloc[i])
                  and len(candidates(police.iloc[i], waze, max_km, window_h)) > 2), 0)
    ss.seed, ss.police_idx, ss.waze_idx, ss.annotations = seed, first, 0, {}

st.title("🚓 Police ↔ Waze – Accident Annotation Tool")
st.warning("**Synthetic data.** Every accident and every Waze report here is randomly "
           "generated. The demo shows how the annotation tool works, not real events.",
           icon="🧪")



def step_waze(step):
    ss.waze_idx += step


def step_police(step):
    ss.police_idx, ss.waze_idx = ss.police_idx + step, 0


police_row = police.iloc[ss.police_idx]
cand = candidates(police_row, waze, max_km, window_h)
if ss.waze_idx >= len(cand):
    ss.waze_idx = 0

st.subheader(f"Police accident {ss.police_idx + 1} / {len(police)}")
st.dataframe(
    police_row[["accident_id", "datetime", "time", "injuries", "vehicles", "lat", "lon"]].to_frame().T,
    hide_index=True, width="stretch")
if is_whole_day(police_row):
    st.info("Police time is invalid (25:60), so matching uses ±1 day.")
else:
    st.info(f"Time window ±{window_h} h · radius {max_km:.1f} km · {len(cand)} candidate reports")

left, right = st.columns([1.4, 1])

# =====================
# LEFT: MAP
# =====================
with left:
    cand_map = cand.assign(current=cand.index == ss.waze_idx)
    layers = [
        pdk.Layer("ScatterplotLayer", data=pd.DataFrame([police_row[["lat", "lon"]]]),
                  get_position="[lon, lat]", get_radius=max_km * 1000, stroked=True, filled=False,
                  get_line_color=[214, 39, 40], line_width_min_pixels=1.5),
        pdk.Layer("ScatterplotLayer", data=pd.DataFrame([police_row[["lat", "lon"]]]),
                  get_position="[lon, lat]", get_radius=45, radius_min_pixels=7,
                  get_fill_color=[214, 39, 40, 255]),
        pdk.Layer("ScatterplotLayer", data=cand_map[~cand_map.current],
                  get_position="[lon, lat]", get_radius=40, radius_min_pixels=5,
                  get_fill_color=[31, 119, 180, 220], pickable=True),
        pdk.Layer("ScatterplotLayer", data=cand_map[cand_map.current],
                  get_position="[lon, lat]", get_radius=60, radius_min_pixels=8,
                  get_fill_color=[255, 127, 14, 255], get_line_color=[255, 255, 255],
                  stroked=True, line_width_min_pixels=2, pickable=True),
    ]
    zoom = 13.2 - np.log2(max(max_km, 0.2))
    st.pydeck_chart(pdk.Deck(
        layers=layers, map_style="light",
        initial_view_state=pdk.ViewState(latitude=police_row["lat"], longitude=police_row["lon"], zoom=zoom),
        tooltip={"text": "{uuid}\n{distance_km} km"},
    ), height=480)
    st.caption("Red: police record and search radius · blue: Waze reports in the window · "
               "orange: the one being inspected")

# =====================
# RIGHT: ANNOTATION
# =====================
with right:
    st.subheader("Waze report")
    if cand.empty:
        st.info("No Waze report fits the rules. Save as *no match*.")
        waze_row = None
    else:
        waze_row = cand.iloc[ss.waze_idx]
        delta = (waze_row["datetime"] - police_row["datetime"]).total_seconds() / 60
        st.markdown(
            f"**{waze_row['uuid']}** &nbsp;·&nbsp; {ss.waze_idx + 1} of {len(cand)}  \n"
            f"Reported {waze_row['datetime']:%d.%m. %H:%M} ({delta:+.0f} min)  \n"
            f"Distance {waze_row['distance_km'] * 1000:.0f} m")
        c1, c2 = st.columns(2)
        c1.button("← Previous", disabled=ss.waze_idx == 0, width="stretch",
                  on_click=step_waze, args=(-1,))
        c2.button("Next →", disabled=ss.waze_idx >= len(cand) - 1, width="stretch",
                  on_click=step_waze, args=(1,))

    match = st.checkbox("Same accident", disabled=waze_row is None,
                        key=f"m_{seed}_{ss.police_idx}_{ss.waze_idx}")
    sure = st.checkbox("I am sure", disabled=not match,
                       key=f"s_{seed}_{ss.police_idx}_{ss.waze_idx}")
    match = match and waze_row is not None
    if st.button("Save annotation", type="primary", width="stretch"):
        ss.annotations[police_row["accident_id"]] = {
            "police_id": police_row["accident_id"],
            "waze_id": waze_row["uuid"] if match else "",
            "match": int(match),
            "sure": int(sure and match),
        }
        st.toast("Saved")

st.divider()
p1, p2, p3 = st.columns([1, 1, 2])
p1.button("← Previous accident", disabled=ss.police_idx == 0, width="stretch",
          on_click=step_police, args=(-1,))
p2.button("Next accident →", disabled=ss.police_idx >= len(police) - 1, width="stretch",
          on_click=step_police, args=(1,))

done = pd.DataFrame(ss.annotations.values(), columns=["police_id", "waze_id", "match", "sure"])
with p3:
    m1, m2, m3 = st.columns(3)
    m1.metric("Annotated", f"{len(done)} / {len(police)}")
    m2.metric("Matched", int(done["match"].sum()) if len(done) else 0)
    m3.metric("Sure", int(done["sure"].sum()) if len(done) else 0)

if len(done):
    with st.expander("Annotations so far"):
        st.dataframe(done, hide_index=True, width="stretch")
        st.download_button("Download CSV", done.to_csv(index=False), "annotations.csv", "text/csv")
