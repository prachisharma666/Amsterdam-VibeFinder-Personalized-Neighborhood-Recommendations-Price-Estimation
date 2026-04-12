"""
Amsterdam Airbnb Stay Planner — app.py
=======================================
Model pipeline (from your notebook):
  1. preprocessor (ColumnTransformer) expects ALL of:
       bool_cols  = host_is_superhost, host_identity_verified,
                    has_availability, instant_bookable
       num_cols   = latitude, longitude, accommodates, bathrooms_count,
                    bedrooms, beds, minimum_nights, maximum_nights,
                    availability_365, number_of_reviews, reviews_per_month,
                    review_scores_value, review_scores_location,
                    review_scores_cleanliness, review_scores_rating,
                    first_review_days, last_review_days
       cat_cols   = room_type, neighbourhood_cleansed
     -> outputs a scaled/one-hot DataFrame (x_processed)

  2. final_df adds squared terms; lmfit1 (statsmodels OLS) predicts log1p(price).

  Actual model metrics (from your notebook):
    R² = 0.529  (training set)
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amsterdam Stay Planner",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

[data-testid="stSidebar"] { background-color: #1a1a2e; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p { color: #ccc !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #fff !important; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #FF5A5F, #c0392b);
    color: white; font-weight: 600; font-size: 1rem;
    border: none; border-radius: 12px;
    padding: 0.75rem 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.price-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px; padding: 1.6rem 2rem;
    text-align: center; color: white; margin-top: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.price-card .label  { font-size:0.82rem; letter-spacing:0.1em; text-transform:uppercase; opacity:0.65; }
.price-card .amount { font-family:'DM Serif Display',serif; font-size:3rem; color:#FF5A5F; }
.price-card .sub    { font-size:0.78rem; opacity:0.5; margin-top:0.3rem; }

.info-box {
    background:#fff; border-left:4px solid #FF5A5F;
    border-radius:8px; padding:0.8rem 1rem; margin:0.6rem 0;
    font-size:0.9rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
.section-title { font-family:'DM Serif Display',serif; font-size:1.5rem; color:#1a1a2e; margin-bottom:0.3rem; }

.metric-row { display:flex; gap:12px; margin-top:0.6rem; }
.metric-pill { flex:1; background:#fff; border-radius:10px; padding:0.6rem 0.8rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.07); }
.metric-pill .m-val { font-weight:700; font-size:1.05rem; color:#1a1a2e; }
.metric-pill .m-lbl { font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:0.05em; }

.hood-card {
    background:#fff; border-radius:10px; padding:0.85rem 1rem;
    margin-bottom:0.7rem; box-shadow:0 2px 8px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model        = joblib.load("airbnb_model_1.pkl")
        preprocessor = joblib.load("preprocessor_1.pkl")
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_geo():
    try:
        return gpd.read_file("neighbourhoods.geojson")
    except Exception as e:
        return None

model, preprocessor, load_err = load_model()
gdf = load_geo()

# ── Column definitions (exactly as in your notebook Cell 17) ──────────────────
BOOL_COLS = [
    "host_is_superhost", "host_identity_verified",
    "has_availability", "instant_bookable",
]
NUM_COLS = [
    "latitude", "longitude", "accommodates", "bathrooms_count", "bedrooms",
    "beds", "minimum_nights", "maximum_nights", "availability_365",
    "number_of_reviews", "reviews_per_month", "review_scores_value",
    "review_scores_location", "review_scores_cleanliness",
    "review_scores_rating", "first_review_days", "last_review_days",
]
CAT_COLS = ["room_type", "neighbourhood_cleansed"]

# Median fill-values for columns not exposed in the UI
MEDIAN_DEFAULTS = {
    "latitude":          52.3676,
    "longitude":         4.9041,
    "maximum_nights":    365.0,
    "number_of_reviews": 18.0,
    "has_availability":  1,
    "host_is_superhost": 0,
    "last_review_days":  180.0,
}

# Only neighbourhoods present in your training data
TRAINING_NEIGHBOURHOODS = [
    "Centrum-West",
    "De Baarsjes - Oud-West",
    "Bos en Lommer",
    "Buitenveldert - Zuidas",
    "Centrum-Oost",
    "De Pijp - Rivierenbuurt",
    "IJburg - Zeeburgereiland",
    "Geuzenveld - Slotermeer",
    "Noord-Oost",
    "Noord-West",
    "Slotervaart",
    "Watergraafsmeer",
    "Oostelijk Havengebied - Indische Buurt",
    "Oud-Oost",
    "Westerpark",
]

# Tourism interest map (hoods restricted to TRAINING_NEIGHBOURHOODS)
INTEREST_MAP = {
    "🏛️ Historical Sites & Old City": {
        "hoods": ["Centrum-West", "Centrum-Oost"],
        "desc":  "The medieval heart of Amsterdam — canals, Anne Frank House, Dam Square.",
    },
    "🎨 Museums & Art": {
        "hoods": ["Centrum-Oost", "Buitenveldert - Zuidas"],
        "desc":  "Rijksmuseum, Van Gogh Museum, Stedelijk — a cultural goldmine.",
    },
    "🎶 Nightlife & Entertainment": {
        "hoods": ["Centrum-West", "De Pijp - Rivierenbuurt"],
        "desc":  "Live music, clubs, and vibrant bar scenes that go well past midnight.",
    },
    "🦁 Zoo & Nature": {
        "hoods": ["Oud-Oost", "Watergraafsmeer"],
        "desc":  "Close to Artis Zoo, Hortus Botanicus, and serene parks.",
    },
    "☕ Local & Trendy": {
        "hoods": ["De Baarsjes - Oud-West", "Westerpark"],
        "desc":  "Independent cafes, vintage shops, and Amsterdam's creative crowd.",
    },
    "🌿 Quiet & Residential": {
        "hoods": ["Buitenveldert - Zuidas", "IJburg - Zeeburgereiland"],
        "desc":  "Peaceful neighbourhoods away from tourist buzz — ideal for a relaxing stay.",
    },
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Your Preferences")
    st.markdown("---")

    room_type    = st.selectbox("Room Type",
                                ["Entire home/apt", "Private room", "Hotel room"])
    accommodates = st.slider("Guests", 1, 16, 2)
    bedrooms     = st.number_input("Bedrooms",   0,   10,   1)
    bathrooms    = st.number_input("Bathrooms",  0.0, 10.0, 1.0, 0.5)
    beds         = st.number_input("Beds",       1,   20,   1)
    min_nights   = st.number_input("Minimum Nights", 1, 30, 2)
    avail_365    = st.slider("Days Available per Year", 0, 365, 150)

    st.markdown("---")
    st.markdown("**Review Scores (1–5 scale)**")
    rev_value    = st.slider("Value",       1.0, 5.0, 4.5, 0.1)
    rev_clean    = st.slider("Cleanliness", 1.0, 5.0, 4.7, 0.1)
    rev_location = st.slider("Location",   1.0, 5.0, 4.8, 0.1)
    rev_rating   = st.slider("Rating",     1.0, 5.0, 4.6, 0.1)
    reviews_pm   = st.number_input("Reviews per Month", 0.0, 30.0, 2.5, 0.5)
    first_rev_days = st.number_input("Days Since First Review", 0, 5000, 500)

    st.markdown("---")
    instant_bookable       = st.checkbox("⚡ Instant Bookable",      value=True)
    host_identity_verified = st.checkbox("✅ Host Identity Verified", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#777'>"
        "Model: OLS on log₁p(price)<br>"
        "R² = 0.529 (training set)<br>"
        "15 neighbourhoods from training data"
        "</small>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.6rem;"
    "color:#1a1a2e;margin-bottom:0'>Amsterdam Stay Planner 🌷</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666;font-size:1rem;margin-top:0.2rem'>"
    "Choose your travel interest → pick a neighbourhood → get a price estimate.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Interest selector ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>What brings you to Amsterdam?</div>",
            unsafe_allow_html=True)

if "selected_vibe" not in st.session_state:
    st.session_state["selected_vibe"] = list(INTEREST_MAP.keys())[0]

cols_vibe = st.columns(3)
for idx, vibe in enumerate(INTEREST_MAP.keys()):
    with cols_vibe[idx % 3]:
        if st.button(vibe, key=f"vibe_{idx}", use_container_width=True):
            st.session_state["selected_vibe"] = vibe

selected_vibe = st.session_state["selected_vibe"]
vibe_info     = INTEREST_MAP[selected_vibe]
recommended   = vibe_info["hoods"]

st.markdown(
    f"<div class='info-box'>📍 <b>Recommended areas:</b> {', '.join(recommended)}<br>"
    f"<span style='color:#555'>{vibe_info['desc']}</span></div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Map + Prediction ──────────────────────────────────────────────────────────
col_map, col_pred = st.columns([3, 2], gap="large")

with col_map:
    st.markdown("<div class='section-title'>Neighbourhood Map</div>",
                unsafe_allow_html=True)
    st.caption("Highlighted in red = recommended for your interest.")

    m = folium.Map(location=[52.3676, 4.9041], zoom_start=12,
                   tiles="CartoDB positron")

    if gdf is not None:
        def style_fn(feature):
            name = feature["properties"]["neighbourhood"]
            hit  = name in recommended
            return {
                "fillColor":   "#FF5A5F" if hit else "#adb5bd",
                "color":       "#1a1a2e" if hit else "#868e96",
                "weight":      2 if hit else 0.8,
                "fillOpacity": 0.65 if hit else 0.15,
            }

        folium.GeoJson(
            gdf,
            style_function=style_fn,
            highlight_function=lambda f: {"fillOpacity": 0.9, "weight": 3},
            tooltip=folium.GeoJsonTooltip(
                fields=["neighbourhood"], aliases=["Neighbourhood:"], sticky=False
            ),
        ).add_to(m)

    st_folium(m, width=None, height=430, returned_objects=[])

with col_pred:
    st.markdown("<div class='section-title'>Price Predictor</div>",
                unsafe_allow_html=True)

    hood_choice = st.selectbox("Select a neighbourhood", recommended)

    st.markdown(
        "<div class='metric-row'>"
        "  <div class='metric-pill'><div class='m-val'>0.529</div>"
        "    <div class='m-lbl'>R² (train)</div></div>"
        "  <div class='metric-pill'><div class='m-val'>log₁p</div>"
        "    <div class='m-lbl'>Target</div></div>"
        "  <div class='metric-pill'><div class='m-val'>OLS</div>"
        "    <div class='m-lbl'>Method</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if load_err:
        st.warning(
            f"⚠️ Could not load model files: `{load_err}`\n\n"
            "Place `airbnb_model_1.pkl` and `preprocessor_1.pkl` next to `app.py`."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Nightly Price", use_container_width=True)

    if predict_btn:
        if model is None or preprocessor is None:
            st.error("Model not loaded — see warning above.")
        else:
            try:
                # Step 1: build the full raw row the preprocessor expects
                raw = pd.DataFrame([{
                    # bool_cols
                    "host_is_superhost":         int(MEDIAN_DEFAULTS["host_is_superhost"]),
                    "host_identity_verified":     int(host_identity_verified),
                    "has_availability":           int(MEDIAN_DEFAULTS["has_availability"]),
                    "instant_bookable":           int(instant_bookable),
                    # num_cols — user-controlled
                    "accommodates":              float(accommodates),
                    "bathrooms_count":           float(bathrooms),
                    "bedrooms":                  float(bedrooms),
                    "beds":                      float(beds),
                    "minimum_nights":            float(min_nights),
                    "availability_365":          float(avail_365),
                    "reviews_per_month":         float(reviews_pm),
                    "review_scores_value":       float(rev_value),
                    "review_scores_location":    float(rev_location),
                    "review_scores_cleanliness": float(rev_clean),
                    "review_scores_rating":      float(rev_rating),
                    "first_review_days":         float(first_rev_days),
                    # num_cols — median-filled (needed by preprocessor, dropped from model formula)
                    "latitude":                  MEDIAN_DEFAULTS["latitude"],
                    "longitude":                 MEDIAN_DEFAULTS["longitude"],
                    "maximum_nights":            MEDIAN_DEFAULTS["maximum_nights"],
                    "number_of_reviews":         MEDIAN_DEFAULTS["number_of_reviews"],
                    "last_review_days":          MEDIAN_DEFAULTS["last_review_days"],
                    # cat_cols
                    "room_type":                 room_type,
                    "neighbourhood_cleansed":    hood_choice,
                }])

                # Enforce correct column order
                raw = raw[BOOL_COLS + NUM_COLS + CAT_COLS]

                # Step 2: run the preprocessor
                x_proc = preprocessor.transform(raw)
                x_proc.columns = [c.split("__")[-1] for c in x_proc.columns]

                # Step 3: add squared terms
                x_proc["accommodates2"]   = x_proc["accommodates"] ** 2
                x_proc["minimum_nights2"] = x_proc["minimum_nights"] ** 2

                # Step 4: predict
                log_pred = model.predict(x_proc)[0]

                # Step 5: reverse log1p
                price = np.expm1(log_pred)

                st.markdown(
                    f"<div class='price-card'>"
                    f"  <div class='label'>Estimated Nightly Rate</div>"
                    f"  <div class='amount'>€{price:.0f}</div>"
                    f"  <div class='sub'>{room_type} · {accommodates} guest"
                    f"{'s' if accommodates > 1 else ''} · {hood_choice}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"log₁p prediction: {log_pred:.4f} → expm1 → €{price:.2f}. "
                    "R² = 0.529 means the model explains ~53% of price variance on the training set."
                )

            except Exception as e:
                st.error(f"**Prediction Error:** {e}")
                st.markdown(
                    "**Debug checklist:**\n"
                    "- Saved with `joblib.dump(lmfit1, 'airbnb_model_1.pkl', compress=0)`?\n"
                    "- `preprocessor.set_output(transform='pandas')` called before saving?\n"
                    "- Both `.pkl` files are next to `app.py`?"
                )

# ── Neighbourhood guide ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Neighbourhood Guide</div>",
            unsafe_allow_html=True)
st.caption("Only the 15 neighbourhoods present in the training data are listed.")

HOOD_INFO = {
    "Centrum-West":                               ("Old Centre / Canal Ring",  "Historical canals, Anne Frank House, Jordaan"),
    "Centrum-Oost":                               ("Old Centre East",           "Nieuwmarkt, Red Light District, museums"),
    "De Pijp - Rivierenbuurt":                    ("De Pijp",                   "Albert Cuyp market, diverse dining, buzzy cafes"),
    "Buitenveldert - Zuidas":                     ("Zuidas / South",            "Business district, quiet, excellent transport"),
    "IJburg - Zeeburgereiland":                   ("IJburg",                    "Modern waterside living, peaceful and green"),
    "Oud-Oost":                                   ("Oud-Oost",                  "Near Artis Zoo, multicultural, leafy streets"),
    "Watergraafsmeer":                            ("Watergraafsmeer",           "Residential, family-friendly, lots of green space"),
    "De Baarsjes - Oud-West":                     ("Oud-West",                  "Trendy cafes, vintage shops, creative crowd"),
    "Westerpark":                                 ("Westerpark",                "Events, food halls, artisan culture"),
    "Bos en Lommer":                              ("Bos en Lommer",             "Up-and-coming, affordable, authentic local life"),
    "Geuzenveld - Slotermeer":                    ("Geuzenveld",                "Residential West Amsterdam, budget-friendly"),
    "Noord-Oost":                                 ("Noord-Oost",                "NDSM Wharf, creative arts, ferry from Centraal"),
    "Noord-West":                                 ("Noord-West",                "Industrial-chic, emerging neighbourhood"),
    "Slotervaart":                                ("Slotervaart",               "Quiet western suburb, parks, local feel"),
    "Oostelijk Havengebied - Indische Buurt":     ("Eastern Docklands",         "Modern architecture, waterfront, Java Island"),
}

g_cols = st.columns(3)
for i, (hood, (zone, notes)) in enumerate(HOOD_INFO.items()):
    with g_cols[i % 3]:
        is_rec = hood in recommended
        border = "border-left:4px solid #FF5A5F;" if is_rec else "border-left:4px solid #dee2e6;"
        badge  = " 🌷" if is_rec else ""
        st.markdown(
            f"<div class='hood-card' style='{border}'>"
            f"  <b style='color:#1a1a2e'>{hood}{badge}</b><br>"
            f"  <span style='color:#888;font-size:0.78rem'>{zone}</span><br>"
            f"  <span style='color:#555;font-size:0.82rem'>{notes}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.78rem;margin-top:1rem'>"
    "Predictions based on OLS trained on Amsterdam Airbnb data. "
    "R² = 0.529 (training set). Always verify with live Airbnb listings."
    "</p>",
    unsafe_allow_html=True,
)
