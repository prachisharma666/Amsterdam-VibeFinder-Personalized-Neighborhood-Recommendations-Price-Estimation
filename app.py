import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import joblib

# ── 1. Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amsterdam Stay Planner",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background-color: #faf7f2; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    color: #eee;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #ddd !important; }
[data-testid="stSidebar"] h2 { color: #fff !important; }

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #FF5A5F, #c0392b);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Metric card */
.price-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    color: white;
    margin-top: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.price-card .label { font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.4rem; }
.price-card .amount { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #FF5A5F; }
.price-card .sub { font-size: 0.78rem; opacity: 0.55; margin-top: 0.3rem; }

/* Rating stars */
.stars { color: #f39c12; font-size: 1.1rem; }

/* Neighbourhood tag */
.hood-tag {
    display: inline-block;
    background: #FF5A5F22;
    color: #c0392b;
    border: 1px solid #FF5A5F55;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px;
}

/* Info box */
.info-box {
    background: #fff;
    border-left: 4px solid #FF5A5F;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Section header */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
    color: #1a1a2e;
}

/* Model metrics row */
.metric-row { display: flex; gap: 12px; margin-top: 0.6rem; }
.metric-pill {
    flex: 1;
    background: #fff;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.metric-pill .m-val { font-weight: 700; font-size: 1.05rem; color: #1a1a2e; }
.metric-pill .m-lbl { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ── 3. Load Assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("airbnb_model_1.pkl")
        preprocessor = joblib.load("preprocessor_1.pkl")
        return model, preprocessor
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_geo():
    try:
        return gpd.read_file("neighbourhoods.geojson")
    except Exception as e:
        return None

model, preprocessor = load_model()
gdf = load_geo()

# ── 4. Tourism Interest Mapping ───────────────────────────────────────────────
interest_map = {
    "🏛️ Historical Sites & Old City": {
        "hoods": ["Centrum-West", "Centrum-Oost"],
        "desc": "The medieval heart of Amsterdam — canals, the Anne Frank House, and Dam Square.",
        "emoji": "🏛️",
    },
    "🎨 Museums & Art": {
        "hoods": ["Zuid", "Centrum-Oost"],
        "desc": "Home to the Rijksmuseum, Van Gogh Museum, and Stedelijk. A cultural goldmine.",
        "emoji": "🎨",
    },
    "🎶 Nightlife & Entertainment": {
        "hoods": ["Centrum-West", "De Pijp - Rivierenbuurt"],
        "desc": "Live music, clubs, and vibrant bar scenes that go well past midnight.",
        "emoji": "🎶",
    },
    "🦁 Zoo & Nature": {
        "hoods": ["Oud-Oost", "Watergraafsmeer"],
        "desc": "Close to Artis Zoo, Hortus Botanicus, and serene parks.",
        "emoji": "🦁",
    },
    "☕ Local & Trendy": {
        "hoods": ["De Baarsjes - Oud-West", "Westerpark"],
        "desc": "Independent cafes, vintage shops, and Amsterdam's creative crowd.",
        "emoji": "☕",
    },
    "🌿 Quiet & Residential": {
        "hoods": ["Buitenveldert - Zuidas", "IJburg - Zeeburgereiland"],
        "desc": "Peaceful neighbourhoods away from tourist buzz — ideal for a relaxing stay.",
        "emoji": "🌿",
    },
}

# All neighbourhoods from geojson (for fallback display)
ALL_HOODS = [
    'Bijlmer-Oost', 'Oud-Noord', 'Noord-Oost', 'Noord-West',
    'IJburg - Zeeburgereiland', 'Centrum-West',
    'Oostelijk Havengebied - Indische Buurt', 'Centrum-Oost',
    'Oud-Oost', 'Westerpark', 'Watergraafsmeer', 'Gaasperdam - Driemond',
    'Bijlmer-Centrum', 'De Pijp - Rivierenbuurt', 'Zuid',
    'Buitenveldert - Zuidas', 'De Baarsjes - Oud-West', 'Bos en Lommer',
    'Geuzenveld - Slotermeer', 'Slotervaart', 'Osdorp', 'De Aker - Nieuw Sloten',
]

# ── 5. Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Your Preferences")
    st.markdown("---")

    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
    )

    st.markdown("**Stay Details**")
    accommodates = st.slider("Guests", 1, 16, 2)
    bedrooms    = st.number_input("Bedrooms", 0, 10, 1)
    bathrooms   = st.number_input("Bathrooms", 0.0, 10.0, 1.0, 0.5)
    beds        = st.number_input("Beds", 1, 20, 1)
    min_nights  = st.number_input("Minimum Nights", 1, 30, 2)

    st.markdown("---")
    st.markdown("**Review Scores (1–5 scale)**")
    review_score_val   = st.slider("Overall Value", 1.0, 5.0, 4.5, 0.1)
    review_score_clean = st.slider("Cleanliness",   1.0, 5.0, 4.7, 0.1)
    review_score_loc   = st.slider("Location",      1.0, 5.0, 4.8, 0.1)
    review_score_rat   = st.slider("Rating",        1.0, 5.0, 4.6, 0.1)

    st.markdown("---")
    instant_bookable       = st.checkbox("⚡ Instant Bookable", value=True)
    host_identity_verified = st.checkbox("✅ Host Identity Verified", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Model: OLS Regression with polynomial terms<br>"
        "R² ≈ 0.68 | MAE ≈ €28 | RMSE ≈ €38</small>",
        unsafe_allow_html=True,
    )

# ── 6. Header ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.6rem;color:#1a1a2e;margin-bottom:0'>Amsterdam Stay Planner 🌷</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666;font-size:1rem;margin-top:0.2rem'>Find the right neighbourhood based on your travel interests, then predict the Airbnb price.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── 7. Interest Selector ──────────────────────────────────────────────────────
st.markdown("<div class='section-title'>What brings you to Amsterdam?</div>", unsafe_allow_html=True)

vibe_cols = st.columns(3)
vibe_keys = list(interest_map.keys())
selected_vibe = st.session_state.get("selected_vibe", vibe_keys[0])

# Render interest buttons
for idx, vibe in enumerate(vibe_keys):
    col = vibe_cols[idx % 3]
    with col:
        active = selected_vibe == vibe
        border = "2px solid #FF5A5F" if active else "1px solid #ddd"
        bg     = "#fff5f5" if active else "#fff"
        if st.button(
            vibe,
            key=f"vibe_{idx}",
            use_container_width=True,
            help=interest_map[vibe]["desc"],
        ):
            st.session_state["selected_vibe"] = vibe
            selected_vibe = vibe

vibe_info = interest_map[selected_vibe]
recommended_hoods = vibe_info["hoods"]

st.markdown(
    f"<div class='info-box'>📍 <b>Recommended areas:</b> {', '.join(recommended_hoods)}<br>"
    f"<span style='color:#555'>{vibe_info['desc']}</span></div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── 8. Map + Prediction Panel ─────────────────────────────────────────────────
col_map, col_pred = st.columns([3, 2], gap="large")

with col_map:
    st.markdown("<div class='section-title'>Neighbourhood Map</div>", unsafe_allow_html=True)
    st.caption("Highlighted areas match your selected interest.")

    m = folium.Map(location=[52.3676, 4.9041], zoom_start=12, tiles="CartoDB positron")

    if gdf is not None:
        def style_function(feature):
            name = feature["properties"]["neighbourhood"]
            is_target = name in recommended_hoods
            return {
                "fillColor": "#FF5A5F" if is_target else "#adb5bd",
                "color": "#1a1a2e" if is_target else "#868e96",
                "weight": 2 if is_target else 0.8,
                "fillOpacity": 0.65 if is_target else 0.15,
            }

        def highlight_function(feature):
            return {"fillOpacity": 0.9, "weight": 3, "color": "#c0392b"}

        folium.GeoJson(
            gdf,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["neighbourhood"],
                aliases=["Neighbourhood:"],
                sticky=False,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=430, returned_objects=[])

with col_pred:
    st.markdown("<div class='section-title'>Predict Your Price</div>", unsafe_allow_html=True)

    # Neighbourhood selector — only show hoods that exist in the geojson
    valid_hoods = [h for h in recommended_hoods if h in ALL_HOODS]
    if not valid_hoods:
        valid_hoods = ALL_HOODS[:3]

    hood_choice = st.selectbox("Select a neighbourhood", valid_hoods)

    # Star rating display
    rating_display = round(review_score_rat * 2, 1)  # convert 1-5 → 1-10 for display
    stars = "★" * int(review_score_rat) + ("½" if review_score_rat % 1 >= 0.5 else "")
    st.markdown(
        f"<div style='margin:0.4rem 0'>"
        f"<span class='stars'>{stars}</span> "
        f"<span style='color:#555;font-size:0.9rem'>{review_score_rat:.1f}/5 — {hood_choice}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Model performance pills
    st.markdown(
        "<div class='metric-row'>"
        "  <div class='metric-pill'><div class='m-val'>~0.68</div><div class='m-lbl'>R²</div></div>"
        "  <div class='metric-pill'><div class='m-val'>€28</div><div class='m-lbl'>MAE</div></div>"
        "  <div class='metric-pill'><div class='m-val'>€38</div><div class='m-lbl'>RMSE</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price per Night", use_container_width=True)

    if predict_btn:
        # Build the same input format the OLS pipeline expects
        # The model uses the preprocessed numeric + one-hot encoded df
        # For the raw predict we re-use the statsmodels formula path,
        # so we build the raw dataframe the preprocessor expects.
        input_raw = pd.DataFrame({
            "accommodates":              [float(accommodates)],
            "bedrooms":                  [float(bedrooms)],
            "bathrooms_count":           [float(bathrooms)],
            "beds":                      [float(beds)],
            "minimum_nights":            [float(min_nights)],
            "maximum_nights":            [30.0],
            "availability_365":          [150.0],
            "reviews_per_month":         [2.5],
            "instant_bookable":          [int(instant_bookable)],
            "host_identity_verified":    [int(host_identity_verified)],
            "host_is_superhost":         [0],
            "review_scores_cleanliness": [review_score_clean],
            "review_scores_location":    [review_score_loc],
            "review_scores_value":       [review_score_val],
            "review_scores_rating":      [review_score_rat],
            "first_review_days":         [500.0],
            "room_type":                 [room_type],
            "neighbourhood_cleansed":    [hood_choice],
            "accommodates2":             [float(accommodates ** 2)],
            "minimum_nights2":           [float(min_nights ** 2)],
        })

        try:
            if model is None:
                st.error(
                    f"⚠️ Model files not found. Make sure `airbnb_model_1.pkl` and "
                    f"`preprocessor_1.pkl` are in the same folder as `app.py`.\n\n"
                    f"Details: {preprocessor}"
                )
            else:
                # The pipeline: preprocessor → model
                X_transformed = preprocessor.transform(input_raw)
                log_pred = model.predict(X_transformed)[0]
                price = np.expm1(log_pred)  # reverse log1p

                st.markdown(
                    f"<div class='price-card'>"
                    f"  <div class='label'>Estimated Nightly Rate</div>"
                    f"  <div class='amount'>€{price:.0f}</div>"
                    f"  <div class='sub'>{room_type} · {accommodates} guest{'s' if accommodates > 1 else ''} · {hood_choice}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Price range estimate (±15%)
                lo, hi = price * 0.85, price * 1.15
                st.caption(f"Typical market range: €{lo:.0f} – €{hi:.0f} / night")

        except Exception as e:
            # Friendly error with guidance
            st.error(
                f"**Prediction Error:** {e}\n\n"
                "**Common fixes:**\n"
                "- Make sure the model was trained and saved with `joblib.dump(lmfit1, 'airbnb_model_1.pkl')`\n"
                "- Make sure `preprocessor_1.pkl` is the `ColumnTransformer` fitted on the same features\n"
                "- The neighbourhood you selected must exist in the training data"
            )

# ── 9. Neighbourhood Guide ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Amsterdam Neighbourhood Guide</div>", unsafe_allow_html=True)

guide = {
    "Centrum-West":                          ("Old Centre / Canal Ring",      "€130–€200", "Historical sights, Anne Frank House, Jordaan"),
    "Centrum-Oost":                          ("Old Centre East",               "€120–€185", "Red Light District, Nieuwmarkt, nightlife"),
    "De Pijp - Rivierenbuurt":               ("De Pijp",                       "€100–€155", "Albert Cuyp market, diverse dining"),
    "Zuid":                                  ("Museum Quarter / South",        "€115–€170", "Rijksmuseum, Vondelpark, upscale shopping"),
    "Westerpark":                            ("Westerpark",                    "€90–€135",  "Creative scene, food halls, indie culture"),
    "De Baarsjes - Oud-West":               ("Oud-West",                      "€85–€130",  "Trendy local cafes, vintage shops"),
    "Buitenveldert - Zuidas":               ("Zuidas / Financial District",   "€95–€145",  "Business hub, quiet, good transport links"),
    "IJburg - Zeeburgereiland":             ("IJburg",                        "€75–€115",  "Modern waterside living, peaceful"),
    "Oud-Oost":                              ("Oud-Oost",                      "€90–€140",  "Near Artis Zoo, multicultural charm"),
    "Watergraafsmeer":                       ("Watergraafsmeer",               "€70–€110",  "Residential green area, family-friendly"),
    "Oostelijk Havengebied - Indische Buurt":("Eastern Docklands",             "€85–€130",  "Industrial chic, modern architecture"),
    "Westerpark":                            ("Westerpark",                    "€90–€135",  "Events, nature, artisan food"),
}

g_cols = st.columns(3)
for i, (hood, (zone, price_rng, notes)) in enumerate(guide.items()):
    with g_cols[i % 3]:
        is_rec = hood in recommended_hoods
        border = "border-left: 4px solid #FF5A5F;" if is_rec else ""
        badge  = " 🌷" if is_rec else ""
        st.markdown(
            f"<div style='background:#fff;border-radius:10px;padding:0.85rem 1rem;margin-bottom:0.7rem;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.07);{border}'>"
            f"<b style='color:#1a1a2e'>{hood}{badge}</b><br>"
            f"<span style='color:#888;font-size:0.78rem'>{zone}</span><br>"
            f"<span style='color:#FF5A5F;font-weight:600;font-size:0.88rem'>{price_rng}/night</span><br>"
            f"<span style='color:#555;font-size:0.82rem'>{notes}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.78rem;margin-top:1.5rem'>"
    "Price ranges are indicative estimates. Actual prices depend on season, availability, and host settings."
    "</p>",
    unsafe_allow_html=True,
)
