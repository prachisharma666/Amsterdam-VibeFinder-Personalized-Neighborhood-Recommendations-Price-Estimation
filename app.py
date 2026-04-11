import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import joblib

# Pagina-instellingen
st.set_page_config(page_title="Amsterdam Verblijf Planner", page_icon="🌷", layout="wide")

# Bestanden laden
@st.cache_resource
def load_assets():
    # We laden alleen het model omdat smf.ols de preprocessing intern regelt
    model = sm.load("airbnb_model.pkl")
    return model

@st.cache_data
def load_geo():
    return gpd.read_file('neighbourhoods.geojson')

model = load_assets()
gdf = load_geo()

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #FF5A5F; color: white; font-weight: bold; }
    </style>
    """, unsafe_with_html=True)

# BUURTEN LIJST (gebaseerd op jouw array)
buurten_lijst = [
    'Centrum-West', 'Centrum-Oost', 'Bos en Lommer', 'Slotervaart',
    'De Pijp - Rivierenbuurt', 'De Baarsjes - Oud-West', 'Zuid',
    'Oud-Oost', 'Westerpark', 'Oostelijk Havengebied - Indische Buurt',
    'Noord-Oost', 'Buitenveldert - Zuidas', 'Bijlmer-Oost',
    'Watergraafsmeer', 'Oud-Noord', 'Geuzenveld - Slotermeer',
    'IJburg - Zeeburgereiland', 'Noord-West', 'De Aker - Nieuw Sloten',
    'Osdorp', 'Bijlmer-Centrum', 'Gaasperdam - Driemond'
]

# SIDEBAR: Accommodatie Filters
with st.sidebar:
    st.header("🏠 Jouw Wensen")
    room_type = st.selectbox("Type Kamer", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    review_score = st.slider("Minimale Waardering (0-10)", 0.0, 10.0, 8.5, 0.1)
    
    with st.expander("Details van het verblijf"):
        accommodates = st.number_input("Aantal Gasten", 1, 16, 2)
        bedrooms = st.number_input("Slaapkamers", 0, 10, 1)
        bathrooms = st.number_input("Badkamers", 0.0, 10.0, 1.0)
        beds = st.number_input("Bedden", 1, 20, 1)
        min_nights = st.number_input("Minimaal aantal nachten", 1, 30, 2)

# HOOFDPANEEL: Toerisme Belangen
st.title("Amsterdam Reis Planner & Prijsvoorspeller 🌷")
st.write("Ontdek de perfecte buurt op basis van jouw interesses en bekijk de verwachte prijs.")

# Mapping van Interesses naar Buurten (gebaseerd op amsterdamsights.com)
interesse_map = {
    "Historische Plekken & Centrum": ["Centrum-West", "Centrum-Oost"],
    "Musea & Kunst": ["Zuid", "Centrum-Oost"],
    "Nachtleven & Entertainment": ["Centrum-West", "De Pijp - Rivierenbuurt"],
    "Dierentuin & Natuur": ["Oud-Oost", "Watergraafsmeer"],
    "Lokaal & Hip": ["De Baarsjes - Oud-West", "Westerpark"],
    "Rust & Residentieel": ["Buitenveldert - Zuidas", "IJburg - Zeeburgereiland"]
}

col_top1, col_top2 = st.columns([1, 1])
with col_top1:
    selected_vibe = st.selectbox("Wat wil je bezoeken?", list(interesse_map.keys()))
    aanbevolen_buurten = interesse_map[selected_vibe]

# Interactieve Kaart
col_map, col_res = st.columns([2, 1])

with col_map:
    m = folium.Map(location=[52.3676, 4.9041], zoom_start=12, tiles="CartoDB positron")
    
    def style_function(feature):
        name = feature['properties']['neighbourhood']
        is_target = name in aanbevolen_buurten
        return {
            'fillColor': '#FF5A5F' if is_target else '#ced4da',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7 if is_target else 0.1,
        }

    folium.GeoJson(gdf, style_function=style_function).add_to(m)
    st_folium(m, width=700, height=450)

with col_res:
    st.subheader("Aanbeveling")
    buurt_keuze = st.selectbox("Kies een specifieke buurt uit de selectie:", aanbevolen_buurten)
    
    predict_btn = st.button("Voorspel Prijs")

    if predict_btn:
        # Dataframe maken voor statsmodels predictie
        # Let op: De namen van de kolommen MOETEN exact overeenkomen met je formule!
        input_df = pd.DataFrame({
            'accommodates': [accommodates],
            'bedrooms': [bedrooms],
            'bathrooms_count': [bathrooms],
            'beds': [beds],
            'availability_365': [150],
            'reviews_per_month': [2.5],
            'instant_bookable': [1],
            'host_identity_verified': [1],
            'review_scores_cleanliness': [9.5],
            'review_scores_location': [9.5],
            'review_scores_value': [review_score],
            'first_review_days': [500],
            'room_type': [room_type],
            'neighbourhood_cleansed': [buurt_keuze],
            'accommodates2': [accommodates**2],
            'minimum_nights2': [min_nights**2]
        })

        try:
            # Statsmodels formule-gebaseerde modellen kunnen direct predicten op de ruwe DF
            prediction = model.predict(input_df)
            
            st.markdown("---")
            st.metric(label="Geschatte Prijs per Nacht", value=f"€{prediction[0]:.2f}")
            st.info(f"Deze prijs is gebaseerd op een verblijf in **{buurt_keuze}** met een waardering van **{review_score}**.")
        except Exception as e:
            st.error(f"Fout bij voorspelling: {e}")
            st.write("Controleer of de kolomnamen in je dataframe overeenkomen met de modelformule.")
