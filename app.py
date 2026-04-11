import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import joblib
import pickle

# 1. Page configuration
st.set_page_config(page_title="Amsterdam Stay Planner", page_icon="🌷", layout="wide")

# 2. Asset Loading
@st.cache_resource
def load_assets():
    try:
        # Integrated the new underscored names
        model = joblib.load("airbnb_model_1.pkl")
        preprocessor = joblib.load("preprocessor_1.pkl")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, preprocessor = load_assets()

@st.cache_data
def load_geo():
    return gpd.read_file('neighbourhoods.geojson')

try:
    gdf = load_geo()
except Exception as e:
    st.error(f"Error loading map data: {e}")
    gdf = None

# 3. Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        background-color: #FF5A5F; 
        color: white; 
        font-weight: bold; 
        height: 3em;
    }
    </style>
    """, unsafe_with_html=True)

# 4. SIDEBAR: Accommodation Filters
with st.sidebar:
    st.header("🏠 Your Preferences")
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    review_score = st.slider("Minimum Rating (0-10)", 0.0, 10.0, 8.5, 0.1)
    
    with st.expander("Stay Details", expanded=True):
        accommodates = st.number_input("Number of Guests", 1, 16, 2)
        bedrooms = st.number_input("Bedrooms", 0, 10, 1)
        bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0)
        beds = st.number_input("Beds", 1, 20, 1)
        min_nights = st.number_input("Minimum Nights", 1, 30, 2)

# 5. MAIN PANEL: Tourism Interests
st.title("Amsterdam Trip Planner & Price Predictor 🌷")

interest_map = {
    "Historical Sites & Old City": ["Centrum-West", "Centrum-Oost"],
    "Museums & Art": ["Zuid", "Centrum-Oost"],
    "Nightlife & Entertainment": ["Centrum-West", "De Pijp - Rivierenbuurt"],
    "Zoo & Nature": ["Oud-Oost", "Watergraafsmeer"],
    "Local & Trendy": ["De Baarsjes - Oud-West", "Westerpark"],
    "Quiet & Residential": ["Buitenveldert - Zuidas", "IJburg - Zeeburgereiland"]
}

selected_vibe = st.selectbox("What would you like to explore?", list(interest_map.keys()))
recommended_hoods = interest_map[selected_vibe]

col_map, col_res = st.columns([2, 1])

with col_map:
    m = folium.Map(location=[52.3676, 4.9041], zoom_start=12, tiles="CartoDB positron")
    
    if gdf is not None:
        def style_function(feature):
            name = feature['properties']['neighbourhood']
            is_target = name in recommended_hoods
            return {
                'fillColor': '#FF5A5F' if is_target else '#ced4da',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7 if is_target else 0.1,
            }
        folium.GeoJson(gdf, style_function=style_function).add_to(m)
    
    st_folium(m, width=700, height=450)

with col_res:
    st.subheader("Recommendation")
    hood_choice = st.selectbox("Select a neighborhood:", recommended_hoods)
    
    predict_btn = st.button("Predict Price")

    if predict_btn:
        input_df = pd.DataFrame({
            'accommodates': [float(accommodates)],
            'bedrooms': [float(bedrooms)],
            'bathrooms_count': [float(bathrooms)],
            'beds': [float(beds)],
            'availability_365': [150.0], 
            'reviews_per_month': [2.5],
            'instant_bookable': [1],
            'host_identity_verified': [1],
            'review_scores_cleanliness': [9.5],
            'review_scores_location': [9.5],
            'review_scores_value': [float(review_score)],
            'first_review_days': [500.0],
            'room_type': [room_type], 
            'neighbourhood_cleansed': [hood_choice],
            'accommodates2': [float(accommodates**2)],
            'minimum_nights2': [float(min_nights**2)]
        })

        try:
            if model is not None:
                prediction = model.predict(input_df)
                st.markdown("---")
                st.success(f"Estimated Price per Night:")
                st.metric("", f"€{prediction[0]:.2f}")
            else:
                st.error("Model assets not loaded. Check GitHub filenames.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
