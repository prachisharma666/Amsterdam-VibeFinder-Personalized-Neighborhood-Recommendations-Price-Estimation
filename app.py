import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import folium
from streamlit_folium import st_folium
import geopandas as gpd

# SET PAGE CONFIG (Must be first)
st.set_page_config(page_title="Amsterdam Stay Discovery", page_icon="🌷", layout="wide")

# 1. LOAD ASSETS
@st.cache_resource
def load_models():
    # Load your exported preprocessor and OLS model
    prep = joblib.load('preprocessor.pkl')
    model = sm.load('airbnb_model.pkl')
    return prep, model

@st.cache_data
def load_geo():
    # Load your geojson file
    return gpd.read_file('neighbourhoods.geojson')

preprocessor, model = load_models()
gdf = load_geo()

# 2. CUSTOM CSS FOR STYLING
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #FF5A5F; color: white; }
    .prediction-card { padding: 20px; border-radius: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_with_html_group=True)

# 3. SIDEBAR - Accomodation Filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Amsterdam_logo.svg/2560px-Amsterdam_logo.svg.png", width=150)
    st.header("Accommodation Needs")
    
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Hotel room"])
    rating_val = st.slider("Minimum Quality Rating", 0.0, 10.0, 8.5, 0.5)
    
    with st.expander("Physical Details"):
        accommodates = st.number_input("Guests", 1, 16, 2)
        bedrooms = st.number_input("Bedrooms", 1, 10, 1)
        bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 1.0)
        min_nights = st.number_input("Min Nights", 1, 30, 2)

# 4. MAIN PANEL - Tourism Interests
st.title("Discover Your Amsterdam 🌷")
st.subheader("What are you looking to experience?")

# Mapping Interests -> Neighborhoods (Based on your amsterdamsights.com data)
interest_map = {
    "Historical & Architecture": ["Centrum-West", "Centrum-Oost", "Jordaan"],
    "Nightlife & Fun": ["Centrum-West", "De Pijp - Rivierenbuurt"],
    "Museums & Art": ["Oud-Zuid", "Centrum-Oost"],
    "Zoos & Nature": ["Plantage", "Watergraafsmeer"],
    "Quiet & Residential": ["Oostelijk Havengebied - Indische Buurt", "Buitenveldert - Zuidas"]
}

selected_interest = st.selectbox("Choose a 'Vibe':", list(interest_map.keys()))
target_hoods = interest_map[selected_interest]

# 5. INTERACTIVE MAP
col_map, col_info = st.columns([2, 1])

with col_map:
    st.write(f"Showing recommended areas for **{selected_interest}**")
    
    # Create Folium Map centered on Amsterdam
    m = folium.Map(location=[52.3676, 4.9041], zoom_start=12, tiles="CartoDB positron")
    
    # Highlight recommended neighborhoods in the GeoJSON
    def style_function(feature):
        hood_name = feature['properties']['neighbourhood']
        return {
            'fillColor': '#FF5A5F' if hood_name in target_hoods else '#B0B0B0',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6 if hood_name in target_hoods else 0.1,
        }

    folium.GeoJson(gdf, style_function=style_function).add_to(m)
    st_folium(m, width=700, height=400)

with col_info:
    st.write("### Area Insights")
    st.info(f"The areas highlighted in red are best suited for {selected_interest.lower()}. We've based our price prediction on the most central of these neighborhoods.")
    
    predict_btn = st.button("Calculate Predicted Price")

# 6. PREDICTION LOGIC
if predict_btn:
    # We take the first neighborhood from our interest list for the prediction
    primary_hood = target_hoods[0]
    
    # Create the input dataframe (Must match the exact structure your preprocessor expects)
    input_df = pd.DataFrame({
        'accommodates': [accommodates],
        'bedrooms': [bedrooms],
        'bathrooms_count': [bathrooms],
        'beds': [bedrooms], # Assuming 1 bed per bedroom
        'minimum_nights': [min_nights],
        'accommodates2': [accommodates**2],
        'minimum_nights2': [min_nights**2],
        'review_scores_value': [rating_val],
        'neighbourhood_cleansed': [primary_hood],
        'room_type': [room_type],
        # Add any other dummy variables your model expects as 0 or 1
        'availability_365': [150],
        'reviews_per_month': [2.0],
        'instant_bookable': [1],
        'host_identity_verified': [1],
        'review_scores_cleanliness': [9.0],
        'review_scores_location': [9.0],
        'first_review_days': [300]
    })

    try:
        # Pre-process
        X_proc = preprocessor.transform(input_df)
        X_const = sm.add_constant(X_proc, has_constant='add')
        
        # Predict
        price_pred = model.predict(X_const)[0]
        
        # UI DISPLAY FOR RESULT
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Estimated Nightly Price", value=f"€{price_pred:.2f}")
        with res_col2:
            st.write(f"**Stay Location:** {primary_hood}")
            st.write(f"**Vibe:** {selected_interest}")
            
    except Exception as e:
        st.error(f"Error in prediction: {e}. Ensure your preprocessor match the input columns.")
