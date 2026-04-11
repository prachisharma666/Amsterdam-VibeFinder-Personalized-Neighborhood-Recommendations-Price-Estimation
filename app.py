import streamlit as st
import pandas as pd
import statsmodels.api as sm

# Load the model
model = sm.load("airbnb_model.pkl")

# ... (UI code for inputs: accommodates, bedrooms, etc.) ...

if st.button("Predict"):
    # 1. Create a raw DataFrame with the EXACT names used in your formula
    # Statsmodels formula API will handle the 'Q' encoding automatically
    input_df = pd.DataFrame({
        'accommodates': [accommodates],
        'bedrooms': [bedrooms],
        'bathrooms_count': [bathrooms_count],
        'beds': [beds],
        'availability_365': [150], # Default if not in UI
        'reviews_per_month': [2.0],
        'instant_bookable': [1 if instant_book else 0],
        'host_identity_verified': [1],
        'review_scores_cleanliness': [9.0],
        'review_scores_location': [9.0],
        'review_scores_value': [review_scores_value],
        'first_review_days': [365],
        'room_type': [room_type], # Raw string like 'Private room'
        'neighbourhood_cleansed': [neighbourhood], # Raw string
        # Pre-calculate your squared terms as the formula expects them
        'accommodates2': [accommodates**2],
        'minimum_nights2': [minimum_nights**2]
    })

    # 2. Predict directly using the model
    # The formula-based model knows how to transform this row
    try:
        prediction = model.predict(input_df)
        st.success(f"Estimated Price: €{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
