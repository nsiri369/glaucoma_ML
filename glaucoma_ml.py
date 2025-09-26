import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Prediction App")

# Load the trained model (ensure 'logreg_model.pkl' exists in the same directory)
try:
    with open('logreg_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions (adjust based on your training notebook)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['No Glaucoma', 'Glaucoma'])  # Update if your model uses different labels

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Example numeric features (replace with your dataset’s features)
age = st.sidebar.slider("Age", min_value=20, max_value=90, value=50)
iop = st.sidebar.slider("Intraocular Pressure (mmHg)", min_value=5.0, max_value=40.0, value=15.0, step=0.1)
cup_disc_ratio = st.sidebar.slider("Cup-to-Disc Ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
visual_field_index = st.sidebar.slider("Visual Field Index (%)", min_value=0, max_value=100, value=85)

# Example categorical features (replace with actual features from your dataset)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", options=["Yes", "No"])

# Function to preprocess input data
def preprocess_input(age, iop, cup_disc_ratio, visual_field_index, gender, family_history):
    data = {
        'Age': age,
        'IOP': iop,
        'CupDiscRatio': cup_disc_ratio,
        'VisualFieldIndex': visual_field_index,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'FamilyHistory_Yes': 1 if family_history == 'Yes' else 0,
        'FamilyHistory_No': 1 if family_history == 'No' else 0
    }
    df = pd.DataFrame([data])

    # Ensure columns order matches training data
    expected_columns = [
        'Age', 'IOP', 'CupDiscRatio', 'VisualFieldIndex',
        'Gender_Female', 'Gender_Male', 'FamilyHistory_No', 'FamilyHistory_Yes'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    input_df = preprocess_input(age, iop, cup_disc_ratio, visual_field_index, gender, family_history)
    try:
        prediction = model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted outcome is: **{predicted_label}**")
        if predicted_label == "No Glaucoma":
            st.write("The patient is not predicted to have glaucoma.")
        else:
            st.write("The patient may have glaucoma.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the patient's details.
2. Adjust the sliders for numerical features like Age, IOP, etc.
3. Select appropriate options for categorical fields.
4. Click the 'Predict' button to see the predicted outcome.
""")
