import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
# The model file MUST be committed to your GitHub repo.
model_filename = 'glaucoma_clf_model.pkl'

# Construct a robust path to the model file
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, model_filename)

# Set page title and layout
st.set_page_config(page_title="Glaucoma Risk Predictor", layout="wide")
st.title("Glaucoma Risk Prediction App")
st.markdown("### Predicting Glaucoma Risk based on Patient Metrics")

# --- Load Compressed Model using Joblib ---
if not os.path.exists(model_path):
    st.error(f"🚨 **FATAL ERROR:** The model file '{model_filename}' was not found.")
    st.markdown("Please ensure the **`glaucoma_clf_model.pkl`** file is committed to your repository.")
    st.stop()

try:
    # Load the compressed model
    clf = joblib.load(model_path)
    st.sidebar.success("Prediction Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model using Joblib. Check file integrity. Error: {e}")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
# Mapping: Glaucoma (0), No Glaucoma (1)
label_encoder.classes_ = np.array(['Glaucoma', 'No Glaucoma']) 

# --- Sidebar for user inputs ---
st.sidebar.header("Enter Patient Metrics")

# Numerical features
age = st.sidebar.slider("Age (Years)", min_value=20, max_value=90, value=65)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", min_value=8.0, max_value=30.0, value=15.0, step=0.1)
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", min_value=0.1, max_value=1.0, value=0.6, step=0.01)
pachymetry = st.sidebar.slider("Pachymetry (Corneal Thickness in µm)", min_value=450, max_value=650, value=550) 
sensitivity = st.sidebar.slider("VFT Sensitivity (0-1)", min_value=0.40, max_value=1.00, value=0.75, step=0.01)
specificity = st.sidebar.slider("VFT Specificity (0-1)", min_value=0.70, max_value=1.00, value=0.85, step=0.01)

# Categorical features
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", options=["No", "Yes"])
cataract_status = st.sidebar.selectbox("Cataract Status", options=["Absent", "Present"])
angle_closure = st.sidebar.selectbox("Angle Closure Status", options=["Open", "Closed"])


# Function to preprocess input data (Must match model training logic)
def preprocess_input(age, iop, cdr, pachymetry, sensitivity, specificity,
                     gender, family_history, cataract_status, angle_closure):

    # 1. Create a DataFrame with numerical features
    data = {
        'Age': age,
        'Intraocular Pressure (IOP)': iop,
        'Cup-to-Disc Ratio (CDR)': cdr,
        'Pachymetry': float(pachymetry), 
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }
    df = pd.DataFrame([data])

    # 2. Apply One-Hot Encoding (drop_first=True equivalent)
    df['Gender_Male'] = 1 if str(gender) == 'Male' else 0
    df['Family History_Yes'] = 1 if str(family_history) == 'Yes' else 0
    df['Cataract Status_Present'] = 1 if str(cataract_status) == 'Present' else 0

    # 3. Apply Label Encoding for Angle Closure Status (Closed=0, Open=1)
    df['Angle Closure Status'] = 1 if str(angle_closure) == 'Open' else 0

    # 4. Ensure column order matches the trained model's feature set
    expected_columns = [
        'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)',
        'Pachymetry', 'Angle Closure Status', 'Sensitivity', 'Specificity',
        'Gender_Male', 'Family History_Yes', 'Cataract Status_Present'
    ]
    
    final_df = df.reindex(columns=expected_columns, fill_value=0)
    final_df['Angle Closure Status'] = final_df['Angle Closure Status'].astype(int)
    
    return final_df

# --- Prediction Logic ---
if st.sidebar.button("Predict Glaucoma Risk"):
    
    # Preprocess the input
    input_df = preprocess_input(
        age, iop, cdr, pachymetry, sensitivity, specificity,
        gender, family_history, cataract_status, angle_closure
    )
    
    # Make prediction
    try:
        prediction = clf.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result")
        
        if predicted_label == "No Glaucoma":
            st.success(f"✅ The predicted diagnosis is: **{predicted_label}**")
            st.write("Based on the input metrics, the model suggests a low risk. Continued monitoring is recommended.")
        else:
            st.warning(f"⚠️ The predicted diagnosis is: **{predicted_label}**")
            st.write("The model suggests a high risk. **Immediate consultation with an ophthalmologist is strongly advised.**")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# --- Informational Content ---
st.markdown("""
### Model Feature Summary
This prediction is based on the most influential features identified during model training:

1.  **Pachymetry** (Corneal Thickness)
2.  **Intraocular Pressure (IOP)**
3.  **Age**
4.  **Cup-to-Disc Ratio (CDR)**
5.  **Visual Field Test (VFT) Metrics** (Sensitivity and Specificity)
""")
