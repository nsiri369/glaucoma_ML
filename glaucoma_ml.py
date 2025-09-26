import streamlit as st
import pandas as pd
import pickle

st.title("Glaucoma Type Prediction App")

# Load the trained logistic regression model
try:
    with open('logreg_model.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

st.sidebar.header("Enter Patient Details")

# Collect numerical inputs (scaled in notebook)
age = st.sidebar.slider("Age", 20, 90, 50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", 5.0, 40.0, 15.0, step=0.1)
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", 0.1, 1.0, 0.5, step=0.01)
pachy = st.sidebar.slider("Pachymetry", 300.0, 700.0, 520.0, step=1.0)

# Categorical features (the ones you encoded in notebook)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
visual_acuity = st.sidebar.selectbox("Visual Acuity Measurements", ["Normal", "Reduced"])
family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
medical_history = st.sidebar.selectbox("Medical History", ["None", "Diabetes", "Hypertension"])
cataract_status = st.sidebar.selectbox("Cataract Status", ["Yes", "No"])
angle_closure_status = st.sidebar.selectbox("Angle Closure Status", ["Open", "Closed"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Suspect", "Confirmed"])

# Preprocess input to match your one-hot encoded dataframe
def preprocess_input():
    data = {
        'Age': age,
        'Intraocular Pressure (IOP)': iop,
        'Cup-to-Disc Ratio (CDR)': cdr,
        'Pachymetry': pachy
    }
    df = pd.DataFrame([data])

    # one-hot encode exactly like in notebook
    df['Gender_Female'] = 1 if gender == "Female" else 0
    df['Gender_Male'] = 1 if gender == "Male" else 0

    df['Visual Acuity Measurements_Normal'] = 1 if visual_acuity == "Normal" else 0
    df['Visual Acuity Measurements_Reduced'] = 1 if visual_acuity == "Reduced" else 0

    df['Family History_Yes'] = 1 if family_history == "Yes" else 0
    df['Family History_No'] = 1 if family_history == "No" else 0

    df['Medical History_None'] = 1 if medical_history == "None" else 0
    df['Medical History_Diabetes'] = 1 if medical_history == "Diabetes" else 0
    df['Medical History_Hypertension'] = 1 if medical_history == "Hypertension" else 0

    df['Cataract Status_Yes'] = 1 if cataract_status == "Yes" else 0
    df['Cataract Status_No'] = 1 if cataract_status == "No" else 0

    df['Angle Closure Status_Open'] = 1 if angle_closure_status == "Open" else 0
    df['Angle Closure Status_Closed'] = 1 if angle_closure_status == "Closed" else 0

    df['Diagnosis_Suspect'] = 1 if diagnosis == "Suspect" else 0
    df['Diagnosis_Confirmed'] = 1 if diagnosis == "Confirmed" else 0

    # Make sure columns order matches training data
    expected_columns = logreg.feature_names_in_  # sklearn 1.0+ stores this automatically
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

if st.sidebar.button("Predict"):
    input_df = preprocess_input()
    try:
        prediction = logreg.predict(input_df)
        predicted_label = prediction[0]  # direct string from your y
        st.subheader("Prediction Result")
        st.write(f"The predicted glaucoma type is: **{predicted_label}**")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.write("""
### Instructions
1. Use the sidebar to enter the patient's details.
2. Adjust sliders for numeric values and choose from dropdowns for categories.
3. Click 'Predict' to see the predicted glaucoma type.
""")
