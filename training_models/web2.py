import os
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models only once
diabetes_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\diabetes_model.sav", "rb"))
heart_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\heart_model.sav", "rb"))
parkinsons_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\parkinsons_model.sav", "rb"))

st.set_page_config(page_title='Prediction of Disease Outbreak', layout='wide', page_icon='🩺')

# Sidebar Menu
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# ------------------- Diabetes Prediction -------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2:
        Age = st.text_input("Enter Age")

    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                                   float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
                                  ).reshape(1, -1)
            diab_prediction = diabetes_model.predict(user_input)
            
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = 'Please enter valid numeric values'
    
    st.success(diab_diagnosis)

# ------------------- Heart Disease Prediction -------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Enter your age')
    with col2:
        sex = st.text_input('Sex (0 = Female, 1 = Male)')
    with col3:
        cp = st.text_input('Enter Chest Pain Type (cp)')
    with col1:
        trestbps = st.text_input('Enter Resting Blood Pressure')
    with col2:
        chol = st.text_input('Enter Cholesterol Level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar (1 = True, 0 = False)')
    with col1:
        restecg = st.text_input('Resting ECG Results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy')
    with col1:
        thal = st.text_input('Thalassemia (0-3)')

    heart_diagnosis = ''

    if st.button('Heart Test Result'):
        try:
            user_input = np.array([float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), 
                                   float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), 
                                   float(ca), float(thal)]).reshape(1, -1)
            heart_prediction = heart_model.predict(user_input)
            
            heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        except ValueError:
            heart_diagnosis = 'Please enter valid numeric values'

    st.success(heart_diagnosis)

# ------------------- Parkinson’s Disease Prediction -------------------
if selected == 'Parkinsons Prediction':
    st.title('Parkinson’s Disease Prediction using ML')

    #Due to the invlid inputs variable names these  Input Fields I have chosen
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR']
#To store 
    user_values = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            value = st.text_input(f'Enter {feature}')
            user_values.append(value if value else '0')  # Default to '0' if empty

    parkinsons_diagnosis = ''

    if st.button('Parkinson’s Test Result'):
        try:
            user_input = np.array([float(v) for v in user_values] + [0.0] * 6).reshape(1, -1)  # Ensure 22 features
            parkinsons_prediction = parkinsons_model.predict(user_input)

            parkinsons_diagnosis = 'Person has Parkinson’s disease' if parkinsons_prediction[0] == 1 else 'Person does not have Parkinson’s disease'
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numeric values'

    st.success(parkinsons_diagnosis)
