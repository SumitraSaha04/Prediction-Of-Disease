import os 
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\diabetes_model.sav", "rb"))
heart_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\heart_model.sav", "rb"))
parkinsons_model = pickle.load(open(r"C:\Users\sahaa\OneDrive\Documents\Disease Outbreaks\training_models\parkinsons_model.sav", "rb"))

st.set_page_config(page_title='Prediction of Disease Outbreak',
                   layout='wide',
                   page_icon='🩺')

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak system',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')   
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')    
    with col2:
        Glucose = st.text_input('Glucose Level') 
    with col3:
        BloodPressure = st.text_input('Blood pressure value') 
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')  
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")   
    with col2:
        Age = st.text_input("Enter age")

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                               DiabetesPedigreeFunction, Age], dtype=float).reshape(1, -1)
        
        diab_prediction = diabetes_model.predict(user_input)
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
    
    st.success(diab_diagnosis)

elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Enter your age') 
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Enter cp')
    with col1:
        trestbps = st.text_input('Enter restbps')
    with col2:
        chol = st.text_input('Enter cholesterol level')
    with col3:
        fbs = st.text_input('Enter fbs')
    with col1:
        restecg = st.text_input('Enter restecg')
    with col2:
        thalach = st.text_input('Enter thalach')
    with col3:                          
        exang = st.text_input('Enter exang')                         
    with col1:                          
        oldpeak = st.text_input('Enter oldpeak')                                 
    with col2:                          
        slope = st.text_input('Enter slope') 
    with col3:                          
        ca = st.text_input('Enter ca')
    with col1:                          
        thal = st.text_input('Enter thal')   
    
    heart_diagnosis = ''
    if st.button('Heart Test Result'):
        user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], 
                              dtype=float).reshape(1, -1)
        
        heart_prediction = heart_model.predict(user_input)
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
    
    st.success(heart_diagnosis)

elif selected == 'Parkinsons Prediction':
    st.title('Parkinsons Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        mdvp_fo_hz = st.text_input('Enter frequency', value="0.0")
    with col2:
        mdvp_fhi = st.text_input('Enter fhi frequency', value="0.0")
    with col3:
        mdvp_flo = st.text_input('Enter flo frequency', value="0.0")
    with col1:
        mdvp_jitter_percent = st.text_input('Enter the jitter in percentage', value="0.0")
    with col2:
        mdvp_jitter_abs = st.text_input('Enter ABS', value="0.0")
    with col3:
        mdvp_rap = st.text_input('Enter rap', value="0.0")
    with col1:
        mdvp_ppq = st.text_input('Enter PPQ', value="0.0")
    with col2:
        jitter_ddp = st.text_input('Enter DDP', value="0.0")
    with col3:
        mdvp_shimmer = st.text_input('Enter shimmer value', value="0.0")
    with col1:
        mdvp_shimmerdb = st.text_input("Enter shimmer value in db", value="0.0")
    with col2:
        shimmer_apq3 = st.text_input("Enter shimmer APQ3", value="0.0")
    with col3:
        shimmer_apq5 = st.text_input("Enter shimmer APQ5", value="0.0")
    with col1:
        mdvp_apq = st.text_input("Enter shimmer APQ", value="0.0")
    with col2:
        shimmer_dda = st.text_input("Enter shimmer DDA", value="0.0")
    with col3:
        nhr = st.text_input("Enter NHR value", value="0.0")
    with col1:
        hnr = st.text_input("Enter HNR value", value="0.0")

    parkinsons_diagnosis = ''
    if st.button('Parkinsons Test Result'):
        user_input = np.array([
            float(mdvp_fo_hz), float(mdvp_fhi), float(mdvp_flo),
            float(mdvp_jitter_percent), float(mdvp_jitter_abs), float(mdvp_rap),
            float(mdvp_ppq), float(jitter_ddp), float(mdvp_shimmer),
            float(mdvp_shimmerdb), float(shimmer_apq3), float(shimmer_apq5),
            float(mdvp_apq), float(shimmer_dda), float(nhr), float(hnr) ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]).reshape(1, -1)

        parkinsons_prediction = parkinsons_model.predict(user_input)
        parkinsons_diagnosis = 'Person has Parkinson’s disease' if parkinsons_prediction[0] == 1 else 'Person does not have Parkinson’s disease'

    st.success(parkinsons_diagnosis)
