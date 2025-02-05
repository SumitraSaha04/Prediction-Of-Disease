import os 
import numpy as np
import pickle #pre trained models loading 
import streamlit as st # for craeting web app
from streamlit_option_menu import option_menu# third party library to create stylish side bar menu read streamlit documentation
diabetes_model = pickle.load(open("C:/Users/sahaa/OneDrive/Documents/Disease Outbreaks/training_models/diabetes_model.sav", "rb"))
heart_model=pickle.load(open("C:/Users/sahaa/OneDrive/Documents/Disease Outbreaks/training_models/heart_model.sav","rb"))
parkinsons_model=pickle.load(open("C:/Users/sahaa/OneDrive/Documents/Disease Outbreaks/training_models/parkinsons_model.sav","rb"))

st.set_page_config(page_title='Prediction of Disease Outbreak',layout='wide',page_icon='🩺')

#Sidebar Menu
with st.sidebar:
    selected=option_menu('Prediction of Disease Outbreak system',
                         ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                         menu_icon='hospital-fill',icons=['activity','heart','person'],default_index=0)
# -----------Diabetes Prediction-------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')   
    col1,col2,col3=st.columns(3)

    with col1:
        Pregnancies=st.text_input('Number of Pregnancies')    
    with col2:
        Glucose=st.text_input('Glucose Level') 
    with col3:
        BloodPressure=st.text_input('Blood pressure value') 
    with col1:
        SkinThickness=st.text_input('Skin Thickness Value')  
    with col2:
        Insulin=st.text_input('Insulin Level')
    with col3:
        BMI=st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedegree Function")   
    with col2:
        Age=st.text_input("Enter age")

diab_diagnosis=''

if st.button('Diabetes Test Result'):
    try:
        user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                           DiabetesPedigreeFunction, Age], dtype=float).reshape(1, -1)
    
        diab_prediction=diabetes_model.predict(user_input)# stores the prediction of the model
        if diab_prediction[0]==1:# the models check prediction is =1 or not
            diab_diagnosis='The person is diabetic'   
        else:
            diab_diagnosis='The person is not diabetic'
    except ValueError:
        diab_diagnosis='Please enter valid inputs '
st.success(diab_diagnosis)   



#----------Heart Disease Prediction-------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1,col2,col3=st.columns(3)
    with col1:
        age=st.text_input('Enter your age') 
    with col2:
        sex=st.text_input('Enter 0 for male,1 for female')
    with col3:
        cp=st.text_input('Enter chest pain type')
    with col1:
        trestbps=st.text_input('Enter Resting Blood Pressure')
    with col2:
        chol=st.text_input('Enter Cholestrol Level')
    with col3:
        fbs=st.text_input('Fasting Blood Sugar(1=True, 0=False)')
    with col1:
        restecg=st.text_input('Resting ECG Results')
    with col2:
        thalach=st.text_input('Maximum Heart Rate Achieved')
    with col3:                          
        exang=st.text_input('Exercise Induced Angina(1=yes,0=No)')                         
    with col1:                          
        oldpeak=st.text_input('ST Depression Induced')                                 
    with col2:                          
        slope=st.text_input('Slope of the Peak Excercise ST Segment') 
    with col3:                          
        ca=st.text_input('Number of Major Vessels Colored by Fluroscopy')
    with col1:                          
        thal=st.text_input('Enter thalassemia(0-3)')   
heart_diagnosis=''
if st.button('Heart Test Result'):
    try:
            user_input = np.array([float(age), sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,  
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Add 10 more features with default values
                    ], dtype=float).reshape(1, -1)

    
            heart_prediction=heart_model.predict(user_input)# stores the prediction of the model
            if heart_prediction[0]==1:# the models check prediction is =1 or not
                heart_diagnosis='The person has heart disease'   
            else:
                heart_diagnosis='The person does not have heart disease'
    except ValueError: 
         heart_diagnosis='Please enter valid inputs '      
st.success(heart_diagnosis) 



#-------------Parkinsons Prediction-------------
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Disease Prediction using ML')
#Due to the invalid inputs variable names I have kept all in this features
features=['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR']
#To store user inputs
user_values=[]
 # Created column for better layout
col1, col2, col3=st.columns(3) 
 #Loop through each feature using enumerate  and created an input field
for i, feature in enumerate(features):
    #To divide the layout into three columns syntax explaination is value then write condition again, value then condition
    with (col1 if i%3 == 0 else col2 if i%3==1 else col3):
           #This f-string replaces the feature with its actual value
        value =st.text_input(f'Enter {feature}') 
        # value if value else '0'is the ternary type operation handling multiple conditions
            #append each user value into the list created user_values
        user_values.append(value if value else '0') # '0' if empty

parkinsons_diagnosis = ''
if st.button('Parkinsons Test Result'):
        try:
            #Convert user inputs to Numpy  and set to float datatype 
            #Padded with '0.0' to equal to 22 features becoz we had 22 features but to adjust it added 6 times (0.0)
            user_input=np.array([float(v) for v in user_values]+[0.0]*6).reshape(1,-1)# This ensure 22 features
            #Predict  by using user_inputs using the pre-trained models 
            parkinsons_prediction = parkinsons_model.predict(user_input)
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'Person has Parkinson’s disease'
            else:
               parkinsons_diagnosis = 'Person does not have Parkinson’s disease'

        except ValueError:
         parkinsons_diagnosis='Please enter valid values'        
st.success(parkinsons_diagnosis)
