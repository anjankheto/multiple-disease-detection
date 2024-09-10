# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:59:28 2024

@author: anjan
"""
import numpy as np
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu



with st.sidebar:
    
    selected=option_menu('Multiple Disease Detection System',
                         ['Diabetes Detection',
                          'Heart Disease Detection',
                          'breast-Cancer Detection'],
                         icons=['activity','heart','person'],
                         default_index=0)
   
if(selected=='Diabetes Detection'):
    loaded_model=pickle.load(open('diabetic_detection.sav','rb'))

    def diabetes_detection(inputdata):
        input_data=np.asarray((inputdata))
        input_data_reshape=input_data.reshape(1,-1)
        prediction=loaded_model.predict(input_data_reshape)
        if(prediction[0]==0):
          return 'You are Non-diabetic'
        else:
          return 'You are diabetic'
      
    def main():
        #giving the title
        st.title('Diabetes Disease Detection')
         
        #loading the data
        col1,col2=st.columns(2)
        with col1:
            Pregnancies=st.text_input('Pregnancies')
        with col2:
            Glucose=st.text_input('Glucose')
        
        col1,col2=st.columns(2)
        with col1:
            BloodPressure=st.text_input('BloodPressure')
        with col2:
            SkinThickness=st.text_input('SkinThickness')
       
        col1,col2=st.columns(2) 
        with col1:
            Insulin=st.text_input('Insulin')
        with col2:
            BMI=st.text_input('BMI')
       
        col1,col2=st.columns(2)
        with col1:
            DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction')
        with col2:
            Age=st.text_input('Age')
        diagnosis=""
        
        
        #creating a button
        if(st.button('test my result')):
            diagnosis=diabetes_detection([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
            
            
        st.success(diagnosis)
        
        
    if __name__=='__main__':
        main()
if(selected=='Heart Disease Detection'):
    loaded_model=pickle.load(open('heart_disease_model.sav','rb'))

    def heart_disease(inputdata):
        input_data=np.asarray((inputdata))
        input_data_reshape=input_data.reshape(1,-1)
        prediction=loaded_model.predict(input_data_reshape)
        if(prediction[0]==0):
          return 'No Heart Disease detected'
        else:
          return 'Heart Disease detected'
      
    def main():
        #giving the title
        st.title('Heart Disease Detection')
         
        #loading the data
        col1,col2=st.columns(2)
        with col1:
            age=st.text_input('age')	
        with col2:
            thal=st.text_input('thalassemia')
            
        col1,col2=st.columns(2)
        with col1:
            sex=st.text_input('gender')
        with col2:
            cp=st.text_input('chest pain')
         
        col1,col2=st.columns(2)
        with col1:
            trestbps=st.text_input('resting blood pressure systolic')
        with col2:
            chol=st.text_input('cholestrol')
        
        col1,col2=st.columns(2)
        with col1:
            fbs=st.text_input('fasting blood sugar')
        with col2:
            restecg=st.text_input('resting electrocardiogram')
        
        col1,col2=st.columns(2)
        with col1:
            thalach=st.text_input('thalach')	
        with col2:     
            exang=st.text_input('angina')
        col1,col2=st.columns(2)
        with col1:
            oldpeak=st.text_input('oldpeak')
        with col2:
            slope=st.text_input('slope')
        col1,col2,col3=st.columns(3)

        with col2:	
            ca=st.text_input('chronic anemia')
        
       
        
        diagnosis=""
        
        
        #creating a button
        if(st.button('test my result')):
            diagnosis=heart_disease([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
            
            
        st.success(diagnosis)
        
        
    if __name__=='__main__':
        main()
if(selected=='breast-Cancer Detection'):
    from sklearn.preprocessing import StandardScaler
    loaded_model=pickle.load(open('breast_cancer_detection.sav','rb'))

    def cancer_detection(inputdata):
        input_data=np.asarray((inputdata))
        input_data_reshaped=input_data.reshape(1,-1)
        scaler=StandardScaler()
        input_data_reshaped=scaler.transform(input_data_reshaped)
        prediction=loaded_model.predict(input_data_reshaped)
        
        prediction_labels=[np.argmax(prediction)]

        if(prediction_labels[0]==0):
            return 'The breast cancer is Malignant'
        else:
            return 'The breast cancer is Benign'
        
       
      
    def main():
        #giving the title
        st.title('Breast Cancer Detection')
         
        #loading the data
        col1,col2,col3,=st.columns(3)
        with col1:
            mean_radius=st.text_input('mean radius')
        with col2:
            mean_texture=st.text_input('mean texture')
        with col3:
            mean_perimeter=st.text_input('mean perimeter')
        col1,col2,col3,=st.columns(3)
        with col1:
            mean_area=st.text_input(' mean area')
        with col2:
            mean_smoothness=st.text_input('mean smoothness')
        with col3:
            mean_compactness=st.text_input('mean compactness')
        col1,col2,col3,=st.columns(3)
        with col1:
            mean_concavity=st.text_input(' mean concavity')
        with col2:
            mean_concave_points=st.text_input('mean concave point')
        with col3:
            mean_symmetry=st.text_input('mean symmetry')
        col1,col2,col3,=st.columns(3)
        with col1:
            mean_fractal_dimension=st.text_input('mean fractal dimension')
        with col2:
            radius_error=st.text_input('radius error')
        with col3:
            texture_error=st.text_input('texture error')
        col1,col2,col3,=st.columns(3)
        with col1:
            perimeter_error=st.text_input('perimeter error')
        with col2:
            area_error=st.text_input('area error')
        with col3:
            smoothness_error=st.text_input('smoothness error')
        col1,col2,col3,=st.columns(3)
        with col1:
            compactness_error=st.text_input('compactness error')
        with col2:
            concavity_error=st.text_input('concavity error')
        with col3:
            concave_points_error=st.text_input('concave points error')
        col1,col2,col3,=st.columns(3)
        with col1:
            symmetry_error=st.text_input('symmetry error')
        with col2:
            fractal_dimension_error=st.text_input('fractal dimension error')
        with col3:
            worst_radius=st.text_input('worst_radius')
        col1,col2,col3,=st.columns(3)
        with col1:
            worst_texture=st.text_input('worst texture')
        with col2:
            worst_perimeter=st.text_input('worst perimeter')
        with col3:
            worst_area=st.text_input('worst area')
        col1,col2,col3,=st.columns(3)
        with col1:
            worst_smoothness=st.text_input('worst smoothness')
        with col2:
            worst_compactness=st.text_input('worst compactness')
        with col3:
            worst_concavity=st.text_input('worst concavity')
        col1,col2,col3,=st.columns(3)
        with col1:
            worst_concave_points=st.text_input('worst concave points')
        with col2:
            worst_symmetry=st.text_input('worst symmetry')
        with col3:
            worst_fractal_dimension=st.text_input('worst fractal dimension')
        
        diagnosis=""
        
        
        #creating a button
        if(st.button('test my result')):
            diagnosis=cancer_detection([mean_radius,mean_texture,mean_perimeter,mean_area,
            mean_smoothness,mean_compactness,mean_concavity,
            mean_concave_points,mean_symmetry,mean_fractal_dimension,
            radius_error,texture_error,perimeter_error,area_error,
            smoothness_error,compactness_error,concavity_error,
            concave_points_error,symmetry_error,
            fractal_dimension_error,worst_radius,worst_texture,
            worst_perimeter,worst_area,worst_smoothness,
            worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension])
            
            
        st.success(diagnosis)
        
        
    if __name__=='__main__':
        main()