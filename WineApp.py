# -*- coding: utf-8 -*-


import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
loaded_model = pickle.load(open('svm_trained_model.sav','rb'))
scaler = pickle.load(open('stdscaler.sav','rb'))
#creating function
def wine_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The wine is Legit'
    elif (prediction[0] == 1):
      return 'The wine is Fraud'
    

    
    
def main():
    
    st.title('Wine Fraud Prediction')
    st.markdown("___")
    side_bar = st.sidebar
    side_bar.image('./wine_fraud.jpg')
    side_bar.image('./fraud.jpg')
    side_bar.image('./wine.jpg')
    side_bar.header('')
    side_bar.header('BE CAREFUL!')

    fixed_acidity=st.slider('Fixed acidity:',min_value=3.0, step=0.1, max_value=18.0, format="%1f")
    volatile_acidity=st.slider('Volatile acidity:',min_value=0.05, step=0.01, max_value=2.00, format="%2f")
    citric_acid=st.slider('Citric acid quantity::',min_value=0.00, step=0.01, max_value=2.00, format="%2f")
    resid_sugar=st.slider('Residual sugar quantity:',min_value=0.0, step=0.1, max_value=70.0, format="%1f")
    chlorides=st.slider('Chlorides',min_value=0.005, step=0.001, max_value=0.7, format="%3f")
    free_sulfur_diox=st.slider('Free Sulfur Dioxide quantity:',min_value=1,max_value=300,step=1)
    total_sulfur_diox=st.slider('Total Sulfur Dioxide quantity:',min_value=1,max_value=450,step=1)
    density=st.slider('Density value:',min_value=0.90000, step=0.00001, max_value=1.1, format="%5f")
    pH=st.slider('PH value:',min_value=2.70, step=0.01, max_value=6.0, format="%2f")
    sulphates=st.slider('Sulphates quantity',min_value=0.02, step=0.01, max_value=2.0, format="%2f")
    alcohol=st.slider('Alcohol degree',min_value=7.0, step=0.1, max_value=16.0, format="%1f")
    tipo=st.radio("Type of wine:", ('Red','White'))
    if (tipo=='White'):
        type_white=1
    elif (tipo=='Red'):
        type_white=0


    result = ''
    if st.button('Wine Test Result'):
        result = wine_prediction([fixed_acidity, volatile_acidity, citric_acid, resid_sugar, chlorides, free_sulfur_diox, total_sulfur_diox, density, pH, sulphates, alcohol, type_white])
        
    st.success(result)
    if (result=='The wine is Legit'):
        st.balloons()   
    


if __name__ == '__main__':
     main()    
    
    
