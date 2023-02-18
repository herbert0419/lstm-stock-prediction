import streamlit as st
import h5py
import numpy as np

# Load the model from the .h5 file
with h5py.File('keras_model.h5', 'r') as model_file:
    weights = model_file['weights'][()]
    biases = model_file['biases'][()]

# Define a function to make predictions using the loaded model
def make_prediction(input_data):
    prediction = np.dot(input_data, weights) + biases
    return prediction

# Define the layout of the Streamlit app
st.title('My Prediction App')
st.write('Enter some data to make a prediction:')
input_data = st.text_input('Input data')

# Make a prediction using the loaded model when the user clicks the "Predict" button
if st.button('Predict'):
    input_data = np.array([float(input_data)])
    prediction = make_prediction(input_data)
    st.write(f'The predicted value is {prediction}')
