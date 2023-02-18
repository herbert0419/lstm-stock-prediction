import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained LSTM model and scaler
model = load_model('keras_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Load the data
data = yf.download("ADANIENT.NS", start="2020-01-01", end="2023-02-19")
data = data.sort_index(ascending=True, axis=0)

# Create a new dataframe with only the 'Close' column
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Close'])
for i in range(0,len(data)):
    new_data['Close'][i] = data['Close'][i]

# Normalize the data using the scaler
scaled_data = scaler.fit_transform(np.array(new_data).reshape(-1,1))

# Define the function to create the dataset
def create_dataset(dataset, time_step=1):
    X_data, y_data = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X_data.append(a)
        y_data.append(dataset[i + time_step, 0])
    return np.array(X_data), np.array(y_data)

# Set the time step for the dataset
time_step = 100

# Create the X dataset
x_input = scaled_data[len(scaled_data)-time_step:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# Define the number of days to predict
n_days = 7

# Make the predictions
lst_output=[]
n_steps=time_step
i=0
while(i<n_days):
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        lst_output.append(yhat[0][0])
        i+=1

# Inverse transform the predictions to get the actual values
predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Set up the Streamlit app
st.title('ADANIENT.NS Stock Price Prediction using LSTM')
st.subheader('Predicted Stock Price for the next {} days'.format(n_days))
st.line_chart(predictions)

st.subheader('Actual vs. Predicted Stock Price')
st.line_chart(data['Close'][-n_days:])
st.line_chart(predictions)

st.subheader('Data for the last {} days'.format(n_days))
st.write(data.tail(n_days))
