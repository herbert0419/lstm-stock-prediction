import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

# Get the data using the yfinance API
data = yf.download('ADANIENT.NS', period='max')
data = data.sort_index(ascending=True, axis=0)

# Set up the Streamlit app
st.title('EDA for ADANIENT.NS')
st.subheader('Historical Data from yfinance API')

# Display the raw data
st.subheader('Raw Data')
st.write(data.tail())

# Create a line chart of the stock prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
fig.layout.update(title='Historical Close Price', xaxis_title='Date', yaxis_title='Price (INR)')
st.plotly_chart(fig)

# Calculate and display the percentage change in the stock prices
st.subheader('Percentage Change')
change = data['Close'].pct_change()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=change.index, y=change, name='Percentage Change'))
fig2.layout.update(title='Daily Percentage Change', xaxis_title='Date', yaxis_title='Percentage Change')
st.plotly_chart(fig2)

# Display a histogram of the daily percentage change
st.subheader('Histogram of Percentage Change')
fig3 = px.histogram(change, x='Close')
st.plotly_chart(fig3)

# Calculate and display the rolling mean and standard deviation of the stock prices
st.subheader('Rolling Statistics')
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
fig4.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name='Rolling Mean'))
fig4.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, name='Rolling Std'))
fig4.layout.update(title='Rolling Mean and Standard Deviation', xaxis_title='Date', yaxis_title='Price (INR)')
st.plotly_chart(fig4)
