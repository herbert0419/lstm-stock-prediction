import streamlit as st
from fbprophet import Prophet
import yfinance as yf

# Set app title and sidebar options
st.set_page_config(page_title='Stock Prediction App')
st.sidebar.title('Select a stock')

# Define a function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    stock_data = yf.Ticker(ticker).history(period='max')
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Close']]
    stock_data = stock_data.rename(columns={'Date': 'ds', 'Close': 'y'})
    return stock_data

# Define a function to make stock predictions using Prophet
def make_predictions(stock_data, num_years):
    model = Prophet()
    model.fit(stock_data)
    future = model.make_future_dataframe(periods=num_years*365, freq='D')
    forecast = model.predict(future)
    return forecast

# Define app layout and functionality
st.title('Stock Prediction App')
ticker = st.sidebar.text_input('Enter a stock symbol (e.g. AAPL)')
num_years = st.sidebar.slider('Select the number of years to predict', 1, 5, 3)
if ticker:
    stock_data = get_stock_data(ticker)
    st.subheader('Historical stock data')
    st.line_chart(stock_data)
    forecast = make_predictions(stock_data, num_years)
    st.subheader('Stock price predictions')
    st.line_chart(forecast[['ds', 'yhat']])
