# In conda cammand prompt write "streamlit run work.py" to run this code on local host

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from datetime import date
from sklearn.preprocessing import MinMaxScaler       #Transform features by scaling each feature to a given range. It is used here in KNN and LSTM.
from fastai.tabular.core import add_datepart         #Helper function that adds columns relevant to a date in the column. Here it is used in
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error #GridSearch is used here in KNN, SVM and Randomforest to find the best parameter.
from keras.models import Sequential
from keras.layers import Dense, LSTM


st.title('Stock Price Prediction Using Machine Learning')

st.header("Prerequisite:\n")
st.subheader("1) The user must Provide a Stock code or Stock Ticker from the site: https://finance.yahoo.com/")


start = "2010-01-01"
end = date.today().strftime("%Y-%m-%d")


stocks = st.text_input('Enter Stock Code from the site: https://finance.yahoo.com/')
if stocks:
    data_load_state = st.text('Loading data...')
    df = data.DataReader(stocks, 'yahoo', start, end)
    df = df.dropna()
    df.reset_index(inplace=True)
    X_future = df[['Date']]         
    X_future = X_future[0:10]  
    data_load_state.text('Loading data... done!')

    #Data Introduction
    st.subheader('discription of data from Date from 2010 to today.')
    st.write(df.describe())
    st.subheader('previouse five days dataset.')
    st.write(df.tail())

    #Data Visualization
    st.subheader('Chart for Closing Price Vs Time')
    df.index = pd.to_datetime(df.Date)
    fig = plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price history')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Stock Price', fontsize=20)
    plt.title('Movement of stock price', fontsize=24)
    st.pyplot(fig)

    #datapreprocessing
    sorted_df = df.sort_index(ascending=True, axis=0)


    data = pd.DataFrame(columns=['Date', 'Close'])
    data['Date'] = sorted_df['Date'].values
    data['Close'] = sorted_df['Close'].values

    add_datepart(data, 'Date')
    data.index = sorted_df.index
    data.drop('Elapsed', axis=1, inplace=True)

    ratio = 0.2
    label = data['Close']
    features = data.drop('Close', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=ratio, shuffle=False)

    X_future_common = X_future.copy(deep=True)

    index = pd.to_datetime(X_future_common['Date'])
    add_datepart(X_future_common, 'Date')
    X_future_common.index = index
    X_future_common.drop('Elapsed', axis=1, inplace=True)

    st.subheader('LSTM Implemention')
    data_load_lstm = st.text('Loading LSTM Algorithm on Your Stock for future prediction....')
    data = pd.DataFrame(columns=['Date', 'Close'])

    data['Date'] = sorted_df['Date'].values
    data['Close'] = sorted_df['Close'].values
    data.index = pd.to_datetime(data['Date'])
    data.drop('Date', axis=1, inplace=True)

    dataset = data.values

    train_df, test_df = train_test_split(data, test_size=ratio, shuffle=False)
    train_size = train_df.shape[0]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X_train, y_train = [], []
    sequence_len = 60
    for i in range(sequence_len, train_size):
        X_train.append(scaled_data[i-sequence_len:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # create and fit the LSTM network

    neurons = [50]
    minimum = 100000
    preds = []
    best_model = Sequential()
    for i, n in enumerate(neurons):
        model = Sequential()
        model.add(LSTM(units=n, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(units=n))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, batch_size=1, validation_split=0.1)

        inputs = data[len(data) - len(test_df) - sequence_len:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        # predicting closing price for test set, using past 60 values from the train data
        X_test_i = []
        for i in range(len(test_df)):
            X_test_i.append(inputs[i:i+sequence_len, 0])
        X_test_i = np.array(X_test_i)
        X_test_i = np.reshape(X_test_i, (X_test_i.shape[0], X_test_i.shape[1], 1))

        preds = model.predict(X_test_i)
        preds = scaler.inverse_transform(preds)

        rms = rms = mean_squared_error(test_df.values, preds, squared=False)
        if rms < minimum:
            minimum = rms
            best_preds = preds 
            best_model = model


    rms_lstm_test = minimum
    preds = best_preds
    model = best_model

    inputs_t = data.values
    inputs_t = inputs_t.reshape(-1,1)
    inputs_t  = scaler.transform(inputs_t)

    # predicting closing price for train set
    Xt = []
    for i in range(len(train_df)-sequence_len):
        Xt.append(inputs_t[i:i+sequence_len, 0])
    Xt = np.array(Xt)
    Xt = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 1))

    preds_train = model.predict(Xt)
    preds_train = scaler.inverse_transform(preds_train)

    rms_lstm_train = np.sqrt(np.mean(np.power((train_df.values[sequence_len:] - preds_train),2)))

    r2_lstm_test = r2_score(test_df.values, preds)

    r2_lstm_train = r2_score(train_df.values[sequence_len:], preds_train)

    # MAE
    mae_lstm_test = mean_absolute_error(test_df.values, preds)

    mae_lstm_train = mean_absolute_error(train_df.values[sequence_len:], preds_train)

    # RMSE
    rms_lstm_test = mean_squared_error(test_df.values, preds, squared=False)

    rms_lstm_train = mean_squared_error(train_df.values[sequence_len:], preds_train, squared=False)

    for i in range(sequence_len):
        preds_train = np.concatenate([[dataset[i]], preds_train])

    # Predict price for n next days

    n = 10
    X_future_test = []
    for i in range(0, n):
        X_future_test.append(inputs[-sequence_len:, 0])
        X_future_test_arr = np.array(X_future_test)
        X_future_test_arr = np.reshape(X_future_test_arr, (X_future_test_arr.shape[0], X_future_test_arr.shape[1], 1))
        
        nextday_closing_price = model.predict(X_future_test_arr)
        inputs = np.append(inputs, nextday_closing_price, 0)

    nextday_closing_price = scaler.inverse_transform(nextday_closing_price)

    future_df = X_future.copy(deep=True)
    index = pd.to_datetime(future_df['Date'])
    future_df.index = index
    future_df.drop('Date', axis=1, inplace=True)

    test_df['Predictions'] = preds
    train_df['Predictions'] = preds_train
    #future_df['Predictions'] = nextday_closing_price

    data_load_lstm.text('Here is your Stock Future....')
    fig4 = plt.figure(figsize=(12,6))
    plt.plot(train_df['Close'], label='Actual Price for training set')
    plt.plot(test_df['Close'], label='Actual Price for testing set')
    plt.plot(train_df['Predictions'], label='Predicted Price for training set')
    plt.plot(test_df['Predictions'], label='Predicted Price for testing set')
    #plt.plot(future_df['Predictions'], label='Predicted Price for future set')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Stock Price', fontsize=15)
    plt.title('Fitting curve between predicted and real prices using LSTM')
    plt.legend()
    st.pyplot(fig4)

    errorrep = pd.DataFrame({"Errors":["r2_lstm_train", "r2_lstm_test", "mae_lstm_train", "mae_lstm_test", "rms_lstm_train", "rms_lstm_test"],
                        "LSTM Scores": [r2_lstm_train, r2_lstm_test, mae_lstm_train, mae_lstm_test, rms_lstm_train, rms_lstm_test],
                        })
    st.write(errorrep)

    predicto = pd.DataFrame({"Dates": ['Day1','Day2','Day3','Day4','Day5','Day6','Day7','Day8','Day9','Day10'],
                        "LSTM Predicted Closing Value": nextday_closing_price.flatten()
                        })
    st.write(predicto)