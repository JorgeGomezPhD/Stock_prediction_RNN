# Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to
# predict the closing stock price of the stock you would like by using the past 90 day stock price.

# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import csv
from statistics import mean
from datetime import datetime
import matplotlib.pyplot as plt
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


stock = input('What stock do you want to predict? ')
start_date = '2012-01-01'
end_date = '2020-09-01'
date_today = datetime.today().strftime('%Y-%m-%d')
pred_days = 90
# Creates a folder in the current directory to store todays files
folder_name = f'./{date_today}_{stock}_Stock_Prediction/'
createFolder(folder_name)

# Get the stock quote
df = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
# Converting the dataframe to a numpy array
dataset = data.values

#  Writing CSV to use later on
with open(f'{folder_name}{date_today}_{stock}_Stock_Prediction.csv', 'a') as csvfile:
    fieldnames = ['Predicted Price', 'RMSE', 'Avg Predicted price', 'Actual Price']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .85)  # round up
# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(pred_days, len(train_data)):
    x_train.append(train_data[i - pred_days:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# running 10 times to take an average
for index in range(1,11):
    # Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model fit==train
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Test data set
    test_data = scaled_data[training_data_len - pred_days:, :]
    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:,
             :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(pred_days, len(test_data)):
        x_test.append(test_data[i - pred_days:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Getting the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Calculate/Get the value of RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    # print(f'RMSE: {rmse}')

    # Create prediction data column
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Show the valid and predicted prices
    # print(valid)

    # Create a new dataframe
    new_df = df.filter(['Close'])
    # Get the last pred_days day closing price in array form
    last_pred_days_days = new_df[-pred_days:].values
    # Scale the data to be values between 0 and 1
    last_pred_days_days_scaled = scaler.transform(last_pred_days_days)
    # Create an empty list
    X_test = []
    # Append teh past pred_days days
    X_test.append(last_pred_days_days_scaled)
    # Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    pred_price = pred_price[-1][-1]

    print(pred_price)
    print(rmse)

    # #Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.savefig(f'{folder_name}{index}_{date_today}_{stock}_Stock_Prediction.jpg')

    #  Writing to Predicted price and RMSE to CSV file
    with open(f'{folder_name}{date_today}_{stock}_Stock_Prediction.csv', 'a') as csvfile:
        fieldnames = ['Predicted Price', 'RMSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Predicted Price': pred_price, 'RMSE': rmse})

#  Bringing back  the predicted and RMSE data back to take the average
data_file = pd.read_csv(f'{folder_name}{date_today}_{stock}_Stock_Prediction.csv')
predicted_price_list = data_file['Predicted Price']
rmse = data_file['RMSE']
average_stock = mean(predicted_price_list)
average_rmse = mean(rmse)

# # Get the quote for Sep. 2, 2020 stock
stock_quote2 = web.DataReader(stock, data_source='yahoo', start='2020-09-02', end='2020-09-02')
print(f"Average predicted price for {stock} stock = {round(average_stock, 2)}")
# print(f"Average RMSE for {stock} stock = {round(average_rmse, 2)}")
print(f"The actual value for {stock} stock is: {stock_quote2['Close'][-1]}")

#  Saving the average stock price and the actual stock price to CSV
with open(f'{folder_name}{date_today}_{stock}_Stock_Prediction.csv', 'a') as csvfile:
    fieldnames = ['Predicted Price', 'RMSE', 'Avg Predicted price', 'Actual Price']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'Predicted Price': ' ', 'RMSE': ' ', 'Avg Predicted price': average_stock,
                     'Actual Price': stock_quote2['Close'][-1]})
