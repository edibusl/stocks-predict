import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split

import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split


def predict_close_lstm(df, batch_size=1, look_back=4, epochs=100, verbose=2):
    # If we don't have a fixed seed, every time that I train the model, it predicts different results.
    # Probably we should make enough epochs in order to get to a very small loss in order to get deterministic results
    np.random.seed(1)

    # Split to 2 dataframes: train and test
    df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)

    # Take only the training columns
    df_train = df_train.loc[:, ["Close"]]
    df_test = df_test.loc[:, ["Close"]]

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = scaler.fit_transform(df_train.values)
    test_features = scaler.fit_transform(df_test.values)

    # Reshape into X=t and Y=t+1
    train_x, train_y = create_lstm_dataset(train_features, look_back=look_back)
    test_x, test_y = create_lstm_dataset(test_features, look_back=look_back)

    # Fit training and testing samples to batch size
    train_x = fit_to_batch_size(train_x, batch_size)
    train_y = fit_to_batch_size(train_y, batch_size)
    test_x = fit_to_batch_size(test_x, batch_size)
    test_y = fit_to_batch_size(test_y, batch_size)

    # Train the model using the training set
    model = create_model(batch_size, look_back=look_back)
    model_res = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)

    # Predict the testing set
    test_y_predicted = model.predict(test_x, batch_size=batch_size)

    # Denormalize
    # We can't use scaler.inverse_transform() because it was previously used to fit the whole dataframe
    # which is 2-dimensional, while test_y and test_y_pred are 1-dimensional.
    # The solution is to manually calculate the original values by using the min/max variables
    # of the scaler (which were saved from last fit operation)
    # (I verified correctness of this method by inversing the test_x values and comparing to original)
    test_y = (test_y * scaler.data_range_[0]) + scaler.data_min_[0]
    # Reshape the predicted array to be flat
    test_y_predicted = test_y_predicted.flatten()
    test_y_predicted = (test_y_predicted * scaler.data_range_[0]) + scaler.data_min_[0]

    # Calc error on the testing set
    test_rmse = math.sqrt(mean_squared_error(test_y, test_y_predicted))

    return {
        "model_loss": model_res.history['loss'],
        "test_y": test_y,
        "test_y_predicted": test_y_predicted,
        "test_rmse": test_rmse
    }


def fit_to_batch_size(dataset, batch_size):
    # Remove from the dataset the last samples that don't fit the batch size
    unfit_samples_num = dataset.shape[0] % batch_size
    if unfit_samples_num > 0:
        dataset = dataset[:-unfit_samples_num]

    return dataset


def create_model(batch_size, look_back, regularization_factor=0, dropout=False, custom_optimizer=False):
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    if dropout:
        model.add(Dropout(0.5))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(regularization_factor)))

    if custom_optimizer:
        learning_rate = 0.005
        optimzer = optimizers.RMSprop(lr=learning_rate)
    else:
        optimzer = 'adam'

    model.compile(loss='mean_squared_error', optimizer=optimzer)

    return model


def create_lstm_dataset(dataset, look_back=1):
    data_x, data_y = [], []

    # Go through all samples, shift by 1 on every iteration
    # Look upfront to next look_back samples
    for i in range(len(dataset) - look_back - 1):
        # Get next look_back samples
        look_back_window = dataset[i:(i + look_back), 0]
        data_x.append(look_back_window)

        # Get the look_back+1 sample for the "y"
        data_y.append(dataset[i + look_back, 0])

    # Reshape input to be [samples, time steps, features]
    data_x = np.array(data_x)
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))

    return data_x, np.array(data_y)


def plot_prediction_results(data):
    # Plot the loss in every epoch during training
    plt.figure(figsize=(15, 10))
    plt.plot(data["model_loss"])
    plt.title("Loss during training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train"])
    plt.show()

    # Plot the actual values vs predicted values in the testing set
    plt.figure(figsize=(15, 10))
    plt.plot(data["test_y"])
    plt.plot(data["test_y_predicted"])
    plt.title("Testing set - Predicted vs Actual")
    plt.xlabel("Day")
    plt.ylabel("Close Price")
    plt.legend(["Real", "Prediction"])
    plt.show()


def run_single_configuration(batch_size=5, look_back=3, epochs=60, plot_results=True):
    # Read data
    df = pd.read_csv('~/code/dscience/stocks/data/facebook.csv', engine='python', skipfooter=3)
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Run prediction
    res = predict_close_lstm(df, batch_size=batch_size, look_back=look_back, epochs=epochs)
    print("Test RMSE: %.2f" % (res["test_rmse"]))

    # Plot results
    if plot_results:
        plot_prediction_results(res)

    return res


def run_multiple_configurations(batch_sizes = [20, 10, 5, 4, 3, 2, 1], look_backs = [20, 10, 7, 5, 4, 3, 2, 1]):
    # Read data
    df = pd.read_csv('~/code/dscience/stocks/data/facebook.csv', engine='python', skipfooter=3)
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


    # Create a DF to hold the results of RMSE for all combinations
    df_rmse = pd.DataFrame(columns=["{} look_back".format(i) for i in look_backs])

    # Calc RMSE for all windows sizes X number of poly features
    for batch_size in batch_sizes:
        rmse_look_back = []
        for look_back in look_backs:
            logger.info("Training batch_size {}, look_back {}".format(batch_size, look_back))

            # Run prediction
            res = predict_close_lstm(df, batch_size=batch_size, look_back=look_back, epochs=140, verbose=0)
            rmse_look_back.append(res["test_rmse"])
            logger.info("RMSE for batch size %s, look back %s: %s", batch_size, look_back, res["test_rmse"])
        df_rmse.loc["batch_size {}".format(batch_size)] = rmse_look_back

    # Plot on map
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_rmse, cmap="YlGnBu", annot=True, fmt='.2f')
    plt.show()

    return df_rmse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# run_multiple_configurations(batch_sizes=[20], look_backs=[3])


df = pd.read_csv('~/code/dscience/stocks/data/facebook.csv', engine='python', skipfooter=3)
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
res = predict_close_lstm(df, batch_size=4, look_back=3, epochs=60, verbose=0)
logger.info("RMSE for batch size %s, look back %s: %s", 20, 3, res["test_rmse"])