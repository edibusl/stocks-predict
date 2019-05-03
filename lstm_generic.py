import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers, optimizers


def predict_close_lstm(df, cur_day, num_of_days, window_size=10, look_back=4, epochs=100, plot=False, plot_callback=None, loss_threshold=1e-4):
    actuals = []
    predictions = []

    start = cur_day
    end = cur_day + num_of_days
    # Go through all requested days and make a new training for every day
    for i in range(start, end):
        print("Training day {}".format(i - start))
        # For each day, train a new model using Close values of previous days.
        # Number of previous days is defined by "window_size"
        samples_start = i - window_size
        samples_end = i - 1

        # Important!
        # The training set will be taken from [samples_start:samples_end + 1]
        # and the last sample is for the prediction of next day
        df_cur_dataset = df[samples_start:samples_end + 2]

        actual, predicted = train_lstm(df_cur_dataset, look_back, epochs, plot=plot, plot_callback=plot_callback, loss_threshold=loss_threshold)
        actuals.append(actual)
        predictions.append(predicted)

    # Build a DataFrame that includes the actual and predicted results on the given data set
    df_results = df.iloc[cur_day:cur_day + num_of_days]
    cols_pred = pd.Series(predictions, index=range(cur_day, cur_day + num_of_days))
    df_results["Prediction"] = pd.Series(cols_pred)

    return actuals, predictions, df_results


def train_lstm(df, look_back, epochs, plot=False, plot_callback=None, loss_threshold=10**-4):
    # Set multi cores and print cores info
    # tf_session = tf.Session(config=tf.ConfigProto(device_count={"CPU": 8}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=2))
    # keras.backend.tensorflow_backend.set_session(tf_session)
    # print(tf_session)

    # Normalize the Close dataset (this will include Xs and Ys)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_vals = np.reshape(df["Close"].values, (len(df), 1))
    dataset = scaler.fit_transform(close_vals.astype('float32'))

    # reshape into X=t and Y=t+1
    train_x, train_y, x_inc_predict, y_inc_predict = create_lstm_dataset(dataset, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    x_inc_predict = np.reshape(x_inc_predict, (x_inc_predict.shape[0], x_inc_predict.shape[1], 1))

    # Create the LSTM network
    batch_size = 1
    model = create_model(batch_size, look_back)

    # Fit the network - use multiple epochs
    for i in range(epochs):
        print("epoch {}".format(i + 1))
        res = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
        if res.history["loss"][0] < loss_threshold:
            print("Loss is lower than threshold of {}. Quitting.".format(loss_threshold))
            break

    # make predictions
    train_predict = model.predict(x_inc_predict, batch_size=batch_size)
    model.reset_states()

    # invert predictions
    train_y = scaler.inverse_transform([train_y])
    train_predict = scaler.inverse_transform(train_predict)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(train_y[0], train_predict[:-1, 0]))
    print('Train Score RMSE: %.2f' % (trainScore))

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    if plot_callback:
        plot_callback(scaler.inverse_transform(dataset), train_predict_plot)

    if plot:
        # plot baseline and predictions
        plt.figure(figsize=(15, 10))
        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(train_predict_plot)
        plt.show()

    actual = df.iloc[-1]["Close"]
    predicted = train_predict[-1, 0]

    return actual, predicted


def create_model(batch_size, look_back, regularization_factor=0, dropout=False, custom_optimizer=True):
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
    # Actual training set
    data_x, data_y = [], []

    # X and Y sets that include the last sample (which is not part of the training set)
    # This will be used for predicting and ploting the next day
    data_inc_predict_x, data_inc_predict_y = [], []

    def add_i_features(ds_x, ds_y, i):
        # Append to the list of timeseries, a 1 dimensional array that
        # includes values of our single feature (index 0) across current look_back range
        ds_x.append(dataset[i:(i + look_back), 0])

        # Append to the list of Ys, the y of the current look_back, which is our feature
        # of next day after our look_back range
        ds_y.append(dataset[i + look_back, 0])

    # Go through all training samples
    # For example
    # if the windows size is 10
    # 11 samples will be passed
    # samples 0-9 will be used for the training set
    # sample 10 will be used for the prediction
    for i in range(len(dataset) - look_back - 1):
        add_i_features(data_x, data_y, i)
        add_i_features(data_inc_predict_x, data_inc_predict_y, i)

    # Add the last sample for prediction purposes
    add_i_features(data_inc_predict_x, data_inc_predict_y, i + 1)

    return np.array(data_x), np.array(data_y), np.array(data_inc_predict_x), np.array(data_inc_predict_y)


# If we don't have a fixed seed, every time that I train the model, it predicts different results.
# Probably we should make enough epochs in order to get to a very small loss in order to get deterministic results
np.random.seed(7)

# Read data
df = pd.read_csv('~/code/dscience/stocks/data/facebook.csv', engine='python', skipfooter=3)
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

##############################3

# # A callback to save plot data from within the training function in order to be able to plot the data in a different cell in Jupyter.
# ds1 = None
# ds2 = None
# def save_plot_data(original_dataset, predicted_shifted_dataset):
#     global ds1
#     global ds2
#     ds1 = original_dataset
#     ds2 = predicted_shifted_dataset
# numebr_of_days = 1
# first_day = int(len(df) / 2)
# res_actuals, res_predicted, df_results = predict_close_lstm(df, first_day, numebr_of_days, window_size=10, look_back=4, epochs=600, plot=False, plot_callback=save_plot_data, loss_threshold=1e-3)
# root_mse = math.sqrt(mean_squared_error(res_actuals, res_predicted))
# print('Test Score RMSE: %.2f' % (root_mse))
#
# # plot baseline and predictions
# plt.figure(figsize=(15, 10))
# plt.plot(ds1)
# plt.plot(ds2)
# plt.show()

##############################3

numebr_of_days = 100
first_day = len(df) - 100  # Same start date that we used for the polynomial regression
res_actuals, res_predicted, df_results = predict_close_lstm(df, first_day, numebr_of_days, window_size=10, look_back=4, epochs=600, plot=False,
                                                            loss_threshold=1e-3)
root_mse = math.sqrt(mean_squared_error(res_actuals, res_predicted))
print('TestScore RMSE: %.2f' % (root_mse))


plt.figure(figsize=(15,10))
plt.plot(df_results[["Date"]], df_results[["Close"]], 'b-')
plt.plot(df_results[["Date"]], df_results[["Prediction"]], 'g-')
#plt.title("Prediction of 8th day ({})".format(predicted_date.date()))
plt.xlabel('Date')
plt.ylabel('Price [$]')
plt.legend(["Actual Close", "Predicted Close"])
plt.show()

# ax = df_results.plot(x='Date', y='Close', style='b-', grid=True)
# ax = df_results.plot(x='Date', y='Prediction', style='g-', grid=True, ax=ax)
# ax.set_xlabel("Date")
# ax.set_ylabel("Price [$]")
# ax.legend(["Actual Close", "Predicted Close"])

##############################3