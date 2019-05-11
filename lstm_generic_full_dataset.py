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
import hyperopt
from hyperopt import hp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_close_lstm(df, configs):
    # If we don't have a fixed seed, every time that I train the model, it predicts different results.
    # Probably we should make enough epochs in order to get to a very small loss in order to get deterministic results
    np.random.seed(7)

    # Unpack configs
    look_back = configs["model"]["look_back"]
    batch_size = configs["model"]["batch_size"]

    # Split to 2 dataframes: train and test
    df_train, df_test = train_test_split(df, train_size=configs["general"]["train_size"], test_size=(1 - configs["general"]["train_size"]),
                                         shuffle=False)

    # Take only the training columns
    feature_columns = ["Close"]
    if configs["general"]["use_all_features"]:
        feature_columns += ["Open", "High", "Low", "Volume"]
    df_train = df_train.loc[:, feature_columns]
    df_test = df_test.loc[:, feature_columns]

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

    # Split testing set to testing set and cross validation set (50%-50%)
    cv_x, test_x = np.array_split(test_x, 2)
    cv_y, test_y = np.array_split(test_y, 2)

    # Fit again after splitting
    test_x = fit_to_batch_size(test_x, batch_size)
    cv_x = fit_to_batch_size(cv_x, batch_size)
    test_y = fit_to_batch_size(test_y, batch_size)
    cv_y = fit_to_batch_size(cv_y, batch_size)
    logger.debug("Training size: %s. Cross validation size: %s. Testing size: %s.", train_x.shape[0], cv_x.shape[0], test_x.shape[0])

    # Train the model using the training set
    model = create_model(configs, features_num=train_x.shape[2])
    model_res = model.fit(train_x, train_y, epochs=configs["model"]["epochs"], batch_size=batch_size, verbose=configs["general"]["verbose"],
                          shuffle=False, validation_data=(cv_x, cv_y))

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
        "cv_loss": model_res.history['val_loss'],
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


def create_model(configs, features_num=1):
    # Unpack configs
    batch_size = configs["model"]["batch_size"]
    look_back = configs["model"]["look_back"]
    dropout_val = configs["model"]["network"]["dropout_val"]
    regularization_factor = configs["model"]["network"]["regularization_factor"]
    lstm_neurons = configs["model"]["network"]["lstm_neurons"]

    model = Sequential()

    # Add LSTM layers
    for i, lstm_neurons_layer in enumerate(lstm_neurons):
        # We need to return sequences only for all layers except the last one
        return_sequences = (i < len(lstm_neurons) - 1)

        # Add the layer
        model.add(LSTM(lstm_neurons_layer, batch_input_shape=(batch_size, look_back, features_num), stateful=True, return_sequences=return_sequences,
                       kernel_regularizer=regularizers.l2(regularization_factor)))

        # Add dropout for the later
        if dropout_val > 0:
            model.add(Dropout(dropout_val))

    # Add output layer
    model.add(Dense(1, kernel_regularizer=regularizers.l2(regularization_factor)))

    # Optional optimizer to control the learning rate
    if configs["model"]["network"]["custom_optimizer"]:
        learning_rate = 0.005
        optimzer = optimizers.RMSprop(lr=learning_rate)
    else:
        optimzer = 'adam'

    model.compile(loss='mean_squared_error', optimizer=optimzer)

    return model


def create_lstm_dataset(dataset, look_back=1, y_column_index=0):
    data_x, data_y = [], []

    # Go through all samples, shift by 1 on every iteration
    # Look upfront to next look_back samples
    for i in range(len(dataset) - look_back - 1):
        # Get next look_back samples
        look_back_window = dataset[i:(i + look_back), 0:dataset.shape[1]]
        data_x.append(look_back_window)

        # Get the look_back+1 sample for the "y"
        data_y.append(dataset[i + look_back, y_column_index])

    # Reshape input to be [samples, time steps, features]
    data_x = np.array(data_x)
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2]))

    return data_x, np.array(data_y)


def plot_prediction_results(data, plot_predicted_vs_actual=True, plot_loss=True, plot_loss_last=False):
    # Plot the loss in every epoch during training
    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(data["model_loss"])
        plt.plot(data["cv_loss"])
        plt.title("Loss during training")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Cross validation"])
        plt.show()

    if plot_loss_last:
        plt.figure(figsize=(10, 5))
        plt.plot(data["model_loss"][30:])
        plt.plot(data["cv_loss"][30:])
        plt.title("Loss during training (last epochs)")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Cross validation"])
        plt.show()

    # Plot the actual values vs predicted values in the testing set
    if plot_predicted_vs_actual:
        plt.figure(figsize=(10, 5))
        plt.plot(data["test_y"])
        plt.plot(data["test_y_predicted"])
        plt.title("Testing set - Predicted vs Actual")
        plt.xlabel("Day")
        plt.ylabel("Close Price")
        plt.legend(["Real", "Prediction"])
        plt.show()


def run_single_configuration(configs):
    # Read data
    df = pd.read_csv(configs['general']['file'], engine='python', skipfooter=3)
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Run prediction
    res = predict_close_lstm(df, configs)
    print("Test RMSE: %.2f" % (res["test_rmse"]))

    # Plot results
    if configs["general"]["plot_results"]:
        plot_prediction_results(res)

    return res


def run_multiple_configurations(configs, batch_sizes = [20, 10, 5, 4, 3, 2, 1], look_backs = [20, 10, 7, 5, 4, 3, 2, 1]):
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
            configs["batch_size"] = batch_size
            configs["look_back"] = look_back
            res = predict_close_lstm(df, configs)

            rmse_look_back.append(res["test_rmse"])
        df_rmse.loc["batch_size {}".format(batch_size)] = rmse_look_back

    # Plot on map
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_rmse, cmap="YlGnBu", annot=True, fmt='.2f')
    plt.show()

    return df_rmse


def optimize_parameters():
    search_space = {
        'model_batch_size': hp.choice('model_batch_size', [10, 5, 3]),
        'model_look_back': hp.choice('model_look_back', [10, 7, 4, 2]),
        'network_lstm_neurons_l1': hp.choice('network_lstm_neurons_l1', [4, 20, 70, 100]),
        'network_lstm_neurons_l2': hp.choice('network_lstm_neurons_l2', [4, 20, 70, 100]),
        'network_dropout_val': hp.uniform('network_dropout_val', 0, 1),
        'network_reg_factor': hp.uniform('network_reg_factor', 0, 0.7)
    }

    def run_hyperopt_single(params):
        configs = {
            "general": {
                "file": "data/facebook.csv",
                "plot_results": False,
                "verbose": 2,
                "use_all_features": False,
                "train_size": 0.7
            },
            "model": {
                "batch_size": params['model_batch_size'],
                "look_back": params['model_look_back'],
                "epochs": 70,
                "network": {
                    "custom_optimizer": False,
                    "regularization_factor": params['network_reg_factor'],
                    "dropout_val": params['network_dropout_val'],
                    "lstm_neurons": [params['network_lstm_neurons_l1'], params['network_lstm_neurons_l2']]
                }
            }
        }
        logger.info("Running current config: %s", configs)

        res_single_run = run_single_configuration(configs)
        cv_loss = np.amin(res_single_run['cv_loss'])

        logger.info("Loss for config %s: %s", params, cv_loss)

        return {
            'status': hyperopt.STATUS_OK,
            'loss': cv_loss,
            'configs': configs,
            'run_result': res_single_run
        }

    trials = hyperopt.Trials()
    res_hyperopt = hyperopt.fmin(run_hyperopt_single, space=search_space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=1000)
    logger.info("Best result:\nConfigs: %s\nTest RMSE: %s", trials.best_trial['result']['configs'],
                trials.best_trial['result']['run_result']['test_rmse'])


if __name__ == '__main__':
    # configs = {
    #     "general": {
    #         "file": "data/facebook.csv",
    #         "plot_results": True,
    #         "verbose": 1,
    #         "use_all_features": False,
    #         "train_size": 0.8
    #     },
    #     "model": {
    #         "batch_size": 4,
    #         "look_back": 10,
    #         "epochs": 70,
    #         "network": {
    #             "custom_optimizer": False,
    #             "regularization_factor": 0,
    #             "dropout_val": 0,
    #             "lstm_neurons": [4, 4]
    #         }
    #     }
    # }
    # configs["model"]["epochs"] = 90
    # res = run_multiple_configurations(configs)

    # res = run_single_configuration(configs)

    optimize_parameters()

