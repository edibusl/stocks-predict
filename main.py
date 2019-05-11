import os
import math
import datetime
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as dates
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv("~/code/dscience/stocks/data/facebook.csv", engine='python')
df.loc[:, 'Date'] = pd.to_datetime(df['Date'] ,format='%Y-%m-%d')
#df.drop(columns="OpenInt", inplace=True)
df.drop(columns="Adj Close", inplace=True)


def predict_close_linear_regression(df, cur_day, num_of_days, window_size=7):
    model = LinearRegression()

    actuals = []
    predictions = []

    # Go through all requested days and make a new traigning for every day
    for i in range(cur_day, cur_day + num_of_days):
        # For each day, train a new model using Close values of previous days.
        # Number of previous days is defined by "window_size"
        samples_start = i - window_size
        samples_end = i - 1

        # The X is the features vector and it includes just an index because we're using time series data
        # X - index
        # Y - actual Close values
        X = np.array(range(samples_start, samples_end + 1))
        Y = np.array(df["Close"][samples_start:samples_end + 1])
        X = X.reshape(window_size, 1)
        Y = Y.reshape(window_size, 1)

        # Train the model
        model.fit(X, Y)

        # Predict current day according to the linear regression trained model
        cur_day_prediction = model.predict(np.array([i]).reshape(1, 1))[0][0]
        predictions.append(cur_day_prediction)

        cur_day_actual = df.iloc[i]["Close"]
        actuals.append(cur_day_actual)

    # Build a DataFrame that includes the actual and predicted results on the given data set
    df_results = df.iloc[cur_day:cur_day + num_of_days]
    df_results = df_results.drop(columns=["Open", "High", "Low", "Volume"])
    cols_pred = pd.Series(predictions, index=range(cur_day, cur_day + num_of_days))
    df_results["Prediction"] = pd.Series(cols_pred)

    return actuals, predictions, df_results


def simulate_buy(df, stock_units, sim_type, buy_decision_threshold=0):
    random.seed(datetime.datetime.now())
    bought = False
    buy_price = 0

    stats = {
        "transactions_count": 0,
        "total_bought": 0,
        "total_revenue": 0,
        "yield": 0,
        "period_days": len(df)
    }

    for i in range(len(df)):
        tomorrow_close_prediction = df.iloc[i]["Prediction"]
        today_close = df.iloc[i]["Close"]

        # Sell what we bought yesterday (if we bought any)
        if bought:
            sell_price = today_close * stock_units
            revenue = (sell_price - buy_price)
            buy_price = 0
            stats["total_revenue"] += revenue
            bought = False

        # Buy next day
        if sim_type == "prediction":
            should_buy = tomorrow_close_prediction - today_close > buy_decision_threshold
        else:
            should_buy = random.randint(1, 2) == 1

        if should_buy:
            # Buy
            bought = True
            buy_price = today_close * stock_units
            stats["total_bought"] += buy_price
            stats["transactions_count"] += 1

    stats["yield"] = stats["total_revenue"] / stats["total_bought"] * 100

    return stats


# Split into Training, Cross Validation and Testing sets.
set_sizes = {
    "total_size": len(df),
    "train_size": int(len(df) * 0.6),
    "cv_size": int(len(df) * 0.2),
    "test_size": int(len(df) * 0.2)
}

df_train = df.iloc[:set_sizes["train_size"]]
df_train.tail()
df_cv = df.iloc[set_sizes["train_size"]:set_sizes["train_size"] + set_sizes["cv_size"]]
df_cv.tail()
df_cv.index = range(len(df_cv))
df_test = df.iloc[set_sizes["train_size"] + set_sizes["cv_size"]:]
df_test.index = range(len(df_test))


ws = 4
first_day = ws
numebr_of_days = len(df_test) - ws
res_actuals, res_predicted, df_results = predict_close_linear_regression(df_test, first_day, numebr_of_days, window_size=ws)
df_results.head()

# Simulate buying with linear regression predictions
stats_prediction = simulate_buy(df_results, 5, sim_type="prediction")
print(stats_prediction)

# Simulate buying with actual tomorrow's value "predictions"
df_actual_predictions = df_results.copy()
for index, row in df_actual_predictions.iterrows():
    try:
        df_actual_predictions.loc[index, 'Prediction'] = df_actual_predictions.loc[index + 1]['Close']
    except:
        pass
stats_actual_prediction = simulate_buy(df_actual_predictions, 5, sim_type="prediction")
print(stats_actual_prediction)

# Simulate buying with random decision
total_revenues = []
for i in range(100):
    stats_randomly = simulate_buy(df_results, 5, sim_type="random")
    total_revenues.append(stats_randomly["total_revenue"])
print(np.mean(total_revenues))


year_revenue_percents = 100 * (1 - (df_results.iloc[-1]["Close"] / df_results.iloc[0]["Close"]))
year_possible_revenues = (year_revenue_percents / 100) * stats_prediction["total_bought"]
print(year_possible_revenues)



