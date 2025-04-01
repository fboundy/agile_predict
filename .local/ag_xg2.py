# %%
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import matplotlib.pyplot as plt


# %%
fc = pd.read_hdf(r"forecast.hdf", key="Forecasts")
fd = pd.read_hdf(r"forecast.hdf", key="ForecastData")
ph = pd.read_hdf(r"forecast.hdf", key="PriceHistory")


# %%
fc["ag_start"] = fc["created_at"].dt.normalize() + pd.Timedelta(hours=23)
fc["ag_end"] = fc["created_at"].dt.normalize() + pd.Timedelta(hours=47)
fc_train = fc.sample(frac=0.8, random_state=0)
fc_test = fc.drop(fc_train.index)
df = (
    fd.merge(fc_train, right_on="id", left_on="forecast_id")
    .drop(["id_y", "id_x", "forecast_id", "name"], axis=1)
    .rename({"day_ahead": "day_ahead_pred"}, axis=1)
)

df = (
    df[(df["date_time"] >= df["ag_start"]) & (df["date_time"] < df["ag_end"])]
    .groupby("date_time")
    .last()
    .drop(["ag_start", "ag_end", "created_at"], axis=1)
)
df = (
    df.merge(ph, left_index=True, right_on="date_time")
    .set_index("date_time")
    .drop(["id", "agile", "emb_wind"], axis=1)
)

df["dow"] = df.index.day_of_week
df["time"] = df.index.hour + df.index.minute / 60
train_X = df.drop("day_ahead_pred", axis=1)
train_y = train_X.pop("day_ahead")

xg_model = xg.XGBRegressor(
    objective="reg:squarederror",
    booster="dart",
    # max_depth=0,
    gamma=0.3,
    eval_metric="rmse",
    n_estimators=100,
)

xg_model.fit(train_X, train_y, verbose=True)

phx = ph.set_index("date_time").sort_index()

for id in fc_test["id"]:
    test_X = fd[fd["forecast_id"] == id].drop(["id", "forecast_id", "emb_wind"], axis=1).set_index("date_time")
    test_X = test_X[test_X.index > fc[fc["id"] == id]["ag_start"].iloc[0]]
    test_X["dow"] = test_X.index.day_of_week
    test_X["time"] = test_X.index.hour + test_X.index.minute / 60
    test_y = pd.Series(index=test_X.index, data=xg_model.predict(test_X.drop("day_ahead", axis=1)))
    fig, ax = plt.subplots(figsize=(16, 6))
    test_y.plot(ax=ax, label="New Model")

    phx.loc[test_X.index[0] : min(test_X.index[-1], phx.index[-1]), "day_ahead"].plot(
        ax=ax, color="black", label="Actual"
    )
    test_X["day_ahead"].plot(ax=ax, label="Old Model")
    ax.legend()

# %%
