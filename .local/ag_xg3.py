# %%
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt


# %%
fc = pd.read_hdf(r"forecast.hdf", key="Forecasts").set_index("id").sort_index()
fd = pd.read_hdf(r"forecast.hdf", key="ForecastData")
ph = pd.read_hdf(r"forecast.hdf", key="PriceHistory")

fc["ag_start"] = fc["created_at"].dt.normalize() + pd.Timedelta(hours=22)
fc["ag_end"] = fc["created_at"].dt.normalize() + pd.Timedelta(hours=46)
fc_train = fc.iloc[:-1].sample(frac=0.8)
fc_test = fc.iloc[:-1].drop(fc_train.index)
# %%
df = (
    fd.merge(fc_train, right_index=True, left_on="forecast_id")
    .drop(["id", "forecast_id", "name", "created_at"], axis=1)
    .rename({"day_ahead": "day_ahead_pred"}, axis=1)
)
# %%
df = (
    df[(df["date_time"] >= df["ag_start"]) & (df["date_time"] < df["ag_end"])]
    .groupby("date_time")
    .last()
    .drop(["ag_start", "ag_end"], axis=1)
)
df = (
    df.merge(ph, left_index=True, right_on="date_time")
    .set_index("date_time")
    .drop(["id", "agile", "emb_wind"], axis=1)
)
# %%
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
results = []
for id in fc_test.index:
    test_X = fd[fd["forecast_id"] == id].drop(["id", "forecast_id", "emb_wind"], axis=1).set_index("date_time")
    test_X = test_X[test_X.index > fc.loc[id]["ag_start"]]
    test_X["dow"] = test_X.index.day_of_week
    test_X["time"] = test_X.index.hour + test_X.index.minute / 60
    test_y = pd.Series(index=test_X.index, data=xg_model.predict(test_X.drop("day_ahead", axis=1)))
    fig, ax = plt.subplots(figsize=(16, 6))
    test_y.plot(ax=ax, label="New Model")
    if phx.index[-1] > test_X.index[0]:
        phx.loc[test_X.index[0] : min(test_X.index[-1], phx.index[-1]), "day_ahead"].plot(
            ax=ax, color="black", label="Actual"
        )
        test_X["day_ahead"].plot(ax=ax, label="Old Model")
        ax.legend()
        y_true = phx.loc[test_X.index[0] :, "day_ahead"].iloc[: 48 * 3]
        y_old = test_X["day_ahead"].iloc[: len(y_true)]
        y_new = test_y.iloc[: len(y_true)]
        results += [
            pd.concat(
                [phx.loc[test_X.index[0] : min(test_X.index[-1], phx.index[-1]), "day_ahead"], test_y], axis=1
            ).dropna()
        ]
        results[-1]["dt"] = (results[-1].index - fc.loc[id]["created_at"]).total_seconds() / 24 / 3600
        print(f"ID: {id:3d} Old: {MSE(y_true,y_old)**0.5:5.1f} New: {MSE(y_true,y_new)**0.5:5.1f}")

results = pd.concat(results).rename({0: "pred"}, axis=1)
results["error"] = results["pred"] - results["day_ahead"]
# %%
test_X = (
    fd.merge(fc_test, right_index=True, left_on="forecast_id").drop(
        ["id", "forecast_id", "name", "day_ahead", "ag_end"], axis=1
    )
).set_index("date_time")
test_X = test_X[test_X.index > fc.loc[id]["ag_start"]]
test_X["dow"] = test_X.index.day_of_week
test_X["time"] = test_X.index.hour + test_X.index.minute / 60
results = pd.DataFrame(
    index=test_X.index, data={"dt": (test_X.index - test_X["created_at"]).dt.total_seconds() / 86400}
)
test_X = test_X.drop(["created_at", "ag_start", "emb_wind"], axis=1)
results["pred"] = xg_model.predict(test_X)
results = results[results.index <= ph.set_index("date_time").sort_index().index[-1]]
results = results.merge(ph[["date_time", "day_ahead"]], left_index=True, right_on="date_time").set_index("date_time")
# %%
fig, ax = plt.subplots(14, 2, figsize=(6, 28), sharex=True, sharey="col", layout="tight")
kde = KernelDensity()
kde.fit(results[["dt", "pred", "day_ahead"]].to_numpy())
for dt in range(14):
    x = results[results["dt"].astype(int) == dt]
    print(f"Delta T: {dt:3d} days  Count:{len(x):5d}  ")
    sns.kdeplot(y=x["day_ahead"], x=x["pred"], ax=ax[dt, 0], fill=True)
    pred = np.array([[dt, 90, p] for p in range(150)])
    c = np.exp(kde.score_samples(pred)).cumsum()
    c /= c[-1]
    ax[dt, 1].plot(pred[:, 2], c)


# %%
def kde_quantiles(kde, dt, pred, quantiles=(0.1, 0.5, 0.9)):
    x = np.array([[dt, pred, p] for p in range(150)])
    c = pd.Series(index=x[:, 2], data=np.exp(kde.score_samples(x)).cumsum())
    c /= c.iloc[-1]
    return {q: c[c < q].index[-1] for q in quantiles}


# %%
