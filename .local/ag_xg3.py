# %%
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt


def kde_quantiles(kde, dt, pred, quantiles=(0.1,0.5, 0.9), lim=(0, 150)):
    if not isinstance(dt, list):
        dt = [dt]
    if not isinstance(pred, list):
        pred = [pred]

    results = {f"pred_{q*100:0.0f}": [] for q in quantiles}
    for dt1, pred1 in zip(dt, pred):
        x = np.array([[dt1, pred1, p] for p in range(int(lim[0]), int(lim[1]))])
        c = pd.Series(index=x[:, 2], data=np.exp(kde.score_samples(x)).cumsum())
        c /= c.iloc[-1]

        for q in quantiles:
            idx = c[c < q].index[-1]
            results[f"pred_{q*100:0.0f}"] += [(q - c[idx]) / (c[idx + 1] - c[idx]) + idx]

    return results


PLOT = False

# %%
%%time
ff = pd.read_hdf(r"forecast.hdf", key="Forecasts").set_index("id").sort_index()
fd = pd.read_hdf(r"forecast.hdf", key="ForecastData")
ph = pd.read_hdf(r"forecast.hdf", key="PriceHistory")

# ff['date']=ff['created_at'].dt.tz_convert('GB').dt.normalize()
# ff['dt1600']=(ff['date']+pd.Timedelta(hours=15,minutes=45)-ff['created_at'].dt.tz_convert('GB')).dt.total_seconds().abs()
# ff=ff.sort_values('dt1600').drop_duplicates('date').sort_index().drop(['date', 'dt1600'],axis=1)
ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)
ff_train = ff.iloc[:-1].sample(frac=0.8)
ff_test = ff.iloc[:-1].drop(ff_train.index)
df = (
    fd.merge(ff_train, right_index=True, left_on="forecast_id")
    .drop(["id", "forecast_id", "name", "created_at"], axis=1)
    .rename({"day_ahead": "day_ahead_pred"}, axis=1)
)
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

test_X = (
    fd.merge(ff_test, right_index=True, left_on="forecast_id").drop(
        ["id", "forecast_id", "name", "day_ahead", "ag_end"], axis=1
    )
).set_index("date_time")
test_X = test_X[test_X.index > test_X["ag_start"]]
test_X["dow"] = test_X.index.day_of_week
test_X["time"] = test_X.index.hour + test_X.index.minute / 60
results = pd.DataFrame(
    index=test_X.index, data={"dt": (test_X.index - test_X["created_at"]).dt.total_seconds() / 86400}
)

test_X = test_X.drop(["created_at", "ag_start", "emb_wind"], axis=1)
results["pred"] = xg_model.predict(test_X)
results = results[results.index <= ph.set_index("date_time").sort_index().index[-1]]
results = results.merge(ph[["date_time", "day_ahead"]], left_index=True, right_on="date_time").set_index("date_time")

kde = KernelDensity()
kde.fit(results[["dt", "pred", "day_ahead"]].to_numpy())

xlim = (
    np.floor(results[["pred", "day_ahead"]].min(axis=1).min() / 11) * 10,
    np.ceil(results[["pred", "day_ahead"]].max(axis=1).max() / 9) * 10,
)

if PLOT:
    fig, ax = plt.subplots(14, 2, figsize=(6, 28), sharex=True, sharey="col", layout="tight")


    ax[0, 0].set_xlim(xlim)
    ax[0, 1].set_xlim(xlim)
    ax[0, 0].set_ylim(xlim)

    for dt in range(14):
        x = results[results["dt"].astype(int) == dt]
        print(f"Delta T: {dt:3d} days  Count:{len(x):5d}  ")
        sns.kdeplot(y=x["day_ahead"], x=x["pred"], ax=ax[dt, 0], fill=True)
        pred = np.array([[dt, 90, p] for p in range(150)])
        c = np.exp(kde.score_samples(pred)).cumsum()
        c /= c[-1]
        ax[dt, 1].plot(pred[:, 2], c)



pred_X = (
    fd[fd["forecast_id"] == ff.index[-1]]
    .set_index("date_time")
    .drop(["id", "forecast_id", "emb_wind", "day_ahead"], axis=1)
)

pred_X["dow"] = pred_X.index.day_of_week
pred_X["time"] = pred_X.index.hour + pred_X.index.minute / 60
pred_y = pd.DataFrame(
    index=pred_X.index, data={"dt": (pred_X.index - ff["created_at"].iloc[-1]).total_seconds() / 86400}
)
pred_y["pred"] = xg_model.predict(pred_X)

pred_y = pd.concat(
    [
        pred_y,
        pd.DataFrame(index=pred_y.index, data=kde_quantiles(kde, pred_y["dt"].to_list(), pred_y["pred"].to_list(), lim=xlim,quantiles=(0.1,0.9))),
    ],
    axis=1,
)
#%%
for q in (10,90):
    pred_y[f'pred_{q}']=pred_y[f'pred_{q}'].rolling(3, center=True).mean()

fig, ax = plt.subplots(figsize=(16, 6))
pred_y.drop(["dt", "pred"], axis=1).plot(ax=ax, lw=1)
pred_y['pred'].plot(ax=ax,lw=3)
fd[fd["forecast_id"] == ff.index[-1]].set_index('date_time')['day_ahead'].plot(ax=ax, color='black')
# %%
