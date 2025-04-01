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

ff['date']=ff['created_at'].dt.tz_convert('GB').dt.normalize()

ff['dt1600']=(ff['date']+pd.Timedelta(hours=15,minutes=45)-ff['created_at'].dt.tz_convert('GB')).dt.total_seconds().abs()

ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)

ff_filt=ff.sort_values('dt1600').drop_duplicates('date').sort_index().drop(['date', 'dt1600'],axis=1)


#%%
df = (
    fd.merge(ff, right_index=True, left_on="forecast_id")
    .drop(["id", "name" ], axis=1)
    .rename({"day_ahead": "day_ahead_pred"}, axis=1)
).set_index('date_time')

df['dt']=(df.index-df['created_at']).dt.total_seconds()/3600/24
#%%
# df = (
#     df[(df["date_time"] >= df["ag_start"]) & (df["date_time"] < df["ag_end"])]
#     .drop(["ag_start", "ag_end"], axis=1)
# ).set_index('date_time')
#%%

df = (
    df.merge(ph, left_index=True, right_on="date_time")
    .set_index("date_time")
    .drop(["id", "agile"], axis=1)
)

df["dow"] = df.index.day_of_week
df["time"] = df.index.hour + df.index.minute / 60

df['gen']=df['bm_wind']+df['solar']+df['emb_wind']
df['surplus']=(df['gen']-df['demand'])/1000
df['weekend']=(df['dow']>5).astype(int)
df['days_ago']=(pd.Timestamp.now(tz='GB')-df['created_at']).dt.total_seconds()/3600/24

df=df.drop('created_at',axis=1)
# #%%
# fig,ax=plt.subplots(2,2,figsize=(12,12), sharex=True, sharey=True)
# ax=ax.flatten()
# for i in range(2):
#     df[df['weekend']==i].plot.scatter(x='surplus', y='day_ahead', c=f"C{i}", ax=ax[1])

#     df[df['weekend']==i].plot.scatter(x='surplus', y='day_ahead', c='time', ax=ax[i+2],  cmap='viridis')
#     ax[i+2].set_xlabel("Renewable Surplus [GW]")
#     ax[i*2].set_ylabel("Day Ahead Price [£/MW]")


# df.plot.scatter(x='surplus', y='day_ahead', c='forecast_id', ax=ax[0], cmap='viridis')
# ax[0].set_ylabel("Day Ahead Price [£/MW]")
# ax[2].set_title("Weekdays")
# ax[3].set_title("Weekends")


#%%
xg_model = xg.XGBRegressor(
    objective="reg:squarederror",
    booster="dart",
    # max_depth=0,
    gamma=1,
    subsample=0.5,
    n_estimators=100,
    max_depth=10,
    colsample_bytree=1,
)

cols = ['bm_wind', 'solar', 'demand', 'time', 'days_ago',  'weekend','day_ahead',]

train_X = df[df["forecast_id"].isin(ff_filt.index[:-1])]
train_X=train_X[(train_X.index>=train_X['ag_start']) & (train_X.index<train_X['ag_end'])][cols]
sns.pairplot(train_X)
train_y = train_X.pop("day_ahead")
xg_model.fit(train_X, train_y)

test_X = df[~df["forecast_id"].isin(ff_filt.index)][cols]
test_y = test_X.pop("day_ahead")

# ax[0].scatter(test_y, pred_y,)
results = pd.DataFrame(df[~df["forecast_id"].isin(ff_filt.index)][['dt', 'day_ahead']])
results['pred']= xg_model.predict(test_X)
#%%
fig,ax=plt.subplots(1,2, figsize=(12,6))
results.plot.scatter(y='day_ahead', x='pred', c='dt', cmap='viridis', ax=ax[0])
pd.Series(index=cols[:-1], data=xg_model.feature_importances_).T.plot.bar(ax=ax[1])
#%%
fig,ax=plt.subplots(4,4,figsize=(16,16), sharex=True, sharey=True, )
fig.subplots_adjust(hspace=0, wspace=0)
ax=ax.flatten()
ax[0].axis='equal'
for dt in range(1,14):
    x=results[(results['dt']<dt) & (results['dt']>=dt-1)]
    print(dt, MSE(x['day_ahead'], x['pred'])**0.5)
    sns.kdeplot(y=x['day_ahead'], x=x['pred'],fill=True, ax=ax[dt-1])
    # x.plot.scatter(y='day_ahead', x='pred', ax=ax[dt])

#%%
kde = KernelDensity()
kde.fit(results[["dt", "pred", "day_ahead"]].to_numpy())

xlim = (
    np.floor(results[["pred", "day_ahead"]].min(axis=1).min() / 11) * 10,
)

if PLOT:
    # fig, ax = plt.subplots(14, 2, figsize=(12, 28), sharex=True, sharey="col", layout="tight", width_ratios=[1,3])


    # ax[0, 0].set_xlim(xlim)
    # ax[0, 1].set_xlim(xlim)
    # ax[0, 0].set_ylim(xlim)

    # for dt in range(14):
    #     x = results[results["dt"].astype(int) == dt]
    #     print(f"Delta T: {dt:3d} days  Count:{len(x):5d}  ")
    #     sns.kdeplot(y=x["day_ahead"], x=x["pred"], ax=ax[dt, 0], fill=True)
    #     for y in range(20, 120, 20):
    #         pred = np.array([[dt, p, y] for p in range(150)])
    #         c = np.exp(kde.score_samples(pred)).cumsum()
    #         c /= c[-1]
    #         ax[dt, 1].plot(pred[:, 2], c, color=f"C{int(y/20+2)}")
    #         ax[dt, 1].plot((y,y),(0,1), ls='--', color=f"C{int(y/20+2)}")
    #     ax[dt,0].plot((0,150),(0,150), color='black', ls='--')

    for dt in range(14):
        fig,ax=plt.subplots(1,2, figsize=(18,6), sharex=True, width_ratios=[1,2])
        x = results[results["dt"].astype(int) == dt]
        print(f"Delta T: {dt:3d} days  Count:{len(x):5d}  ")
        sns.kdeplot(y=x["day_ahead"], x=x["pred"], ax=ax[0], fill=True)
        ax[0].plot((10,130),(10,130), color='black', ls='--', lw=1)
        diag=np.array([[dt, d, d] for d in range(10,130)])
        c_max = np.max(np.exp(kde.score_samples(diag)))
        for p in range(10,130):
            slice=np.array([[dt, p, d] for d in range(10,130)])
            c = np.exp(kde.score_samples(slice))/c_max
            ax[0].plot(p+c*5,slice[:,2], color='gray', lw=1)
            if p % 10 == 0 :
                ax[1].plot(slice[:,2],c.cumsum()/c.cumsum()[-1], lw=1)
                
#%%
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
