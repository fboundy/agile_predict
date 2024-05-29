# %%
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# %%
fc = pd.read_hdf(r"forecast.hdf", key="Forecasts")
fd = pd.read_hdf(r"forecast.hdf", key="ForecastData")
ph = pd.read_hdf(r"forecast.hdf", key="PriceHistory")
# %%
df = (
    fd.merge(fc, right_on="id", left_on="forecast_id")
    .groupby("date_time")
    .last()
    .drop(["id_y", "id_x", "forecast_id", "name"], axis=1)
    .rename({"day_ahead": "day_ahead_pred"}, axis=1)
)
df = (
    df.merge(ph, left_index=True, right_on="date_time")
    .set_index("date_time")
    .drop(["id", "agile", "emb_wind", "created_at"], axis=1)
)
# %%
df["dow"] = df.index.day_of_week
df["time"] = df.index.hour + df.index.minute / 60
dfx = df.drop("day_ahead_pred", axis=1)
# %%
train = dfx.sample(frac=0.8, random_state=0)
test = dfx.drop(train.index)
sns.pairplot(data=train, diag_kind="kde")
# %%
train_X = train.copy()
test_X = test.copy()
train_Y = train_X.pop("day_ahead")
test_Y = test_X.pop("day_ahead")

# %%
xg_model = xg.XGBRegressor(
    objective="reg:squarederror",
    booster="dart",
    # max_depth=0,
    gamma=0.3,
    eval_metric="rmse",
)

xg_model.fit(train_X, train_Y, verbose=True)
xg_pred_Y = xg_model.predict(test_X)
fig, ax = plt.subplots(1, 1)
ax.scatter(test_Y, xg_pred_Y)

# %%
norm = tf.keras.layers.Normalization(axis=1)
norm.adapt(np.array(train_X))
np.set_printoptions(precision=3, suppress=True)
print(norm.mean.numpy())
# %%
first = np.array(train_X[:1])

with np.printoptions(precision=2, suppress=True):
    print("First example:", first)
    print()
    print("Normalized:", norm(first).numpy())
# %%
demand = np.array(train_X["demand"])

demand_normalizer = layers.Normalization(
    input_shape=[
        1,
    ],
    axis=None,
)
demand_normalizer.adapt(demand)
# %%
demand_model = tf.keras.Sequential([demand_normalizer, layers.Dense(units=1)])

demand_model.summary()
# %%
demand_model.predict(demand[:10])
# %%
demand_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error")

# %!time
history = demand_model.fit(
    train_X["demand"],
    train_Y,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2,
)
# %%
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


plot_loss(history)
# %%
test_results = {}

test_results["demand_model"] = demand_model.evaluate(test_X["demand"], test_Y, verbose=0)

x = tf.linspace(15000, 30000, 151)
y = demand_model.predict(x)


def plot_horsepower(x, y):
    plt.scatter(train_X["demand"], train_Y, label="Data")
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("demand")
    plt.ylabel("day_ahead")
    plt.legend()


plot_horsepower(x, y)
# %%
linear_model = tf.keras.Sequential([norm, layers.Dense(units=1)])
# %%
linear_model.predict(train_X[:10])
linear_model.layers[1].kernel
# %%
linear_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error")

history = linear_model.fit(
    train_X,
    train_Y,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2,
)
# %%
plot_loss(history)
# %%
test_results["linear_model"] = linear_model.evaluate(test_X, test_Y, verbose=0)


# %%
def build_and_compile_model(norm):
    model = tf.keras.Sequential(
        [norm, layers.Dense(64, activation="relu"), layers.Dense(64, activation="relu"), layers.Dense(1)]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_demand_model = build_and_compile_model(demand_normalizer)
dnn_demand_model.summary()


history = dnn_demand_model.fit(train_X["demand"], train_Y, validation_split=0.2, verbose=0, epochs=100)

plot_loss(history)

x = tf.linspace(15000, 30000, 151)
y = dnn_demand_model.predict(x)

plot_horsepower(x, y)
# %%
test_results["dnn_demand_model"] = dnn_demand_model.evaluate(test_X["demand"], test_Y, verbose=0)
# %%
dnn_model = build_and_compile_model(norm)
dnn_model.summary()

history = dnn_model.fit(train_X, train_Y, validation_split=0.2, verbose=0, epochs=500)
# %%
plot_loss(history)
# %%
test_results["dnn_model"] = dnn_model.evaluate(test_X, test_Y, verbose=0)
test_results  # %%

# %%
fig, ax = plt.subplots()
ax.scatter(test_Y, [z[0] for z in dnn_model.predict(test_X)])
ax.scatter(test_Y, xg_pred_Y)
# %%
