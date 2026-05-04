import pandas as pd


FEATURE_SETS = {
    "default": (
        "bm_wind",
        "solar",
        "demand",
        "peak",
        "days_ago",
        "wind_10m",
        "weekend",
    ),
    "minimal": (
        "demand",
        "peak",
        "days_ago",
        "weekend",
    ),
    "generation": (
        "bm_wind",
        "solar",
        "emb_wind",
        "demand",
        "peak",
        "days_ago",
        "weekend",
    ),
    "weather": (
        "bm_wind",
        "solar",
        "demand",
        "peak",
        "days_ago",
        "wind_10m",
        "temp_2m",
        "rad",
        "weekend",
    ),
    "default_fuel": (
        "bm_wind",
        "solar",
        "demand",
        "peak",
        "days_ago",
        "wind_10m",
        "weekend",
        "nuclear",
        "gas_ttf",
    ),
}


def resolve_feature_columns(feature_set="generation", explicit_features=None, drop_features=None, no_day_of_week=False):
    if explicit_features:
        features = [feature.strip() for feature in explicit_features.split(",") if feature.strip()]
    else:
        try:
            features = list(FEATURE_SETS[feature_set])
        except KeyError as exc:
            valid = ", ".join(sorted(FEATURE_SETS))
            raise ValueError(f"Unknown feature set '{feature_set}'. Valid feature sets: {valid}") from exc

    if no_day_of_week:
        features = [feature for feature in features if feature not in {"day_of_week", "dow"}]

    for feature in drop_features or []:
        features = [candidate for candidate in features if candidate != feature]

    if not features:
        raise ValueError("At least one feature column is required.")

    return features


def add_derived_features(df, now=None):
    df = df.copy()
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    df["dow"] = df.index.day_of_week
    df["weekend"] = (df.index.day_of_week >= 5).astype(int)
    df["time"] = df.index.tz_convert("GB").hour + df.index.minute / 60
    df["days_ago"] = (now - df["created_at"]).dt.total_seconds() / 3600 / 24
    df["dt"] = (df.index - df["created_at"]).dt.total_seconds() / 3600 / 24
    df["peak"] = ((df["time"] >= 16) & (df["time"] < 19)).astype(float)
    return df


def add_latest_forecast_features(fc, now=None):
    fc = fc.copy()
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    fc["dow"] = fc.index.day_of_week
    fc["weekend"] = (fc.index.day_of_week >= 5).astype(int)
    fc["days_ago"] = 0
    fc["time"] = fc.index.tz_convert("GB").hour + fc.index.minute / 60
    fc["dt"] = (fc.index - now).total_seconds() / 86400
    fc["peak"] = ((fc["time"] >= 16) & (fc["time"] < 19)).astype(float)
    return fc


def build_forecast_frame(forecast_data, forecasts):
    ff = forecasts.copy().set_index("id").sort_index()
    ff["created_at"] = pd.to_datetime(ff["name"]).dt.tz_localize("GB")
    ff["date"] = ff["created_at"].dt.tz_convert("GB").dt.normalize()
    ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
    ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)

    df = (forecast_data.merge(ff, right_index=True, left_on="forecast_id")).set_index("date_time")
    if "day_ahead" in df.columns:
        df = df.drop("day_ahead", axis=1)
    return add_derived_features(df), ff


def select_daily_training_forecasts(forecasts):
    forecasts = forecasts.copy()
    forecasts["dt1600"] = (
        (forecasts["date"] + pd.Timedelta(hours=16, minutes=15) - forecasts["created_at"].dt.tz_convert("GB"))
        .dt.total_seconds()
        .abs()
    )
    return forecasts.sort_values("dt1600").drop_duplicates("date").sort_index().drop(["date", "dt1600"], axis=1)


def validate_feature_columns(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature column(s): {', '.join(missing)}")


def build_training_data(df, training_forecasts, prices, features, max_days):
    validate_feature_columns(df, features)
    train_X = df[df["forecast_id"].isin(training_forecasts.index)]
    train_X = train_X[train_X["days_ago"] < max_days]
    train_X = train_X[(train_X.index >= train_X["ag_start"]) & (train_X.index < train_X["ag_end"])][features]
    train_X = train_X.merge(prices["day_ahead"], left_index=True, right_index=True)
    train_y = train_X.pop("day_ahead")
    return train_X, train_y


def build_holdout_data(df, training_forecasts, prices, max_days):
    test_X = df[~df["forecast_id"].isin(training_forecasts.index)]
    test_X = test_X[test_X.index > test_X["ag_start"]]
    test_X = test_X[test_X["days_ago"] < max_days]
    test_X = test_X.merge(prices["day_ahead"], left_index=True, right_index=True)
    test_y = test_X["day_ahead"]
    return test_X, test_y


def latest_prediction_features(fc, features):
    validate_feature_columns(fc, features)
    return fc.reindex(features, axis=1)
