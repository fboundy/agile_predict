# AgilePredict

Forecasts Octopus Agile electricity import and export prices up to 14 days ahead using an ensemble machine learning model. Covers all Agile regions (A–P) plus a national aggregate.

## Data sources

| Source | Data |
|---|---|
| [Elexon BMRS](https://bmrs.elexon.co.uk/) | UK nuclear availability, demand |
| [NESO](https://www.nationalgrideso.com/data-portal) | Wind, solar, embedded wind and demand forecasts; daily operating margin reserve (OPMR) |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | French nuclear generation (interconnector signal) |
| [Open-Meteo](https://open-meteo.com) | UK and French temperature, wind speed, radiation (forecast + ensemble) |
| [Octopus Energy](https://developer.octopus.energy/docs/api/) | Agile tariff prices (actuals) |
| [Nord Pool](https://www.nordpoolgroup.com/) | GB60 day-ahead prices |
| [Yahoo Finance](https://finance.yahoo.com/) | TTF natural gas futures |

## Model

Three-model ensemble (CatBoost, LightGBM, ExtraTrees) trained on a rolling 28-day window of half-hourly forecasts. Features include UK generation mix (wind, solar, nuclear), demand, gas price (TTF), UK and French weather, French nuclear output, and NESO operating margin reserve (OPMR). Forecast intervals are derived empirically from holdout residuals binned by horizon and from Open-Meteo ensemble weather perturbations.

## Development setup

The project runs on Python with Django 4.2. Dependencies are managed with conda.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Create the environment

```bash
conda env create -f environment.yml
conda activate agile_predict
```

### Configure secrets

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

### Run migrations and start the dev server

```bash
python manage.py migrate
python manage.py runserver
```

The dev server starts at `http://localhost:8000`.

### Run the forecast update manually

```bash
python manage.py update
```

This fetches all upstream data, retrains the ensemble, and writes new `ForecastData` and `AgileData` rows.

Key flags:

| Flag | Description |
|---|---|
| `--feature_set <name>` | Override the stored optimal feature set for this run |
| `--features col1,col2,...` | Specify exact feature columns instead of a named set |
| `--drop_feature <col>` | Remove a column from the selected feature set (repeatable) |
| `--force_experiment` | Force the feature-set experiment regardless of the 14-day schedule |
| `--debug` | Verbose logging of intermediate data and shapes |

### Feature experiments

Every 14 days the update run automatically evaluates all candidate feature sets defined in `prices/forecast_features.py` (`EXPERIMENT_FEATURE_SETS`) using walk-forward cross-validation (5 folds, 21-day train / 3-day test). Each set is scored on a weighted combination of MAE and RMSE, with short-horizon forecasts (<3 days) weighted 3× and medium-horizon (<7 days) weighted 2×. The winning set is persisted in the database and used for all subsequent runs until the next experiment.

Current candidate sets: `generation`, `weather`, `fuel`, `weather_fuel`, `weather_fuel_fr`, `fr_weather`, `fr_weather_opmr`, `fr_weather_nuclear`, `weather_fuel_fr_weather`, `full`, `generation_gas_opmr`, `fr_weather_gas_opmr`.

To force a re-evaluation immediately:

```bash
python manage.py update --force_experiment
```

To lock in a specific set for a single run (bypasses the experiment result):

```bash
python manage.py update --feature_set weather_fuel_fr
```

## Production

Hosted on [fly.io](https://fly.io) (app name: `prices`). Deploy with:

```bash
fly deploy --app prices
```

Migrations run automatically as the `release_command` before each rolling update.

The daily forecast update and Agile price refresh are triggered by cron jobs on the self-hosted Proxmox CT that POST to the fly.io web app (`bin/cron_update.sh`, `bin/cron_latest_agile.sh`). The CT also runs a local dev server started at boot via `bin/runserver.sh`.

The chart UI defaults to the v2 interface (`/v2/<region>/`). Forecast comparison overlays from [AgileForecast](https://agileforecast.co.uk) and [X2R](https://x2r.uk) can be toggled on; if a live fetch fails the UI falls back to the most recent stored data and shows a status indicator.

## Project structure

```
config/         Django settings, URLs, and shared utilities (data fetching, model helpers)
prices/         Main app: models, views, forecast pipeline, management commands
api/            REST API (forecast, accuracy, metadata endpoints)
templates/      Jinja-style Django templates (v2 UI and classic UI)
home_assistant/ Example HA sensor and Apex chart YAML
```
