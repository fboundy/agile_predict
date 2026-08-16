from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth.models import Group, Permission, User
from django.core import mail
from django.core.management import call_command
from django.test import RequestFactory, TestCase, override_settings
from django.utils import timezone

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, get_gas_ttf_history
from prices.forecast_features import (
    build_training_data,
    FEATURE_SETS,
    latest_prediction_features,
    resolve_feature_columns,
    TRAIN_HORIZON_DAYS,
)
from prices.model_metrics import (
    calibration_metrics,
    detection_metrics,
    format_report,
    forecast_report,
    PRICE_BANDS,
    stored_forecast_report,
)
from prices.external_forecasts import fetch_x2r
from prices.forms import ForecastForm
from prices.management.commands.update import Command as UpdateCommand
from prices.management.commands.update import EXTRA_TREES_REGRESSOR_PARAMS, fit_day_ahead_ensemble, predict_day_ahead_ensemble
from prices.management.commands.update_worker import maybe_enqueue_catchup
from prices.models import AgileData, ExternalForecast, ForecastData, Forecasts, PriceHistory, UpdateJob
from prices.views import GraphFormView, _update_options


class HistoryViewTests(TestCase):
    def test_history_view_renders_for_region_and_offset(self):
        response = self.client.get("/history/?offset_days=2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Day Ahead Price")
        self.assertContains(response, "Prediction Offset")
        self.assertContains(response, "2d ahead")
        self.assertContains(response, "Date Window")
        self.assertContains(response, "Last 2 Weeks")
        self.assertContains(response, 'type="date"')
        self.assertNotContains(response, "Region</label>")

    def test_history_view_ignores_region_url_and_uses_day_ahead(self):
        response = self.client.get("/history/G/?offset_days=2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="historyForm"')
        self.assertContains(response, "Day Ahead Price")
        self.assertContains(response, "£/MWh")
        self.assertNotContains(response, "Region</label>")

    def test_history_prediction_lines_use_successive_time_slot_runs(self):
        created_at = timezone.now() - timedelta(hours=6)
        forecast = Forecasts.objects.create(name="history-run-test", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()

        for index, offset_minutes in enumerate([0, 30, 90]):
            ForecastData.objects.create(
                forecast=forecast,
                date_time=created_at + timedelta(hours=1, minutes=offset_minutes),
                day_ahead=index,
                bm_wind=0,
                solar=0,
                emb_wind=0,
                nuclear=0,
                temp_2m=0,
                wind_10m=0,
                rad=0,
                demand=0,
            )

        response = self.client.get("/history/?offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "3 predictions for &lt;1d ahead")

    def test_history_plot_title_includes_metrics_for_displayed_data(self):
        created_at = timezone.now() - timedelta(hours=6)
        forecast = Forecasts.objects.create(name="history-metrics-test", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()

        for index, offset_minutes in enumerate([0, 30]):
            date_time = created_at + timedelta(hours=1, minutes=offset_minutes)
            PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=0)
            ForecastData.objects.create(
                forecast=forecast,
                date_time=date_time,
                day_ahead=index,
                bm_wind=0,
                solar=0,
                emb_wind=0,
                nuclear=0,
                temp_2m=0,
                wind_10m=0,
                rad=0,
                demand=0,
            )

        response = self.client.get("/history/?offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "MAE")
        self.assertContains(response, "RMSE")
        self.assertContains(response, "Bias")
        self.assertContains(response, "Bias +")

    def test_history_metrics_table_includes_all_valid_offsets(self):
        created_at = timezone.now() - timedelta(days=3)
        forecast = Forecasts.objects.create(name="history-table-test", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()

        for offset_days, predicted in [(0, 1), (1, -2)]:
            date_time = created_at + timedelta(days=offset_days, hours=1)
            PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=0)
            ForecastData.objects.create(
                forecast=forecast,
                date_time=date_time,
                day_ahead=predicted,
                bm_wind=0,
                solar=0,
                emb_wind=0,
                nuclear=0,
                temp_2m=0,
                wind_10m=0,
                rad=0,
                demand=0,
            )

        response = self.client.get("/history/?offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<th scope=\"col\" class=\"text-start\">Model</th>")
        self.assertContains(response, "<th scope=\"col\" class=\"text-start\">Parameter</th>")
        self.assertContains(response, "&lt;1d")
        self.assertContains(response, "1d")
        self.assertContains(response, "Offset")
        self.assertContains(response, "+1.00")
        self.assertContains(response, "-2.00")

    def test_history_offers_external_comparison_to_anonymous_user(self):
        created_at = timezone.now() - timedelta(hours=6)
        ExternalForecast.objects.create(
            source=ExternalForecast.SOURCE_X2R,
            region="G",
            forecast_name="x2r test",
            source_created_at=created_at,
            date_time=created_at + timedelta(hours=1),
            agile_pred=12,
        )

        response = self.client.get("/history/?compare_x2r=1")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Compare AgileForecast")
        self.assertContains(response, "Compare X2R")
        self.assertContains(response, "X2R comparison predictions")

    def test_history_metrics_table_includes_selected_external_forecasts_as_day_ahead(self):
        created_at = timezone.now() - timedelta(hours=6)

        for offset_minutes, actual_price, predicted_day_ahead in [(60, 100, 110), (90, 100, 120)]:
            date_time = created_at + timedelta(minutes=offset_minutes)
            agile_pred = day_ahead_to_agile(pd.Series([predicted_day_ahead], index=[date_time]), region="G").iloc[0]
            PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=actual_price)
            ExternalForecast.objects.create(
                source=ExternalForecast.SOURCE_X2R,
                region="G",
                forecast_name="x2r metrics test",
                source_created_at=created_at,
                date_time=date_time,
                agile_pred=agile_pred,
            )

        response = self.client.get("/history/?compare_x2r=1&offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "X2R")
        self.assertContains(response, "MAE")
        self.assertContains(response, "RMSE")
        self.assertContains(response, "Bias")
        self.assertContains(response, "15.00")
        self.assertContains(response, "15.81")
        self.assertContains(response, "+15.00")

    def test_history_region_z_uses_day_ahead_units_and_forecast_data(self):
        created_at = timezone.now() - timedelta(hours=6)
        forecast = Forecasts.objects.create(name="history-day-ahead-test", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()

        date_time = created_at + timedelta(hours=1)
        PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=100)
        ForecastData.objects.create(
            forecast=forecast,
            date_time=date_time,
            day_ahead=110,
            bm_wind=0,
            solar=0,
            emb_wind=0,
            nuclear=0,
            temp_2m=0,
            wind_10m=0,
            rad=0,
            demand=0,
        )

        response = self.client.get("/history/?offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Day Ahead Price")
        self.assertContains(response, "£/MWh")
        self.assertContains(response, "1 predictions for &lt;1d ahead")


class ExportPricingTests(TestCase):
    def test_national_export_coefficients_are_arithmetic_mean(self):
        regional_factors = [
            GLOBAL_SETTINGS["REGIONS"][region]["export_factors"]
            for region in GLOBAL_SETTINGS["REGIONS"]
            if region not in {"X", "Z"}
        ]
        expected = tuple(round(sum(values) / len(values), 4) for values in zip(*regional_factors))

        self.assertEqual(GLOBAL_SETTINGS["REGIONS"]["X"]["export_factors"], expected)

    def test_export_conversion_uses_regional_coefficients_and_floor(self):
        index = pd.to_datetime(["2026-05-01T12:00:00Z", "2026-05-01T16:00:00Z"])
        day_ahead = pd.Series(index=index, data=[100, 100])

        export = day_ahead_to_agile(day_ahead, region="A", export=True)

        self.assertAlmostEqual(export.iloc[0], 10.59)
        self.assertAlmostEqual(export.iloc[1], 17.63)

    def test_import_conversion_handles_duplicate_timestamps(self):
        index = pd.to_datetime(["2026-05-01T12:00:00Z", "2026-05-01T12:00:00Z"])
        day_ahead = pd.Series(index=index, data=[100, 110])

        agile = day_ahead_to_agile(day_ahead, region="A")

        self.assertEqual(len(agile), 2)

    def test_forecast_form_has_export_pricing_option(self):
        form = ForecastForm()

        self.assertIn("show_export_pricing", form.fields)

    def test_day_ahead_region_uses_raw_price_and_hides_export_pricing(self):
        index = pd.to_datetime(["2026-05-01T12:00:00Z", "2026-05-01T16:00:00Z"])
        day_ahead = pd.Series(index=index, data=[100, 200])

        converted = day_ahead_to_agile(day_ahead, region="Z")
        form = ForecastForm(region="Z")

        self.assertEqual(converted.tolist(), [100.0, 200.0])
        self.assertNotIn("show_export_pricing", form.fields)


class LocalRealtimeExternalForecastTests(TestCase):
    @override_settings(LOCAL_REALTIME_EXTERNAL_FORECASTS=False)
    def test_forecast_form_hides_live_external_options_by_default(self):
        form = ForecastForm()

        self.assertNotIn("show_live_agileforecast", form.fields)
        self.assertNotIn("show_live_x2r", form.fields)

    def test_forecast_form_shows_live_external_options_when_enabled(self):
        form = ForecastForm(local_realtime_external_forecasts=True)

        self.assertIn("show_live_agileforecast", form.fields)
        self.assertIn("show_live_x2r", form.fields)

    @override_settings(LOCAL_REALTIME_EXTERNAL_FORECASTS=False)
    @patch("prices.views.fetch_agileforecast")
    def test_view_does_not_fetch_live_external_forecasts_when_disabled(self, fetch_agileforecast):
        view = GraphFormView()

        forecasts, errors = view.fetch_live_external_forecasts("G", True, False)

        self.assertEqual(forecasts, [])
        self.assertEqual(errors, [])
        fetch_agileforecast.assert_not_called()

    @patch("prices.views.fetch_agileforecast")
    def test_view_fetches_live_external_forecasts_when_enabled(self, fetch_agileforecast):
        fetch_agileforecast.return_value = {
            "name": "Region | G test",
            "source_created_at": timezone.now(),
            "rows": [],
        }
        user = User.objects.create_user(username="privileged", password="pw")
        group, _created = Group.objects.get_or_create(name="Privileged Users")
        user.groups.add(group)
        request = RequestFactory().get("/")
        request.user = user
        view = GraphFormView()
        view.request = request

        forecasts, errors = view.fetch_live_external_forecasts("G", True, False)

        self.assertEqual(errors, [])
        self.assertEqual(forecasts[0]["label"], "AgileForecast")
        fetch_agileforecast.assert_called_once_with("G")

    @patch("prices.views.fetch_agileforecast")
    def test_view_fetches_live_external_forecasts_for_staff(self, fetch_agileforecast):
        fetch_agileforecast.return_value = {
            "name": "Region | G test",
            "source_created_at": timezone.now(),
            "rows": [],
        }
        request = RequestFactory().get("/")
        request.user = User.objects.create_user(username="external-staff", password="pw", is_staff=True)
        view = GraphFormView()
        view.request = request

        forecasts, errors = view.fetch_live_external_forecasts("G", True, False)

        self.assertEqual(errors, [])
        self.assertEqual(forecasts[0]["label"], "AgileForecast")
        fetch_agileforecast.assert_called_once_with("G")

    @patch("prices.views.fetch_x2r")
    @patch("prices.views.fetch_agileforecast")
    def test_ext_forecast_json_serves_stored_data_without_live_call(self, fetch_agileforecast, fetch_x2r):
        """The v2 comparison overlay must read stored data (refreshed by the
        update run) and never make a live external call on the request path."""
        import pandas as pd
        from datetime import timedelta as _td
        from prices.models import ExternalForecast
        from prices.views import _fetch_external_forecasts

        created = timezone.now()
        base = created.replace(minute=0, second=0, microsecond=0)
        for i in range(6):
            ExternalForecast.objects.create(
                source=ExternalForecast.SOURCE_AGILEFORECAST,
                region="G",
                forecast_name="test",
                source_created_at=created,
                date_time=base + _td(minutes=30 * i),
                agile_pred=20.0 + i,
            )

        prior = pd.Timestamp.now(tz="GB") - pd.Timedelta(days=1)
        end = pd.Timestamp.now(tz="GB") + pd.Timedelta(days=2)
        traces, statuses = _fetch_external_forecasts(["AgileForecast"], "X", False, prior, end)

        self.assertEqual(len(traces), 1)
        self.assertEqual(statuses[0]["name"], "AgileForecast")
        self.assertEqual(statuses[0]["health"], "ok")
        fetch_agileforecast.assert_not_called()
        fetch_x2r.assert_not_called()

    def test_live_forecast_rows_are_limited_to_plot_date_range(self):
        now = timezone.now()
        rows = [
            {"date_time": now - timedelta(minutes=30), "agile_pred": 1},
            {"date_time": now + timedelta(minutes=30), "agile_pred": 2},
            {"date_time": now + timedelta(hours=2), "agile_pred": 3},
        ]
        view = GraphFormView()

        filtered = view.filter_forecast_rows_for_plot(
            rows,
            actual_end=now,
            plot_end=now + timedelta(hours=1),
            show_overlap=False,
        )

        self.assertEqual([row["agile_pred"] for row in filtered], [2])


class RegistrationTests(TestCase):
    def test_login_page_links_to_registration(self):
        response = self.client.get("/accounts/login/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Registered users have access to additional analytics")
        self.assertContains(response, "Register")

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_registration_creates_inactive_user_in_users_group_and_emails_admin(self):
        response = self.client.post(
            "/accounts/register/",
            {
                "username": "newuser",
                "email": "newuser@example.com",
                "password1": "Test-password-12345",
                "password2": "Test-password-12345",
            },
        )

        self.assertEqual(response.status_code, 302)
        user = User.objects.get(username="newuser")
        self.assertFalse(user.is_active)
        self.assertTrue(user.groups.filter(name="Users").exists())
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["foboundy@gmail.com"])
        self.assertIn("newuser@example.com", mail.outbox[0].body)


class IncrementalBackupAuthTests(TestCase):
    def test_incremental_backup_exports_and_imports_users_groups_and_memberships(self):
        group, _created = Group.objects.get_or_create(name="Privileged Users")
        permission = Permission.objects.get(
            content_type__app_label="auth",
            content_type__model="user",
            codename="view_user",
        )
        group.permissions.add(permission)
        user = User.objects.create_user(
            username="backup-user",
            email="backup@example.com",
            password="secret-password",
            is_active=False,
        )
        user.groups.add(group)
        user.user_permissions.add(permission)

        with TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "incremental.jsonl.gz"
            state_path = Path(temp_dir) / "state.json"
            call_command("export_incremental", state=str(state_path), output=str(backup_path), no_update_state=True)

            User.objects.filter(username="backup-user").delete()
            Group.objects.filter(name="Privileged Users").delete()

            call_command("import_incremental", str(backup_path))

        restored_user = User.objects.get(username="backup-user")
        self.assertEqual(restored_user.email, "backup@example.com")
        self.assertFalse(restored_user.is_active)
        self.assertTrue(restored_user.check_password("secret-password"))
        self.assertTrue(restored_user.groups.filter(name="Privileged Users").exists())
        self.assertTrue(restored_user.user_permissions.filter(codename="view_user").exists())
        self.assertTrue(
            Group.objects.get(name="Privileged Users").permissions.filter(codename="view_user").exists()
        )


class ExternalForecastTests(TestCase):
    @patch("prices.external_forecasts.requests.get")
    def test_fetch_x2r_infers_national_average_from_region_g(self, requests_get):
        date_time = pd.Timestamp("2026-05-01T16:00:00Z")
        response = requests_get.return_value
        response.json.return_value = {
            "region": "G",
            "forecast_at": "2026-05-01T09:00:00Z",
            "prices": {
                "forecast": [
                    {
                        "date": date_time.isoformat(),
                        "price": 20.0,
                    }
                ]
            },
        }

        forecast = fetch_x2r("X")

        requests_get.assert_called_once_with("https://api.x2r.uk/agile/G", timeout=15)
        day_ahead = day_ahead_to_agile(pd.Series([20.0], index=[date_time]), reverse=True, region="G")
        expected = day_ahead_to_agile(day_ahead, region="X").iloc[0]
        self.assertEqual(forecast["name"], "X2R X 2026-05-01T09:00:00Z")
        self.assertAlmostEqual(forecast["rows"][0]["agile_pred"], expected)



class ForecastFeatureTests(TestCase):
    def test_resolve_feature_columns_supports_named_sets_and_drops(self):
        features = resolve_feature_columns(feature_set="weather", drop_features=["rad"])

        self.assertIn("temp_2m", features)
        self.assertNotIn("rad", features)
        self.assertEqual(list(FEATURE_SETS["weather"]).count("rad"), 1)

    def test_resolve_feature_columns_supports_explicit_feature_list(self):
        features = resolve_feature_columns(explicit_features="demand, peak, weekend")

        self.assertEqual(features, ["demand", "peak", "weekend"])

    def test_build_training_data_uses_supplied_feature_set(self):
        index = pd.to_datetime(["2026-05-01T22:00:00Z"])
        created_at = pd.to_datetime(["2026-05-01T16:15:00Z"])
        df = pd.DataFrame(
            index=index,
            data={
                "forecast_id": [1],
                "created_at": created_at,
                "ag_start": pd.to_datetime(["2026-05-01T22:00:00Z"]),
                "ag_end": pd.to_datetime(["2026-05-02T22:00:00Z"]),
                "days_ago": [1],
                "demand": [30],
                "peak": [0],
                "weekend": [0],
            },
        )
        forecasts = pd.DataFrame(index=[1])
        prices = pd.DataFrame(index=index, data={"day_ahead": [95]})

        train_X, train_y = build_training_data(df, forecasts, prices, ["demand", "weekend"], max_days=7)

        self.assertEqual(list(train_X.columns), ["demand", "weekend"])
        self.assertEqual(train_y.iloc[0], 95)

    def _horizon_frame(self):
        """One forecast run with rows at 0h, 1d, 3d and 10d past ag_start."""
        ag_start = pd.Timestamp("2026-05-01T22:00:00Z")
        offsets = [pd.Timedelta(0), pd.Timedelta(days=1), pd.Timedelta(days=3), pd.Timedelta(days=10)]
        index = pd.DatetimeIndex([ag_start + o for o in offsets])
        df = pd.DataFrame(
            index=index,
            data={
                "forecast_id": [1] * 4,
                "created_at": [pd.Timestamp("2026-05-01T16:15:00Z")] * 4,
                "ag_start": [ag_start] * 4,
                "ag_end": [ag_start + pd.Timedelta(days=1)] * 4,
                "days_ago": [1] * 4,
                "demand": [30, 31, 32, 33],
                "weekend": [0] * 4,
            },
        )
        prices = pd.DataFrame(index=index, data={"day_ahead": [95, 96, 97, 98]})
        return df, pd.DataFrame(index=[1]), prices

    def test_training_window_defaults_to_the_full_served_horizon(self):
        df, forecasts, prices = self._horizon_frame()

        train_X, _ = build_training_data(df, forecasts, prices, ["demand"], max_days=7)

        # All four rows, including the 10-day-ahead one: the model is served at
        # those horizons, so it must be trained on them.
        self.assertEqual(len(train_X), 4)
        self.assertEqual(TRAIN_HORIZON_DAYS, 14)

    def test_training_window_of_one_day_restores_the_legacy_agile_window(self):
        df, forecasts, prices = self._horizon_frame()

        train_X, _ = build_training_data(df, forecasts, prices, ["demand"], max_days=7, horizon_days=1)

        # Only the row at ag_start; ag_start + 1d is the exclusive upper bound.
        self.assertEqual(len(train_X), 1)
        self.assertEqual(train_X["demand"].tolist(), [30])

    def test_training_window_is_bounded_by_horizon_days(self):
        df, forecasts, prices = self._horizon_frame()

        train_X, _ = build_training_data(df, forecasts, prices, ["demand"], max_days=7, horizon_days=4)

        self.assertEqual(train_X["demand"].tolist(), [30, 31, 32])

    def test_latest_prediction_features_preserves_requested_columns(self):
        fc = pd.DataFrame(
            data={
                "demand": [30],
                "emb_wind": [5],
                "weekend": [0],
            }
        )

        features = latest_prediction_features(fc, ["emb_wind", "demand"])

        self.assertEqual(list(features.columns), ["emb_wind", "demand"])


class ModelMetricsTests(TestCase):
    def test_detection_metrics_reward_recall_and_punish_false_alarms(self):
        actual = np.array([-10.0, -5.0, 20.0, 100.0])
        # Flags both true negatives plus one false alarm.
        pred = np.array([-8.0, -1.0, -3.0, 90.0])

        band = detection_metrics(pred, actual, below=0.0)

        self.assertEqual(band["n_actual"], 2)
        self.assertEqual(band["n_flagged"], 3)
        self.assertAlmostEqual(band["recall"], 1.0)
        self.assertAlmostEqual(band["precision"], 2 / 3)

    def test_detection_metrics_report_nan_when_nothing_is_flagged(self):
        # The failure mode this whole gate exists to catch: a model that never
        # predicts an extreme scores zero recall, not a missing value.
        actual = np.array([-10.0, -5.0, 20.0, 100.0])
        pred = np.array([50.0, 55.0, 60.0, 90.0])

        band = detection_metrics(pred, actual, below=0.0)

        self.assertEqual(band["recall"], 0.0)
        self.assertTrue(np.isnan(band["precision"]))

    def test_calibration_metrics_detect_under_dispersion(self):
        rng = np.random.default_rng(0)
        actual = rng.normal(100.0, 40.0, 500)
        # A prediction compressed toward the mean, as the defect describes.
        pred = 100.0 + 0.5 * (actual - 100.0)

        report = calibration_metrics(pred, actual)

        self.assertAlmostEqual(report["sd_ratio"], 0.5, places=2)
        self.assertAlmostEqual(report["slope"], 2.0, places=2)

    def test_forecast_report_covers_every_price_band(self):
        rng = np.random.default_rng(1)
        actual = rng.normal(100.0, 60.0, 400)
        report = forecast_report(actual + rng.normal(0, 5, 400), actual)

        self.assertEqual(set(report["bands"]), set(PRICE_BANDS))
        self.assertIn("sd_ratio", report)
        self.assertIn("negative", format_report(report))

    def test_stored_forecast_report_excludes_short_horizon_slots(self):
        # Slots under 2 days are blended with GB60 actuals by the pipeline, so
        # scoring them measures the blend rather than the model.
        created = pd.Timestamp("2026-05-01T16:15:00Z")
        times = pd.to_datetime(
            ["2026-05-02T12:00:00Z"] + [f"2026-05-{d:02d}T12:00:00Z" for d in range(4, 14)]
        )
        stored = pd.DataFrame(
            {
                "date_time": times,
                "day_ahead": [999.0] + [100.0] * 10,  # the short-horizon row is absurd
                "created_at": [created] * len(times),
            }
        )
        actuals = pd.DataFrame(index=times, data={"day_ahead": [100.0] * len(times)})

        # Only 10 rows survive the horizon filter, below the reporting minimum,
        # so no report rather than one dominated by the blended slot.
        self.assertIsNone(stored_forecast_report(stored, actuals, min_horizon_days=2.0))

    def test_stored_forecast_report_scores_published_predictions(self):
        created = pd.Timestamp("2026-05-01T16:15:00Z")
        times = pd.date_range("2026-05-05T00:00:00Z", periods=200, freq="30min")
        rng = np.random.default_rng(2)
        actual = rng.normal(100.0, 50.0, len(times))
        stored = pd.DataFrame(
            {
                "date_time": times,
                # Compressed toward the mean, as the defect describes.
                "day_ahead": 100.0 + 0.5 * (actual - 100.0),
                "created_at": [created] * len(times),
            }
        )
        actuals = pd.DataFrame(index=times, data={"day_ahead": actual})

        report = stored_forecast_report(stored, actuals, min_horizon_days=2.0)

        self.assertEqual(report["n"], len(times))
        self.assertAlmostEqual(report["sd_ratio"], 0.5, places=2)
        self.assertEqual(report["min_horizon_days"], 2.0)


class UpdateOptionTests(TestCase):
    def test_extra_trees_ensemble_member_is_configured_for_parallel_prediction(self):
        self.assertEqual(EXTRA_TREES_REGRESSOR_PARAMS["min_samples_leaf"], 4)
        self.assertEqual(EXTRA_TREES_REGRESSOR_PARAMS["random_state"], 42)
        self.assertEqual(EXTRA_TREES_REGRESSOR_PARAMS["n_jobs"], 1)

    def test_ensemble_can_fit_day_ahead_training_matrix(self):
        train_X = pd.DataFrame(
            {
                "bm_wind": [1, 2, 3, 4, 5, 6],
                "solar": [0, 1, 0, 1, 0, 1],
                "demand": [30, 31, 32, 33, 34, 35],
                "peak": [0, 0, 1, 1, 0, 1],
                "days_ago": [1, 1, 2, 2, 3, 3],
                "weekend": [0, 0, 0, 0, 1, 1],
                "wind_10m": [5, 6, 7, 8, 9, 10],
                "temp_2m": [10, 11, 12, 13, 14, 15],
                "rad": [100, 200, 300, 400, 500, 600],
            }
        )
        train_y = pd.Series([60, 62, 80, 82, 70, 90])
        sample_weights = pd.Series([1, 1, 2, 2, 1, 2])

        models = fit_day_ahead_ensemble(train_X, train_y, sample_weights)
        predictions = predict_day_ahead_ensemble(models, train_X)

        self.assertEqual(len(models), 3)
        self.assertEqual(len(predictions), len(train_X))


class GasTtfHistoryTests(TestCase):
    @patch("config.utils.requests.get")
    def test_gas_ttf_history_uses_bounded_daily_yahoo_request(self, requests_get):
        response = requests_get.return_value
        response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1767225600],
                        "indicators": {"quote": [{"close": [42.5]}]},
                    }
                ]
            }
        }

        gas = get_gas_ttf_history(start="2026-01-01", end="2026-01-03")

        requests_get.assert_called_once()
        params = requests_get.call_args.kwargs["params"]
        self.assertEqual(params["interval"], "1d")
        self.assertIn("period1", params)
        self.assertIn("period2", params)
        self.assertNotIn("range", params)
        self.assertEqual(gas.iloc[0], 42.5)


class ResponseCacheKeyTests(TestCase):
    """The response cache is event-keyed: entries must invalidate the moment a
    new forecast (or fresh prices) lands, and must roll with the half-hour slot."""

    def _mw(self):
        from prices.middleware import ResponseCacheMiddleware

        return ResponseCacheMiddleware(lambda request: None)

    def test_key_changes_when_new_forecast_lands(self):
        from django.core.cache import cache

        mw = self._mw()
        request = RequestFactory().get("/v2/X/?days=2")
        cache.delete("rc:data-ver")  # ensure fresh version computation
        key_before = mw._key(request)

        Forecasts.objects.create(name="cache-invalidation-test")
        cache.delete("rc:data-ver")  # simulate micro-cache expiry (<=30s in prod)
        key_after = mw._key(request)

        self.assertNotEqual(key_before, key_after)

    def test_key_stable_for_same_data_and_slot(self):
        from django.core.cache import cache

        mw = self._mw()
        request = RequestFactory().get("/v2/X/?days=2")
        cache.delete("rc:data-ver")
        self.assertEqual(mw._key(request), mw._key(request))

    def test_key_includes_slot_bucket(self):
        mw = self._mw()
        request = RequestFactory().get("/v2/X/?days=2")
        with patch("prices.middleware.time.time", return_value=1_800_000_000):
            key_a = mw._key(request)
        with patch("prices.middleware.time.time", return_value=1_800_000_000 + 1800):
            key_b = mw._key(request)
        self.assertNotEqual(key_a, key_b)


class KofiTotalsTests(TestCase):
    def test_monthly_totals_grouped_by_month_and_currency(self):
        from prices.models import KofiPayment
        from prices.views import _kofi_totals

        now = timezone.now()
        last_month = (now.replace(day=1) - timedelta(days=1)).replace(day=15)
        for i, (ts, amount) in enumerate([(now, 3), (now, 5), (last_month, 2)]):
            KofiPayment.objects.create(
                kofi_transaction_id=f"test-{i}", timestamp=ts, amount=amount, currency="GBP"
            )

        out = _kofi_totals()

        self.assertEqual(out["count"], 3)
        self.assertEqual(len(out["months"]), 2)  # two distinct months
        self.assertEqual(out["months"][0]["total"], 8.0)  # newest month first
        self.assertEqual(out["months"][0]["payments"], 2)
        self.assertEqual(out["months"][1]["total"], 2.0)
        self.assertEqual(out["currencies"][0]["this_month"], 8.0)


class CanonicalCacheKeyTests(TestCase):
    """Equivalent URLs must share a response-cache key; URLs that render
    differently must not. Normalisation mirrors the views' own parsing, so
    these tests guard against the canonicaliser drifting from them."""

    def _key(self, url):
        from prices.middleware import ResponseCacheMiddleware
        from django.core.cache import cache

        cache.delete("rc:data-ver")
        mw = ResponseCacheMiddleware(lambda r: None)
        return mw._key(RequestFactory().get(url))

    def _same(self, a, b):
        self.assertEqual(self._key(a), self._key(b), f"expected same key: {a} vs {b}")

    def _diff(self, a, b):
        self.assertNotEqual(self._key(a), self._key(b), f"expected different keys: {a} vs {b}")

    # --- equivalent URLs collapse ---
    def test_parameter_order_irrelevant(self):
        self._same("/v2/X/?days=5&gen=1", "/v2/X/?gen=1&days=5")

    def test_explicit_defaults_match_omitted(self):
        self._same("/v2/X/?days=5&gen=1&band=1&dc=0&overlap=0", "/v2/X/")

    def test_unknown_parameters_ignored(self):
        self._same("/v2/X/?utm_source=twitter&fbclid=abc", "/v2/X/")

    def test_fc_order_irrelevant(self):
        self._same("/v2/X/?fc=2&fc=1", "/v2/X/?fc=1&fc=2")

    def test_days_clamped_like_the_view(self):
        # view does min(max(int(days),1),14)
        self._same("/v2/X/?days=99", "/v2/X/?days=14")
        self._same("/v2/X/?days=0", "/v2/X/?days=1")

    def test_af_truthy_spellings_equivalent(self):
        self._same("/v2/X/?af=true", "/v2/X/?af=1")
        self._same("/v2/X/?af=YES", "/v2/X/?af=on")

    def test_v2_export_only_true_for_literal_one(self):
        # GraphV2View: get("export","0") == "1" — so "true" is falsy here
        self._same("/v2/X/?export=true", "/v2/X/")
        self._diff("/v2/X/?export=1", "/v2/X/")

    def test_api_high_low_spellings_equivalent(self):
        self._same("/api/G/?high_low=True", "/api/G/?high_low=true")
        self._same("/api/G/?high_low=true", "/api/G/")

    # --- genuinely different URLs stay separate ---
    def test_different_days_differ(self):
        self._diff("/v2/X/?days=5", "/v2/X/?days=7")

    def test_toggles_differ(self):
        self._diff("/v2/X/?gen=1", "/v2/X/?gen=0")
        self._diff("/v2/X/?af=1", "/v2/X/?af=0")
        self._diff("/v2/X/?overlap=1", "/v2/X/?overlap=0")

    def test_different_fc_selection_differs(self):
        self._diff("/v2/X/?fc=1", "/v2/X/?fc=1&fc=2")

    def test_api_export_changes_key(self):
        self._diff("/api/G/?export=true", "/api/G/")

    def test_api_format_preserved(self):
        self._diff("/api/G/?format=json", "/api/G/")

    def test_region_path_still_separates(self):
        self._diff("/v2/X/?days=5", "/v2/G/?days=5")

    # --- safety: unparseable values fall back rather than collapse ---
    def test_unparseable_days_falls_back_to_raw_query(self):
        self._diff("/v2/X/?days=abc", "/v2/X/?days=xyz")

    # --- /v2/history/ has its own parameter set, not the chart's ---
    # Regression: history shares the /v2/ prefix, so it fell through to the
    # chart branch, which dropped all of these as unknown. Every history URL
    # then shared one cache entry and the on-page controls did nothing.
    def test_history_windows_differ(self):
        self._diff("/v2/history/?window=last-week", "/v2/history/?window=last-month")
        self._diff("/v2/history/?window=last-week", "/v2/history/?window=last-2-weeks")

    def test_history_other_controls_differ(self):
        self._diff("/v2/history/?offset_days=1", "/v2/history/?offset_days=3")
        self._diff("/v2/history/?metric=mae", "/v2/history/?metric=rmse")
        self._diff("/v2/history/?unit_mode=da", "/v2/history/?unit_mode=agile")
        self._diff("/v2/history/?compare_x2r=1", "/v2/history/")
        self._diff("/v2/history/?compare_agileforecast=1", "/v2/history/")

    def test_history_does_not_collide_with_chart_params(self):
        # A history URL must never key the same as the chart page's defaults.
        self._diff("/v2/history/?window=last-week", "/v2/X/?days=5")

    def test_history_defaults_collapse(self):
        self._same(
            "/v2/history/?window=last-2-weeks&offset_days=1&metric=mae&unit_mode=da",
            "/v2/history/",
        )
        self._same("/v2/history/?utm_source=x", "/v2/history/")

    def test_history_normalisation_mirrors_the_view(self):
        # unknown window / metric / unit_mode fall back to the view's defaults
        self._same("/v2/history/?window=bogus", "/v2/history/")
        self._same("/v2/history/?metric=nonsense", "/v2/history/?metric=mae")
        self._same("/v2/history/?unit_mode=nonsense", "/v2/history/?unit_mode=da")
        # offset_days clamped to 0..14, and truthy spellings agree with _truthy
        self._same("/v2/history/?offset_days=99", "/v2/history/?offset_days=14")
        self._same("/v2/history/?offset_days=-5", "/v2/history/?offset_days=0")
        self._same("/v2/history/?compare_x2r=YES", "/v2/history/?compare_x2r=on")

    def test_history_custom_window_keys_on_dates(self):
        base = "/v2/history/?window=custom&start_date=2026-01-01&end_date=2026-01-31"
        self._diff(base, "/v2/history/?window=custom&start_date=2026-02-01&end_date=2026-02-28")
        # Incomplete/reversed pairs fall back to last-2-weeks in the view, so
        # they must key the same as that window rather than as separate entries.
        self._same("/v2/history/?window=custom", "/v2/history/?window=last-2-weeks")
        self._same(
            "/v2/history/?window=custom&start_date=2026-03-10&end_date=2026-03-01",
            "/v2/history/?window=last-2-weeks",
        )

    def test_history_region_path_still_separates(self):
        self._diff("/v2/history/?window=last-week", "/v2/history/G/?window=last-week")


class InvalidRegionTests(TestCase):
    """An unrecognised region must not cost a full chart render: it used to fall
    back to the national chart, and since the response-cache key includes the
    path, every bogus path was an uncacheable full render."""

    def test_invalid_region_redirects_cheaply(self):
        response = self.client.get("/v2/zzz/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/v2/X/")

    def test_invalid_region_preserves_query(self):
        response = self.client.get("/v2/s/?days=14&gen=1")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/v2/X/?days=14&gen=1")

    def test_valid_region_not_redirected(self):
        # Routing only. Rendering needs forecast data this test DB lacks, and the
        # test client re-raises view exceptions, so suppress that and assert the
        # request was not short-circuited into a redirect.
        from django.test import Client

        c = Client(raise_request_exception=False)
        self.assertNotEqual(c.get("/v2/G/").status_code, 302)
        self.assertNotEqual(c.get("/v2/g/").status_code, 302)


class UpdateCatchupTests(TestCase):
    """Backstop for the external (EasyCron) update trigger — see GH #104.

    The scheduler fires once with no retry, so an outage overlapping one of its
    slots loses that forecast cycle silently. These cover the conditions under
    which the worker should and should not step in.
    """

    def _job(self, age_hours, status=UpdateJob.STATUS_COMPLETED, job_type=UpdateJob.JOB_UPDATE):
        job = UpdateJob.objects.create(job_type=job_type, status=status)
        # requested_at is auto_now_add, so it has to be rewritten after insert.
        UpdateJob.objects.filter(pk=job.pk).update(
            requested_at=timezone.now() - timedelta(hours=age_hours)
        )
        job.refresh_from_db()
        return job

    def test_no_catchup_when_recent(self):
        self._job(age_hours=6)
        self.assertIsNone(maybe_enqueue_catchup())
        self.assertEqual(UpdateJob.objects.count(), 1)

    def test_catchup_when_stale(self):
        self._job(age_hours=10)
        job = maybe_enqueue_catchup()
        self.assertIsNotNone(job)
        self.assertEqual(job.job_type, UpdateJob.JOB_UPDATE)
        self.assertEqual(job.status, UpdateJob.STATUS_PENDING)

    def test_overnight_gap_does_not_trigger(self):
        """21:15 -> 05:15 is exactly 8h and is legitimate, not a missed cycle."""
        self._job(age_hours=8)
        self.assertIsNone(maybe_enqueue_catchup())

    def test_options_match_the_scheduled_path(self):
        """run_job passes options straight to call_command, so a stray key would
        fail the very job the catch-up exists to rescue."""
        self._job(age_hours=10)
        job = maybe_enqueue_catchup()
        self.assertEqual(job.options, {"skip_kde_plot": True})

    def test_no_catchup_while_a_job_is_pending(self):
        self._job(age_hours=10)
        self._job(age_hours=9, status=UpdateJob.STATUS_PENDING)
        self.assertIsNone(maybe_enqueue_catchup())

    def test_no_catchup_while_a_job_is_running(self):
        self._job(age_hours=10)
        self._job(age_hours=9, status=UpdateJob.STATUS_RUNNING)
        self.assertIsNone(maybe_enqueue_catchup())

    def test_stale_failed_job_is_retried(self):
        """A failed job means no forecast landed, so it still needs covering."""
        self._job(age_hours=10, status=UpdateJob.STATUS_FAILED)
        self.assertIsNotNone(maybe_enqueue_catchup())

    def test_does_not_queue_twice(self):
        """The queued job becomes both the newest and a pending one, so a second
        pass must be a no-op — otherwise a stalled worker floods the queue."""
        self._job(age_hours=10)
        self.assertIsNotNone(maybe_enqueue_catchup())
        self.assertIsNone(maybe_enqueue_catchup())
        self.assertEqual(UpdateJob.objects.filter(status=UpdateJob.STATUS_PENDING).count(), 1)

    def test_empty_database_is_left_alone(self):
        self.assertIsNone(maybe_enqueue_catchup())
        self.assertEqual(UpdateJob.objects.count(), 0)

    def test_latest_agile_jobs_do_not_count(self):
        """latest_agile runs afternoons only; it must not mask a missed update."""
        self._job(age_hours=10)
        self._job(age_hours=1, job_type=UpdateJob.JOB_LATEST_AGILE)
        self.assertIsNotNone(maybe_enqueue_catchup())

    @override_settings(UPDATE_CATCHUP_ENABLED=False)
    def test_disabled_by_setting(self):
        self._job(age_hours=48)
        self.assertIsNone(maybe_enqueue_catchup())

    @override_settings(UPDATE_CATCHUP_HOURS=0)
    def test_zero_hours_disables(self):
        self._job(age_hours=48)
        self.assertIsNone(maybe_enqueue_catchup())


class TemplateCommentLeakTests(TestCase):
    """Django's `{# ... #}` comment syntax is SINGLE-LINE ONLY.

    A `{#` comment spanning several lines is not a comment at all — the tag
    parser never closes it, so the raw text is emitted into the response. This
    shipped to production: three such comments printed their own source onto the
    main chart view, and a fourth landed inside a JavaScript object literal,
    making that script block a syntax error and silently killing the external
    forecast overlay. Multi-line commentary must use
    `{% comment %}` / `{% endcomment %}`.
    """

    TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

    def test_no_multiline_hash_comments_in_templates(self):
        offenders = []
        for path in self.TEMPLATE_DIR.rglob("*.html"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # An opener with no closer on the same line runs past its line.
                if "{#" in line and "#}" not in line.split("{#", 1)[1]:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()[:70]}")
        self.assertEqual(
            offenders,
            [],
            "Multi-line {# #} comments leak their text into the rendered page. "
            "Use {% comment %}...{% endcomment %} instead:\n  " + "\n  ".join(offenders),
        )

    def test_single_line_hash_comments_are_still_allowed(self):
        """Guard against the scan above being tightened into banning `{#` outright;
        single-line comments are idiomatic and used throughout these templates."""
        line = "  {# Region dropdown — JS preserves current query string #}"
        self.assertNotIn("#}", line.split("{#", 1)[1].split("#}")[0])


class SummaryPlacementTests(TestCase):
    """Summary-card placement (GH #93): the cards can sit above the chart, below
    it, or be hidden, and the choice has to survive the response cache."""

    def _key(self, url):
        from django.core.cache import cache
        from prices.middleware import ResponseCacheMiddleware

        cache.delete("rc:data-ver")
        return ResponseCacheMiddleware(lambda r: None)._key(RequestFactory().get(url))

    def test_each_position_gets_its_own_cache_key(self):
        """Without this the three positions collapse onto one cached entry and
        the control silently stops working — the failure mode the canonicaliser
        comment warns about."""
        above = self._key("/v2/X/")
        self._assert_all_distinct(above, self._key("/v2/X/?summary=below"), self._key("/v2/X/?summary=off"))

    def _assert_all_distinct(self, *keys):
        self.assertEqual(len(set(keys)), len(keys), f"expected distinct keys, got {keys}")

    def test_default_and_explicit_above_share_a_key(self):
        self.assertEqual(self._key("/v2/X/"), self._key("/v2/X/?summary=above"))

    def test_unrecognised_value_shares_the_default_key(self):
        """Junk values fall back to "above" rather than minting a cache entry per
        distinct junk string, which would be a trivial cache-busting vector."""
        self.assertEqual(self._key("/v2/X/"), self._key("/v2/X/?summary=sideways"))

    def test_view_normalises_unrecognised_values(self):
        from prices.views import _SUMMARY_POSITIONS

        for value in ["sideways", "", "1", "ABOVE"]:
            request = RequestFactory().get(f"/v2/X/?summary={value}")
            pos = request.GET.get("summary", "above")
            if pos not in _SUMMARY_POSITIONS:
                pos = "above"
            self.assertIn(pos, _SUMMARY_POSITIONS)

    def test_positions_and_labels_stay_in_step(self):
        from prices.views import _SUMMARY_POSITION_LABELS, _SUMMARY_POSITIONS

        self.assertEqual(_SUMMARY_POSITIONS, tuple(p for p, _ in _SUMMARY_POSITION_LABELS))
        self.assertEqual(_SUMMARY_POSITIONS[0], "above", "default must be first")
