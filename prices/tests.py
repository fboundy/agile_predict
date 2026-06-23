from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
)
from prices.external_forecasts import fetch_x2r
from prices.forms import ForecastForm
from prices.management.commands.update import Command as UpdateCommand
from prices.management.commands.update import EXTRA_TREES_REGRESSOR_PARAMS, fit_day_ahead_ensemble, predict_day_ahead_ensemble
from prices.models import AgileData, ExternalForecast, ForecastData, Forecasts, PriceHistory, RequestClientSeen, RequestMetric
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


class RequestMetricsTests(TestCase):
    def test_request_metric_records_web_and_unique_client(self):
        response = self.client.get("/about", HTTP_USER_AGENT="metrics-test")

        self.assertEqual(response.status_code, 200)
        metric = RequestMetric.objects.get(surface=RequestMetric.SURFACE_WEB, path="/about")
        self.assertEqual(metric.request_count, 1)
        self.assertEqual(RequestClientSeen.objects.filter(surface=RequestMetric.SURFACE_WEB).count(), 1)

    def test_repeated_same_client_increments_requests_without_new_unique_client(self):
        for _ in range(2):
            self.client.get("/about", HTTP_USER_AGENT="same-client")

        metric = RequestMetric.objects.get(surface=RequestMetric.SURFACE_WEB, path="/about")
        self.assertEqual(metric.request_count, 2)
        self.assertEqual(RequestClientSeen.objects.filter(surface=RequestMetric.SURFACE_WEB).count(), 1)

    def test_metrics_page_requires_staff_login(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response["Location"])

    def test_metrics_page_renders_for_staff_user(self):
        user = User.objects.create_user(username="staff", password="pw", is_staff=True)
        self.client.force_login(user)

        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Daily Requests")


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
