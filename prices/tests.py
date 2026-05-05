from datetime import timedelta
from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase, override_settings
from django.utils import timezone

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile
from prices.forecast_features import (
    build_training_data,
    FEATURE_SETS,
    latest_prediction_features,
    resolve_feature_columns,
)
from prices.forms import ForecastForm
from prices.management.commands.update import Command as UpdateCommand
from prices.management.commands.update import XGBOOST_PARAMETER_SETS
from prices.models import AgileData, ExternalForecast, Forecasts, PriceHistory, RequestClientSeen, RequestMetric
from prices.views import GraphFormView, _update_options


class HistoryViewTests(TestCase):
    def test_history_view_renders_for_region_and_offset(self):
        response = self.client.get("/history/X/?offset_days=2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction Offset")
        self.assertContains(response, "2d ahead")
        self.assertContains(response, "Date Window")
        self.assertContains(response, "Last 2 Weeks")
        self.assertContains(response, 'type="date"')

    def test_history_view_marks_selected_region(self):
        response = self.client.get("/history/G/?offset_days=2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'action="/history/G/"')
        self.assertContains(response, '<option value="G" selected>G - North Western England</option>', html=True)

    def test_history_prediction_lines_use_successive_time_slot_runs(self):
        created_at = timezone.now() - timedelta(hours=6)
        forecast = Forecasts.objects.create(name="history-run-test", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()

        for index, offset_minutes in enumerate([0, 30, 90]):
            AgileData.objects.create(
                forecast=forecast,
                region="X",
                date_time=created_at + timedelta(hours=1, minutes=offset_minutes),
                agile_pred=index,
                agile_low=index,
                agile_high=index,
            )

        response = self.client.get("/history/X/?offset_days=0")

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
            AgileData.objects.create(
                forecast=forecast,
                region="X",
                date_time=date_time,
                agile_pred=index,
                agile_low=index,
                agile_high=index,
            )

        response = self.client.get("/history/X/?offset_days=0")

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
            day_ahead = day_ahead_to_agile(pd.Series([0], index=[date_time]), reverse=True, region="X").iloc[0]
            PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=day_ahead)
            AgileData.objects.create(
                forecast=forecast,
                region="X",
                date_time=date_time,
                agile_pred=predicted,
                agile_low=predicted,
                agile_high=predicted,
            )

        response = self.client.get("/history/X/?offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<th scope=\"col\" class=\"text-start\">Metric</th>")
        self.assertContains(response, "&lt;1d")
        self.assertContains(response, "1d")
        self.assertContains(response, "Offset")
        self.assertContains(response, "+1.00")
        self.assertContains(response, "-2.00")

    def test_history_region_g_offers_external_comparison_for_staff(self):
        user = User.objects.create_user(username="history-staff", password="pw", is_staff=True)
        self.client.force_login(user)
        created_at = timezone.now() - timedelta(hours=6)
        ExternalForecast.objects.create(
            source=ExternalForecast.SOURCE_X2R,
            region="G",
            forecast_name="x2r test",
            source_created_at=created_at,
            date_time=created_at + timedelta(hours=1),
            agile_pred=12,
        )

        response = self.client.get("/history/G/?compare_x2r=1")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Compare AgileForecast")
        self.assertContains(response, "Compare X2R")
        self.assertContains(response, "X2R comparison predictions")

    def test_history_region_g_metrics_table_includes_selected_external_forecasts_for_staff(self):
        user = User.objects.create_user(username="history-metrics-staff", password="pw", is_staff=True)
        self.client.force_login(user)
        created_at = timezone.now() - timedelta(hours=6)

        for offset_minutes, actual_price, predicted_price in [(60, 10, 12), (90, 10, 14)]:
            date_time = created_at + timedelta(minutes=offset_minutes)
            day_ahead = day_ahead_to_agile(pd.Series([actual_price], index=[date_time]), reverse=True, region="G").iloc[0]
            PriceHistory.objects.create(date_time=date_time, agile=actual_price, day_ahead=day_ahead)
            ExternalForecast.objects.create(
                source=ExternalForecast.SOURCE_X2R,
                region="G",
                forecast_name="x2r metrics test",
                source_created_at=created_at,
                date_time=date_time,
                agile_pred=predicted_price,
            )

        response = self.client.get("/history/G/?compare_x2r=1&offset_days=0")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "X2R MAE")
        self.assertContains(response, "X2R RMSE")
        self.assertContains(response, "X2R Bias")
        self.assertContains(response, "3.00")
        self.assertContains(response, "3.16")
        self.assertContains(response, "+3.00")

    @override_settings(LOCAL_REALTIME_EXTERNAL_FORECASTS=False)
    def test_history_region_g_does_not_offer_external_comparison_to_anonymous_user(self):
        response = self.client.get("/history/G/")

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Compare AgileForecast")
        self.assertNotContains(response, "Compare X2R")


class ExportPricingTests(TestCase):
    def test_national_export_coefficients_are_arithmetic_mean(self):
        regional_factors = [
            GLOBAL_SETTINGS["REGIONS"][region]["export_factors"]
            for region in GLOBAL_SETTINGS["REGIONS"]
            if region != "X"
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

    @override_settings(LOCAL_REALTIME_EXTERNAL_FORECASTS=True)
    @patch("prices.views.fetch_agileforecast")
    def test_view_fetches_live_external_forecasts_when_enabled(self, fetch_agileforecast):
        fetch_agileforecast.return_value = {
            "name": "Region | G test",
            "source_created_at": timezone.now(),
            "rows": [],
        }
        view = GraphFormView()

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
    def test_update_options_include_xgboost_params(self):
        request = RequestFactory().get("/update", {"xgboost_params": "regularized_dart"})

        options = _update_options(request)

        self.assertEqual(options["xgboost_params"], "regularized_dart")

    def test_xgboost_parameter_sets_include_current_default_and_alternatives(self):
        self.assertEqual(
            set(XGBOOST_PARAMETER_SETS),
            {"current_dart", "conservative_gbtree", "shallow_regularized", "regularized_dart"},
        )
        self.assertEqual(XGBOOST_PARAMETER_SETS["current_dart"]["booster"], "dart")

    def test_update_command_defaults_to_regularized_dart(self):
        parser = UpdateCommand().create_parser("manage.py", "update")

        options = parser.parse_args([])

        self.assertEqual(options.xgboost_params, "regularized_dart")
