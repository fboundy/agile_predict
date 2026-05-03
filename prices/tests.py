from datetime import timedelta

import pandas as pd
from django.test import TestCase
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
from prices.models import AgileData, ExternalForecast, Forecasts, PriceHistory


class HistoryViewTests(TestCase):
    def test_history_view_renders_for_region_and_offset(self):
        response = self.client.get("/history/X/?offset_days=2")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction Offset")
        self.assertContains(response, "2d ahead")
        self.assertContains(response, "Date Window")
        self.assertContains(response, "Last 2 Weeks")
        self.assertContains(response, 'type="date"')

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

    def test_history_gx_offers_external_comparison_without_dropdown_region(self):
        created_at = timezone.now() - timedelta(hours=6)
        ExternalForecast.objects.create(
            source=ExternalForecast.SOURCE_X2R,
            region="G",
            forecast_name="x2r test",
            source_created_at=created_at,
            date_time=created_at + timedelta(hours=1),
            agile_pred=12,
        )

        response = self.client.get("/history/Gx/?compare_x2r=1")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Compare AgileForecast")
        self.assertContains(response, "Compare X2R")
        self.assertContains(response, "X2R comparison predictions")
        self.assertNotContains(response, 'value="GX"')

    def test_history_regular_region_does_not_offer_external_comparison(self):
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
