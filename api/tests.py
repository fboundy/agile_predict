from datetime import datetime, timedelta, timezone as datetime_timezone

from django.test import TestCase
from django.utils import timezone

from prices.models import AgileData, Forecasts, PriceHistory


class AccuracyAPITests(TestCase):
    def create_forecast(self, created_at):
        forecast = Forecasts.objects.create(name=f"forecast-{created_at.isoformat()}", mean=0, stdev=0)
        Forecasts.objects.filter(pk=forecast.pk).update(created_at=created_at)
        forecast.refresh_from_db()
        return forecast

    def add_pair(self, forecast, date_time, region, actual_day_ahead, predicted):
        PriceHistory.objects.get_or_create(
            date_time=date_time,
            defaults={"agile": actual_day_ahead, "day_ahead": actual_day_ahead},
        )
        AgileData.objects.create(
            forecast=forecast,
            region=region,
            date_time=date_time,
            agile_pred=predicted,
            agile_low=predicted,
            agile_high=predicted,
        )

    def test_accuracy_defaults_to_region_x_and_buckets_by_lead_time(self):
        created_at = datetime(2026, 5, 1, tzinfo=datetime_timezone.utc)
        forecast = self.create_forecast(created_at)

        self.add_pair(forecast, created_at + timedelta(hours=1), "X", actual_day_ahead=0, predicted=2)
        self.add_pair(forecast, created_at + timedelta(hours=24), "X", actual_day_ahead=0, predicted=-4)
        self.add_pair(forecast, created_at + timedelta(hours=48), "X", actual_day_ahead=0, predicted=3)
        self.add_pair(forecast, created_at + timedelta(hours=72), "X", actual_day_ahead=0, predicted=-5)
        self.add_pair(forecast, created_at + timedelta(days=7), "X", actual_day_ahead=0, predicted=6)
        self.add_pair(forecast, created_at + timedelta(hours=1), "A", actual_day_ahead=0, predicted=30)

        response = self.client.get("/api/accuracy/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_pairs"], 5)
        self.assertEqual(payload["regions_covered"], ["X"])
        self.assertEqual(
            payload["buckets"],
            [
                {"label": "0\u201324h", "n": 1, "mae": 2.0, "rmse": 2.0, "bias": 2.0},
                {"label": "24\u201348h", "n": 1, "mae": 4.0, "rmse": 4.0, "bias": -4.0},
                {"label": "48\u201372h", "n": 1, "mae": 3.0, "rmse": 3.0, "bias": 3.0},
                {"label": "3\u20137d", "n": 1, "mae": 5.0, "rmse": 5.0, "bias": -5.0},
                {"label": "7d+", "n": 1, "mae": 6.0, "rmse": 6.0, "bias": 6.0},
            ],
        )

    def test_accuracy_accepts_region_query_param(self):
        created_at = datetime(2026, 5, 1, tzinfo=datetime_timezone.utc)
        forecast = self.create_forecast(created_at)
        self.add_pair(forecast, created_at + timedelta(hours=1), "A", actual_day_ahead=0, predicted=15)

        response = self.client.get("/api/accuracy/?region=A")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_pairs"], 1)
        self.assertEqual(payload["regions_covered"], ["A"])

    def test_accuracy_only_includes_last_30_days_by_date_time(self):
        now = timezone.now()
        recent_forecast = self.create_forecast(now - timedelta(days=1))
        old_forecast = self.create_forecast(now - timedelta(days=40))

        self.add_pair(recent_forecast, now - timedelta(hours=12), "X", actual_day_ahead=0, predicted=2)
        self.add_pair(old_forecast, now - timedelta(days=31), "X", actual_day_ahead=0, predicted=20)

        response = self.client.get("/api/accuracy/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_pairs"], 1)
        self.assertEqual(payload["buckets"][0], {"label": "0\u201324h", "n": 1, "mae": 2.0, "rmse": 2.0, "bias": 2.0})

    def test_region_forecast_api_accepts_export_parameter(self):
        created_at = datetime(2026, 5, 1, tzinfo=datetime_timezone.utc)
        forecast = self.create_forecast(created_at)
        AgileData.objects.create(
            forecast=forecast,
            region="A",
            date_time=created_at + timedelta(hours=12),
            agile_pred=21,
            agile_low=20,
            agile_high=24,
        )

        response = self.client.get("/api/A/?export=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertAlmostEqual(payload[0]["prices"][0]["agile_pred"], 10.59)
