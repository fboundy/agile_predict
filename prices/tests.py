from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from prices.models import AgileData, Forecasts, PriceHistory


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
            PriceHistory.objects.create(date_time=date_time, agile=0, day_ahead=0)
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
