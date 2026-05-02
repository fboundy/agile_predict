import logging
import math

from django.shortcuts import render
from django.utils import timezone
import pandas as pd


# Create your views here.
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from config.utils import day_ahead_to_agile, regions
from prices.models import AgileData, Forecasts, PriceHistory
from .serializers import PriceForecastSerializer, PriceForecastRegionSerializer


logger = logging.getLogger("prices.api")


# class PriceForecastAPIView(generics.ListAPIView):
#     ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:1]]


#     queryset = Forecasts.objects.filter(id__in=ids)
#     # queryset = Forecasts.objects.all()
#     serializer_class = PriceForecastSerializer
class PriceForecastAPIView(generics.ListAPIView):
    serializer_class = PriceForecastSerializer

    def get_queryset(self):
        latest = Forecasts.objects.order_by("-created_at")[:1]
        return latest


class PriceForecastRegionAPIView(generics.ListAPIView):
    serializer_class = PriceForecastRegionSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        context.update({"region": self.kwargs["region"].upper()})
        context.update({"days": int(self.request.query_params.get("days", 14))})
        context.update({"high_low": self.request.query_params.get("high_low", "true").lower() in ["true", "1"]})
        return context

    def get_queryset(self):
        forecast_count = int(self.request.query_params.get("forecast_count", 1))
        region = self.kwargs["region"].upper()
        logger.debug("API forecast request region=%s forecast_count=%s", region, forecast_count)
        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:forecast_count]]
        queryset = Forecasts.objects.filter(id__in=ids).order_by("-created_at")
        return queryset


class AccuracyAPIView(APIView):
    buckets = [
        ("0\u201324h", 0, 24),
        ("24\u201348h", 24, 48),
        ("48\u201372h", 48, 72),
        ("3\u20137d", 72, 168),
        ("7d+", 168, None),
    ]

    def get(self, request, *args, **kwargs):
        region = request.query_params.get("region", "X").upper()
        if region not in regions:
            region = "X"

        data_from = timezone.now() - pd.Timedelta(days=30)
        bucket_values = {label: [] for label, _, _ in self.buckets}
        date_times = []

        forecasts = (
            AgileData.objects.filter(region=region, date_time__gte=data_from)
            .select_related("forecast")
            .order_by("date_time")
        )
        actual_rows = list(PriceHistory.objects.filter(
            date_time__in=forecasts.values("date_time")
        ).values_list("date_time", "day_ahead"))
        if actual_rows:
            day_ahead = pd.Series(data=[row[1] for row in actual_rows], index=[row[0] for row in actual_rows])
            actual_by_date_time = day_ahead_to_agile(day_ahead, region=region).to_dict()
        else:
            actual_by_date_time = {}

        for forecast in forecasts:
            actual = actual_by_date_time.get(forecast.date_time)
            if actual is None:
                continue

            lead_hours = (forecast.date_time - forecast.forecast.created_at).total_seconds() / 3600
            if lead_hours < 0:
                continue

            for label, start_hour, end_hour in self.buckets:
                if start_hour <= lead_hours and (end_hour is None or lead_hours < end_hour):
                    bucket_values[label].append(forecast.agile_pred - actual)
                    date_times.append(forecast.date_time)
                    break

        buckets = []
        for label, _, _ in self.buckets:
            errors = bucket_values[label]
            n = len(errors)
            if n:
                mae = sum(abs(error) for error in errors) / n
                rmse = math.sqrt(sum(error * error for error in errors) / n)
                bias = sum(errors) / n
            else:
                mae = rmse = bias = None

            buckets.append(
                {
                    "label": label,
                    "n": n,
                    "mae": round(mae, 2) if mae is not None else None,
                    "rmse": round(rmse, 2) if rmse is not None else None,
                    "bias": round(bias, 2) if bias is not None else None,
                }
            )

        return Response(
            {
                "computed_at": timezone.now().isoformat().replace("+00:00", "Z"),
                "data_from": min(date_times).isoformat() if date_times else None,
                "data_to": max(date_times).isoformat() if date_times else None,
                "total_pairs": sum(bucket["n"] for bucket in buckets),
                "regions_covered": [region] if date_times else [],
                "buckets": buckets,
            }
        )
