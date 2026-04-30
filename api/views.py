import logging

from django.shortcuts import render
import pandas as pd


# Create your views here.
from rest_framework import generics
from prices.models import Forecasts
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
