from django.shortcuts import render
import pandas as pd


# Create your views here.
from rest_framework import generics
from prices.models import Forecasts
from .serializers import PriceForecastSerializer, PriceForecastRegionSerializer
from django.core.management import call_command

# class PriceHistoryAPIView(generics.ListAPIView):
#     queryset = PriceHistory.objects.all()
#     serializer_class = PriceHistorySerializer


class PriceForecastAPIView(generics.ListAPIView):
    ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

    queryset = Forecasts.objects.filter(id__in=ids)
    serializer_class = PriceForecastSerializer


class PriceForecastRegionAPIView(generics.ListAPIView):
    serializer_class = PriceForecastRegionSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        context.update({"region": self.kwargs["region"]})
        return context

    def get_queryset(self):
        """
        This view should return a list of all the purchases for
        the user as determined by the username portion of the URL.
        """
        f = Forecasts.objects.latest("created_at")

        # hour_now = pd.Timestamp.now(tz="GB").hour
        # hour_updated = pd.Timestamp(f.created_at).hour
        # updated_today = pd.Timestamp(f.created_at).day == pd.Timestamp.now(tz="GB").day

        # if (hour_now >= 10 and not updated_today) or (hour_now >= 16 and hour_updated < 16):
        if (pd.Timestamp.now(tz="GB") - f.created_at).total_seconds() / 3600 > 1:
            call_command("update")

        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

        queryset = Forecasts.objects.filter(id__in=ids).order_by("-created_at")
        return queryset
