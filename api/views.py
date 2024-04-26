from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from prices.models import Forecasts
from .serializers import PriceForecastSerializer, PriceForecastRegionSerializer, HA_PriceForecastRegionSerializer


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
        region = self.kwargs["region"]

        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

        queryset = Forecasts.objects.filter(id__in=ids)
        return queryset


class HA_PriceForecastRegionAPIView(generics.ListAPIView):
    serializer_class = HA_PriceForecastRegionSerializer

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
        region = self.kwargs["region"]

        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

        queryset = Forecasts.objects.filter(id__in=ids)
        return queryset
