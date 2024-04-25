from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from prices.models import PriceHistory, Forecasts
from .serializers import PriceHistorySerializer, PriceForecastSerializer


class PriceHistoryAPIView(generics.ListAPIView):
    queryset = PriceHistory.objects.all()
    serializer_class = PriceHistorySerializer


class PriceForecastAPIView(generics.ListAPIView):
    # queryset = Forecasts.objects.all()
    queryset = [Forecasts.objects.latest("created_at")]
    serializer_class = PriceForecastSerializer
