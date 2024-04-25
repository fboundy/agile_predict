from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from prices.models import PriceHistory, Forecasts
from .serializers import PriceHistorySerializer, PriceForecastSerializer


class PriceHistoryAPIView(generics.ListAPIView):
    queryset = PriceHistory.objects.all()
    serializer_class = PriceHistorySerializer


class PriceForecastAPIView(generics.ListAPIView):
    ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

    queryset = Forecasts.objects.filter(id__in=ids)
    serializer_class = PriceForecastSerializer
