from django.urls import path
from .views import PriceHistoryAPIView, PriceForecastAPIView

urlpatterns = [
    path("hist", PriceHistoryAPIView.as_view()),
    path("", PriceForecastAPIView.as_view()),
]
