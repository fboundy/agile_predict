from django.urls import path
from .views import PriceHistoryAPIView, PriceForecastAPIView

urlpatterns = [
    path("", PriceForecastAPIView.as_view()),
]
