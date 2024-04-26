from django.urls import path
from .views import PriceForecastAPIView, PriceForecastRegionAPIView

urlpatterns = [
    path("", PriceForecastAPIView.as_view()),
    path("<str:region>/", PriceForecastRegionAPIView.as_view()),
]
