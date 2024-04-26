from django.urls import path
from .views import PriceForecastAPIView, PriceForecastRegionAPIView, HA_PriceForecastRegionAPIView

urlpatterns = [
    path("", PriceForecastAPIView.as_view()),
    path("<str:region>/", PriceForecastRegionAPIView.as_view()),
    path("ha/<str:region>/", HA_PriceForecastRegionAPIView.as_view()),
]
