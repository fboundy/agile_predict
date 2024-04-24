from django.urls import path
from .views import ForecastListView

urlpatterns = [path("", ForecastListView.as_view(), name="home")]
