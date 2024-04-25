from django.urls import path
from .views import ForecastsListView, Graph

urlpatterns = [
    path("", ForecastsListView.as_view(), name="home"),
    path("graph", Graph.as_view(), name="graph"),
]
