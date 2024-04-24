from django.shortcuts import render

# Create your views here.
from django.views.generic import ListView, DetailView
from .models import Forecasts, ForecastData


class ForecastListView(ListView):
    model = Forecasts
    template_name = "home.html"


class ForecastDetailView(DetailView):
    model = ForecastData
    template_name = "detail.html"
