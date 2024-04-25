from django.urls import path
from .views import GraphView

urlpatterns = [
    path("", GraphView.as_view(), name="graph"),
]
