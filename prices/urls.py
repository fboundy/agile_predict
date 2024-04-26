from django.urls import path
from .views import GraphView

urlpatterns = [
    path("<str:area>/", GraphView.as_view(), name="graph"),
]
