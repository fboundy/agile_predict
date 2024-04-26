from django.urls import path
from .views import GraphView

urlpatterns = [
    path("<str:region>/", GraphView.as_view(), name="graph"),
]
