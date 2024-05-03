from django.urls import path
from .views import GraphFormView

urlpatterns = [
    path("<str:region>/", GraphFormView.as_view(), name="graph"),
    path("", GraphFormView.as_view(), name="graph"),
]
