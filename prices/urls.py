from django.urls import path
from .views import GraphFormView

urlpatterns = [
    # path("<str:region>/", GraphView.as_view(), name="graph"),
    path("", GraphFormView.as_view(), name="graph"),
]
