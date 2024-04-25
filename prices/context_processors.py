# from django.conf import settings
from . import __version__


def selected_settings(request):
    # return the version value as a dictionary
    # you may add other values here as well
    return {"APP_VERSION_NUMBER": __version__}
