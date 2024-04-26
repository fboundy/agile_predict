# from django.conf import settings
from . import __version__
from config.settings import GLOBAL_SETTINGS


def selected_settings(request):
    # return the version value as a dictionary
    # you may add other values here as well
    return {
        "APP_VERSION_NUMBER": __version__,
        "REGIONS": {r: GLOBAL_SETTINGS["REGIONS"][r]["name"] for r in GLOBAL_SETTINGS["REGIONS"]},
    }
