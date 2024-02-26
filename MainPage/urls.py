from pathlib import Path

from django.urls import path, include
from django.conf import settings

from .views import *


__all__ = (
    'urlpatterns'
)


models_folder = Path(settings.BASE_DIR / 'MainPage' / 'static' / 'MainPage' / 'extra' / 'keras_models')


urlpatterns = [
    path('', index, name='index'),
    path('generic/', generic, name='generic'),
    path('elements/', elements, name='elements'),
    path('check-disease', DiseaseChecker.as_view(path_model=models_folder, name='Test'), name='check-disease')
]
