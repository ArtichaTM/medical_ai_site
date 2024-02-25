from django.urls import path, include

from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('generic/', generic, name='generic'),
    path('elements/', elements, name='elements')
]
