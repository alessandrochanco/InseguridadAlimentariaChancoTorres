from django.urls import path
from . import views

urlpatterns = [
    path('', views.mostrar_resultados, name='mostrar_resultados'),
]