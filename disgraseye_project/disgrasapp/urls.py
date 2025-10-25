from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='landing_page'),
    path('detect/', views.detect_crash, name='detect_crash'),
    path('demo/', views.index, name='demo'),
    path('login/', views.login_view, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout')
]