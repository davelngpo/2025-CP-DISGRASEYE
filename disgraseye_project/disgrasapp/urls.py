from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    #Login Logout path :)
    path('login/', views.login_view, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),

    # Dashboard sections
    path('dashboard/cctv/', views.cctv_monitoring, name='cctv_monitoring'),
    path('dashboard/reports/', views.reports, name='reports'),
    path('dashboard/settings/', views.settings, name='settings'),

    path('detect/', views.detect_crash, name='detect_crash'),
    path('demo/', views.index, name='demo'),
    path('api/detections/', views.get_recent_detections, name='recent_detections'),
]