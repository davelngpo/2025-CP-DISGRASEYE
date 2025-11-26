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

   # Video processing URLs
    path('download/<int:video_id>/', views.download_processed_video, name='download_processed_video'),
    path('view/<int:video_id>/', views.view_processed_video, name='view_processed_video'),
    
    # RTSP Live Monitoring URLs
    path('live/', views.live_monitoring, name='live_monitoring'),
    path('live/stream/', views.rtsp_stream, name='rtsp_stream'),
    path('live/start/', views.start_rtsp_monitoring, name='start_rtsp_monitoring'),
    path('live/stop/', views.stop_rtsp_monitoring, name='stop_rtsp_monitoring'),
    path('live/status/', views.get_detection_status, name='get_detection_status'),  

    path('start-rtsp-monitoring/', views.start_rtsp_monitoring, name='start_rtsp_monitoring'),
    path('get-rtsp-status/', views.get_rtsp_status, name='get_rtsp_status'),

    path('camera-stream/<str:camera_id>/', views.camera_stream, name='camera_stream'),
    path('start-camera-stream/', views.start_camera_stream, name='start_camera_stream'),
    path('stop-camera-stream/', views.stop_camera_stream, name='stop_camera_stream'),
    path('toggle-camera-ai/', views.toggle_camera_ai, name='toggle_camera_ai'),
    path('camera-status/<str:camera_id>/', views.get_camera_status, name='camera_status'),
]