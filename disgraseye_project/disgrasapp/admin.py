from django.contrib import admin
from .models import VideoUpload

@admin.register(VideoUpload)
class VideoUploadAdmin(admin.ModelAdmin):
    list_display = ['id', 'uploaded_at', 'processed', 'crash_detected']
    list_filter = ['processed', 'crash_detected']
    readonly_fields = ['uploaded_at']