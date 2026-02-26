from django.contrib import admin
from .models import VideoUpload

@admin.register(VideoUpload)
class VideoUploadAdmin(admin.ModelAdmin):
    list_display = ['id', 'uploaded_at', 'processed', 'processing_complete', 'crash_detected']
    list_filter = ['processed', 'processing_complete', 'crash_detected']
    readonly_fields = ['uploaded_at']
    search_fields = ['id']