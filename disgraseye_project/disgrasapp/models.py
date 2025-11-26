from django.db import models

class VideoUpload(models.Model):
    video_file = models.FileField(upload_to='videos/')
    processed_video = models.FileField(upload_to='processed_videos/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    crash_detected = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Video {self.id} - {'Crash' if self.crash_detected else 'No Crash'}"