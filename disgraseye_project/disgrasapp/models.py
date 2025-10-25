from django.db import models
from django.utils import timezone

class Detection(models.Model):
    STATUS_CHOICES = [
        ('crash', 'Crash Detected'),
        ('no_crash', 'No Crash'),
    ]
    
    timestamp = models.DateTimeField(default=timezone.now)
    confidence = models.FloatField()
    image_path = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    crash_detected = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.status} - {self.confidence}% at {self.timestamp}"

