from django import forms
from .models import VideoUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['video_file']
        
    def clean_video_file(self):
        video_file = self.cleaned_data.get('video_file')
        if video_file:
            # Validate file extension
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            extension = '.' + video_file.name.split('.')[-1].lower()
            if extension not in valid_extensions:
                raise forms.ValidationError('Unsupported file format. Please upload a video file (MP4, AVI, MOV, MKV, WMV).')
            
            # Validate file size (max 100MB)
            if video_file.size > 100 * 1024 * 1024:
                raise forms.ValidationError('File size too large. Please upload a video smaller than 100MB.')
                
        return video_file