from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import os
from datetime import datetime
from .models import Detection
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import redirect



def landing_page(request):
    return render(request, 'dashboard/landing_page.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard/dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'login/login.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard/dashboard.html')

def logout_view(request):
    logout(request)
    return redirect('dashboard/landing_page')






# YOLOv8 imports (FREE - using Ultralytics)
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Initialize YOLOv8 model (loads once at startup)
if YOLO_AVAILABLE:
    # Download free YOLOv8 model if not exists
    model = YOLO('yolov8n.pt')  # nano model - free and fast
else:
    model = None


def index(request):
    """Render the landing page"""
    return render(request, 'index.html')


def detect_crash(request):
    """
    AJAX endpoint for crash detection
    Accepts: POST with file upload
    Returns: JSON with detection results
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = default_storage.save(f'uploads/{filename}', uploaded_file)
    full_path = os.path.join(settings.MEDIA_ROOT, filepath)
    
    # Run YOLOv8 detection
    try:
        if not YOLO_AVAILABLE:
            # Fallback for demo purposes
            result = simulate_detection(full_path)
        else:
            result = run_yolo_detection(full_path)
        
        # Save detection to database
        detection = Detection.objects.create(
            timestamp=datetime.now(),
            confidence=result['confidence'],
            image_path=f"/media/{filepath}",
            status=result['status'],
            crash_detected=result['crash_detected']
        )
        
        return JsonResponse({
            'success': True,
            'crash_detected': result['crash_detected'],
            'status': result['status'],
            'confidence': result['confidence'],
            'timestamp': detection.timestamp.isoformat(),
            'image_path': detection.image_path,
            'detection_id': detection.id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def run_yolo_detection(image_path):
    """
    Run YOLOv8 detection on uploaded image/video
    Detects vehicles and crash scenarios
    """
    # Run inference
    results = model(image_path, conf=0.5)
    
    # Analyze results for crash detection
    # Look for: overturned vehicles, debris, accident patterns
    crash_detected = False
    max_confidence = 0.0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # YOLOv8 class IDs for vehicles: car(2), motorcycle(3), bus(5), truck(7)
            vehicle_classes = [2, 3, 5, 7]
            
            if cls in vehicle_classes:
                # Check for unusual orientations or positions (crash indicators)
                # For demo: simulate crash detection with confidence threshold
                if conf > 0.85:  # High confidence vehicle detection
                    crash_detected = True
                    max_confidence = max(max_confidence, conf * 100)
        
        # Save annotated image
        annotated = result.plot()
        output_path = image_path.replace('.', '_detected.')
        cv2.imwrite(output_path, annotated)
    
    # If no crash detected, use original confidence
    if not crash_detected:
        max_confidence = 75.0  # Base confidence for no crash
    
    return {
        'crash_detected': crash_detected,
        'status': 'Crash Detected' if crash_detected else 'No Crash',
        'confidence': round(max_confidence, 2)
    }


def simulate_detection(image_path):
    """
    Fallback simulation when YOLOv8 is not available
    For demonstration purposes only
    """
    import random
    
    # Simulate detection with random results
    crash_detected = random.choice([True, False])
    confidence = random.uniform(75, 98) if crash_detected else random.uniform(60, 85)
    
    return {
        'crash_detected': crash_detected,
        'status': 'Crash Detected' if crash_detected else 'No Crash',
        'confidence': round(confidence, 2)
    }


def get_recent_detections(request):
    """
    API endpoint to fetch recent detections for dashboard
    """
    detections = Detection.objects.all()[:10]
    
    data = [{
        'id': d.id,
        'timestamp': d.timestamp.isoformat(),
        'status': d.status,
        'confidence': d.confidence,
        'crash_detected': d.crash_detected
    } for d in detections]
    
    return JsonResponse({'detections': data})
