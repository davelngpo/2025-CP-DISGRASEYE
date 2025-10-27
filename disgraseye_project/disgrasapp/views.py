from django.shortcuts import render, redirect
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

# YOLOv8 imports
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import math
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Initialize YOLOv8 model
if YOLO_AVAILABLE:
    model = YOLO('yolov8n.pt')
else:
    model = None

# Authentication Views
def landing_page(request):
    return render(request, 'dashboard/landing_page.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'login/login.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard/dashboard.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def index(request):
    return render(request, 'index.html')

# Crash Detection Views
@csrf_exempt
@login_required
def detect_crash(request):
    """
    AJAX endpoint for crash detection with improved crash detection logic
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = default_storage.save(f'uploads/{filename}', uploaded_file)
    full_path = os.path.join(settings.MEDIA_ROOT, filepath)
    
    # Run detection
    try:
        if not YOLO_AVAILABLE:
            result = simulate_detection(full_path)
        else:
            # Check if file is video or image
            if uploaded_file.content_type.startswith('video'):
                result = process_video_detection(full_path, filepath)
            else:
                result = process_image_detection(full_path, filepath)
        
        # Save detection to database
        detection = Detection.objects.create(
            timestamp=datetime.now(),
            confidence=result['confidence'],
            image_path=result['annotated_path'],
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
    'detection_id': detection.id,
    'vehicle_count': result.get('vehicle_count', 0),
    'crash_reason': result.get('crash_reason', ''),
    'is_video': result.get('is_video', False),  # ADD THIS LINE
    'total_frames': result.get('total_frames', 0)  # ADD THIS LINE
})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def process_image_detection(image_path, original_filepath):
    """
    Process single image for crash detection with bounding boxes
    """
    # Run YOLOv8 inference
    results = model(image_path, conf=0.3)
    
    crash_detected = False
    max_confidence = 0.0
    vehicle_count = 0
    crash_reason = ""
    
    # Get the first result (single image)
    result = results[0]
    boxes = result.boxes
    
    # Vehicle classes in COCO dataset: car(2), motorcycle(3), bus(5), truck(7)
    vehicle_classes = [2, 3, 5, 7]
    
    vehicles = []
    
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls in vehicle_classes:
            vehicle_count += 1
            vehicles.append({
                'class': cls,
                'confidence': conf,
                'bbox': box.xyxy[0].cpu().numpy(),
                'center': calculate_center(box.xyxy[0].cpu().numpy())
            })
            max_confidence = max(max_confidence, conf * 100)
    
    # Check for crash conditions
    if vehicle_count >= 2:
        # Check for vehicle proximity (potential collision)
        crash_detected, crash_reason = check_crash_conditions(vehicles)
    
    # Generate annotated image with bounding boxes
    annotated_path = create_annotated_image(image_path, vehicles, crash_detected)
    
    return {
        'crash_detected': crash_detected,
        'status': 'Crash Detected' if crash_detected else 'No Crash',
        'confidence': round(max_confidence, 2),
        'vehicle_count': vehicle_count,
        'crash_reason': crash_reason,
        'annotated_path': f"/media/{annotated_path}"
    }

# Replace your process_video_detection function with this

def process_video_detection(video_path, original_filepath):
    """
    Process entire video and create annotated video with bounding boxes
    Returns annotated VIDEO with web-compatible format
    """
    print(f"🎥 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Create output video filename - use .mp4 extension
    original_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(original_name)[0]
    output_filename = f"annotated_{name_without_ext}.mp4"
    output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
    
    # Try different codecs for better browser compatibility
    # Try H.264 codec first (most compatible)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If H.264 fails, try MP4V
    if not out.isOpened():
        print("🔄 Trying MP4V codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If still fails, use XVID (should work on most systems)
    if not out.isOpened():
        print("🔄 Trying XVID codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception("Could not create video writer with any codec")
    
    crash_detected = False
    crash_reasons = []
    all_vehicles = []
    frame_count = 0
    max_confidence = 0.0
    
    print(f"⚙️ Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 on this frame
        results = model(frame, conf=0.45, verbose=False)
        result = results[0]
        
        # Extract vehicles from detections
        vehicle_classes = [2, 3, 5, 7]
        class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        
        frame_vehicles = []
        
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls in vehicle_classes:
                bbox = box.xyxy[0].cpu().numpy()
                frame_vehicles.append({
                    'class': cls,
                    'class_name': class_names[cls],
                    'confidence': conf,
                    'bbox': bbox,
                    'center': calculate_center(bbox)
                })
                max_confidence = max(max_confidence, conf * 100)
        
        # Check for crash in this frame
        frame_crash = False
        if len(frame_vehicles) >= 2:
            frame_crash, reason = check_crash_conditions(frame_vehicles)
            if frame_crash and not crash_detected:
                crash_detected = True
                crash_reasons.append(f"Frame {frame_count}: {reason}")
                print(f"  🚨 Crash detected at frame {frame_count}")
        
        # Draw bounding boxes on frame
        annotated_frame = draw_bounding_boxes(frame, frame_vehicles, frame_crash)
        
        # Write annotated frame to output video
        out.write(annotated_frame)
        
        all_vehicles.extend(frame_vehicles)
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            print(f"  ⏳ Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✅ Video processing complete: {frame_count} frames")
    print(f"📹 Output saved to: {output_path}")
    
    # Calculate overall statistics
    avg_vehicles = len(all_vehicles) / frame_count if frame_count > 0 else 0
    crash_reason = crash_reasons[0] if crash_reasons else ""
    
    # Return the correct media URL path
    return {
        'crash_detected': crash_detected,
        'status': 'Crash Detected' if crash_detected else 'No Crash',
        'confidence': round(max_confidence, 2),
        'vehicle_count': int(avg_vehicles),
        'crash_reason': crash_reason,
        'annotated_path': f"/media/uploads/{output_filename}",
        'is_video': True,
        'total_frames': frame_count
    }
def draw_bounding_boxes(frame, vehicles, crash_detected):
    """
    Draw bounding boxes on video frame
    GREEN for normal, RED for crash
    """
    # Colors (BGR)
    vehicle_colors = {
        2: (0, 255, 0),      # Car - Green
        3: (0, 200, 0),      # Motorcycle - Dark Green
        5: (0, 255, 100),    # Bus - Light Green
        7: (0, 180, 0)       # Truck - Medium Green
    }
    
    crash_color = (0, 0, 255)  # Red
    
    for vehicle in vehicles:
        # Choose color
        if crash_detected:
            color = crash_color
        else:
            color = vehicle_colors.get(vehicle['class'], (0, 255, 0))
        
        # Get bbox
        bbox = vehicle['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
        
        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), color, -1)
        
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    # Status banner
    status_text = "🚨 CRASH DETECTED!" if crash_detected else f"✓ Monitoring - {len(vehicles)} vehicles"
    status_color = crash_color if crash_detected else (0, 255, 0)
    
    # Banner background
    cv2.rectangle(frame, (10, 10), (500, 60), status_color, -1)
    cv2.putText(frame, status_text, (20, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def calculate_center(bbox):
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def check_crash_conditions(vehicles):
    """
    Improved crash detection logic
    Only detects REAL crashes, not normal traffic
    """
    if len(vehicles) < 2:
        return False, ""
    
    # Calculate average distance between all vehicles
    distances = []
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            dist = calculate_distance(vehicles[i]['center'], vehicles[j]['center'])
            distances.append(dist)
    
    if not distances:
        return False, ""
    
    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)
    
    # STRICT crash criteria:
    # 1. At least 2 vehicles EXTREMELY close (overlapping)
    # 2. Much closer than average traffic distance
    
    # Very strict threshold for actual crashes
    CRASH_THRESHOLD = 50  # pixels - vehicles must be overlapping
    
    if min_distance < CRASH_THRESHOLD:
        # Check if this is actually unusual (not just normal traffic)
        if min_distance < avg_distance * 0.3:  # 30% of average distance
            return True, f"Collision detected: vehicles {min_distance:.0f}px apart (critical proximity)"
    
    # Additional check: 3+ vehicles in very tight cluster
    tight_cluster = sum(1 for d in distances if d < 80)
    if tight_cluster >= 3 and min_distance < 60:
        return True, f"Multi-vehicle collision detected ({tight_cluster} vehicles in tight cluster)"
    
    return False, ""

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_annotated_image(image_path, vehicles, crash_detected):
    """
    Create annotated image with colored bounding boxes
    """
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Define colors
    crash_color = (255, 0, 0)  # Red for crash
    normal_color = (0, 255, 0)  # Green for normal
    text_color = (255, 255, 255)  # White text
    
    # Class names
    class_names = {
        2: "Car",
        3: "Motorcycle", 
        5: "Bus",
        7: "Truck"
    }
    
    for vehicle in vehicles:
        # Choose color based on crash detection
        color = crash_color if crash_detected else normal_color
        
        # Draw bounding box
        bbox = vehicle['bbox']
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], 
                      outline=color, width=3)
        
        # Draw label
        class_name = class_names.get(vehicle['class'], "Vehicle")
        confidence = vehicle['confidence']
        label = f"{class_name} {confidence:.2f}"
        
        # Simple text background
        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([bbox[0], bbox[1] - text_height - 5, 
                       bbox[0] + text_width + 10, bbox[1]], 
                      fill=color)
        draw.text((bbox[0] + 5, bbox[1] - text_height - 2), 
                 label, fill=text_color)
    
    # Add overall status text
    status_text = "🚨 CRASH DETECTED!" if crash_detected else "✅ No Crash - Normal"
    status_color = crash_color if crash_detected else normal_color
    
    # Draw status box
    status_bbox = draw.textbbox((0, 0), status_text)
    status_width = status_bbox[2] - status_bbox[0]
    draw.rectangle([10, 10, status_width + 30, 50], fill=status_color)
    draw.text((20, 15), status_text, fill=text_color)
    
    # Save annotated image
    annotated_filename = f"annotated_{os.path.basename(image_path)}"
    annotated_path = os.path.join(settings.MEDIA_ROOT, 'uploads', annotated_filename)
    image.save(annotated_path)
    
    return f"uploads/{annotated_filename}"

def create_annotated_image_from_frame(video_path, frame_number, vehicles, crash_detected):
    """
    Extract frame from video and create annotated image
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image)
        
        # Same annotation logic as create_annotated_image
        crash_color = (255, 0, 0)
        normal_color = (0, 255, 0)
        text_color = (255, 255, 255)
        
        class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        
        for vehicle in vehicles:
            color = crash_color if crash_detected else normal_color
            bbox = vehicle['bbox']
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], 
                          outline=color, width=3)
            
            class_name = class_names.get(vehicle['class'], "Vehicle")
            confidence = vehicle['confidence']
            label = f"{class_name} {confidence:.2f}"
            
            text_bbox = draw.textbbox((0, 0), label)
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([bbox[0], bbox[1] - text_height - 5, 
                           bbox[0] + (text_bbox[2] - text_bbox[0]) + 10, bbox[1]], 
                          fill=color)
            draw.text((bbox[0] + 5, bbox[1] - text_height - 2), 
                     label, fill=text_color)
        
        # Status text
        status_text = "🚨 CRASH DETECTED!" if crash_detected else "✅ No Crash - Normal"
        status_color = crash_color if crash_detected else normal_color
        
        status_bbox = draw.textbbox((0, 0), status_text)
        status_width = status_bbox[2] - status_bbox[0]
        draw.rectangle([10, 10, status_width + 30, 50], fill=status_color)
        draw.text((20, 15), status_text, fill=text_color)
        
        # Save annotated frame
        annotated_filename = f"annotated_frame_{frame_number}_{os.path.basename(video_path)}.jpg"
        annotated_path = os.path.join(settings.MEDIA_ROOT, 'uploads', annotated_filename)
        image.save(annotated_path)
        
        cap.release()
        return f"uploads/{annotated_filename}"
    
    cap.release()
    # Fallback to original file if frame extraction fails
    return f"uploads/{os.path.basename(video_path)}"

def simulate_detection(image_path):
    """
    Fallback simulation when YOLOv8 is not available
    For demonstration purposes only
    """
    import random
    
    # Simulate detection with random results
    crash_detected = random.choice([True, False])
    confidence = random.uniform(75, 98) if crash_detected else random.uniform(60, 85)
    
    # Create a simple annotated image for simulation
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Add simulated status text
        status_text = "🚨 CRASH DETECTED!" if crash_detected else "✅ No Crash - Normal"
        status_color = (255, 0, 0) if crash_detected else (0, 255, 0)
        text_color = (255, 255, 255)
        
        status_bbox = draw.textbbox((0, 0), status_text)
        status_width = status_bbox[2] - status_bbox[0]
        draw.rectangle([10, 10, status_width + 30, 50], fill=status_color)
        draw.text((20, 15), status_text, fill=text_color)
        
        # Save annotated image
        annotated_filename = f"simulated_{os.path.basename(image_path)}"
        annotated_path = os.path.join(settings.MEDIA_ROOT, 'uploads', annotated_filename)
        image.save(annotated_path)
        
        annotated_path_url = f"uploads/{annotated_filename}"
    except Exception as e:
        # If annotation fails, use original path
        annotated_path_url = f"uploads/{os.path.basename(image_path)}"
    
    return {
        'crash_detected': crash_detected,
        'status': 'Crash Detected' if crash_detected else 'No Crash',
        'confidence': round(confidence, 2),
        'vehicle_count': random.randint(0, 4),
        'crash_reason': "Simulated detection" if crash_detected else "",
        'annotated_path': f"/media/{annotated_path_url}"
    }

@login_required
def get_recent_detections(request):
    """
    API endpoint to fetch recent detections for dashboard
    """
    try:
        offset = int(request.GET.get('offset', 0))
        limit = int(request.GET.get('limit', 5))
        
        # Get all detections ordered by newest first
        all_detections = Detection.objects.all().order_by('-timestamp')
        
        # Apply pagination
        start = offset
        end = offset + limit
        detections_page = all_detections[start:end]
        
        detections_list = []
        for detection in detections_page:
            # Check if it's a video by file extension
            is_video = False
            if detection.image_path:
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                is_video = any(detection.image_path.lower().endswith(ext) for ext in video_extensions)
            
            detections_list.append({
                'id': detection.id,
                'detection_id': detection.id,
                'timestamp': detection.timestamp.isoformat(),
                'status': detection.status,
                'confidence': detection.confidence,
                'crash_detected': detection.crash_detected,
                'image_path': detection.image_path,
                'vehicle_count': getattr(detection, 'vehicle_count', 0),
                'crash_reason': getattr(detection, 'crash_reason', ''),
                'is_video': is_video
            })
        
        return JsonResponse({
            'detections': detections_list,
            'has_more': len(all_detections) > end,
            'total': all_detections.count()
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
