import os
import cv2
import time
import threading
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse, StreamingHttpResponse
from django.conf import settings
from django.conf import settings as django_settings  
from ultralytics import YOLO
from django.contrib.auth.decorators import login_required
import numpy as np
from .forms import VideoUploadForm
from .models import VideoUpload
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
import threading
from collections import defaultdict
from django.views.decorators.csrf import csrf_exempt
from supabase import create_client
from .decorators import login_required
from wsgiref.util import FileWrapper
import mimetypes
from collections import defaultdict
import queue

# Load your trained model
model_path = os.path.join(settings.BASE_DIR, 'best.pt')
model = YOLO(model_path)

# Global variables for RTSP monitoring
is_monitoring = False
latest_detection_result = None
current_rtsp_url = ""

from collections import deque
import uuid

# Rolling frame buffer for each camera: { camera_id: deque of (timestamp, frame) }
# At 5fps × 5 seconds = 25 frames pre-crash, ~5-15 MB RAM per camera
camera_frame_buffers = {}

# Tracks cameras currently collecting post-crash frames
# { camera_id: { 'before': [...], 'after': [...], 'target': int, 'accident_id': int } }
camera_post_crash_capture = {}

CLIP_FPS          = 10   # fps stored in buffer (NOT stream fps — saves RAM)
CLIP_BEFORE_SECS  = 5   # seconds of footage before crash
CLIP_AFTER_SECS   = 5   # seconds of footage after crash
CLIP_BEFORE_FRAMES = CLIP_FPS * CLIP_BEFORE_SECS   # 25 frames
CLIP_AFTER_FRAMES  = CLIP_FPS * CLIP_AFTER_SECS    # 25 frames

# How often to sample a frame into the buffer (e.g. if stream is 15fps, sample 1-in-3)
# Adjust STREAM_FPS to match your actual RTSP fps
STREAM_FPS    = 15
SAMPLE_EVERY  = max(1, STREAM_FPS // CLIP_FPS)  # = 3  (every 3rd frame)

# =============================================================================
# AUTHENTICATION & BASIC VIEWS
# =============================================================================

# Initialize Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY) 


@login_required
def dashboard(request):
    # Get user data from session (stored during login)
    supabase_user = request.session.get('supabase_user', {})
    
    # Initialize Supabase client
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    
    # Get the user's profile from the users table to get the correct user_id
    try:
        profile = supabase.table('users')\
            .select('user_id, first_name, last_name, role')\
            .eq('auth_id', supabase_user.get('id'))\
            .single()\
            .execute()
        
        if profile.data:
            user_data = {
                "user_id": profile.data['user_id'],  # This is the INTEGER ID
                "auth_id": supabase_user.get('id'),   # This is the UUID
                "first_name": profile.data.get('first_name', ''),
                "last_name": profile.data.get('last_name', ''),
                "role": profile.data.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
        else:
            user_data = {
                "user_id": supabase_user.get('id'), 
                "auth_id": supabase_user.get('id'),
                "first_name": "",
                "last_name": "",
                "role": supabase_user.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        user_data = {
            "user_id": supabase_user.get('id'),
            "auth_id": supabase_user.get('id'),
            "first_name": "",
            "last_name": "",
            "role": supabase_user.get('role', 'admin'),
            "email": supabase_user.get('email')
        }
    
    return render(request, 'dashboard/dashboard.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": settings.SUPABASE_SERVICE_ROLE_KEY,
        "user": user_data
    })

@login_required
def cctv_monitoring(request):
    supabase_user = request.session.get('supabase_user', {})
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    
    try:
        profile = supabase.table('users')\
            .select('user_id, first_name, last_name, role')\
            .eq('auth_id', supabase_user.get('id'))\
            .single()\
            .execute()
        
        if profile.data:
            user_data = {
                "user_id": profile.data['user_id'],
                "auth_id": supabase_user.get('id'),
                "first_name": profile.data.get('first_name', ''),
                "last_name": profile.data.get('last_name', ''),
                "role": profile.data.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
        else:
            user_data = {
                "user_id": supabase_user.get('id'),
                "auth_id": supabase_user.get('id'),
                "first_name": "",
                "last_name": "",
                "role": supabase_user.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        user_data = {
            "user_id": supabase_user.get('id'),
            "auth_id": supabase_user.get('id'),
            "first_name": "",
            "last_name": "",
            "role": supabase_user.get('role', 'admin'),
            "email": supabase_user.get('email')
        }
    
    return render(request, 'dashboard/cctv_monitoring.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": settings.SUPABASE_SERVICE_ROLE_KEY,
        "user": user_data 
    })

@login_required
def live_monitoring(request):
    global current_rtsp_url
    
    supabase_user = request.session.get('supabase_user', {})
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    
    # Get RTSP URL
    if not current_rtsp_url:
        current_rtsp_url = request.session.get('current_rtsp_url', '')
    
    # Get user profile
    try:
        profile = supabase.table('users')\
            .select('user_id, first_name, last_name, role')\
            .eq('auth_id', supabase_user.get('id'))\
            .single()\
            .execute()
        
        if profile.data:
            user_data = {
                "user_id": profile.data['user_id'],
                "auth_id": supabase_user.get('id'),
                "first_name": profile.data.get('first_name', ''),
                "last_name": profile.data.get('last_name', ''),
                "role": profile.data.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
        else:
            user_data = {
                "user_id": supabase_user.get('id'),
                "auth_id": supabase_user.get('id'),
                "first_name": "",
                "last_name": "",
                "role": supabase_user.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        user_data = {
            "user_id": supabase_user.get('id'),
            "auth_id": supabase_user.get('id'),
            "first_name": "",
            "last_name": "",
            "role": supabase_user.get('role', 'admin'),
            "email": supabase_user.get('email')
        }
    
    return render(request, 'dashboard/live_monitoring.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": settings.SUPABASE_SERVICE_ROLE_KEY,
        "RTSP_URL": current_rtsp_url,
        "user": user_data
    })
@login_required
def reports(request):
    supabase_user = request.session.get('supabase_user', {})
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    
    try:
        profile = supabase.table('users')\
            .select('user_id, first_name, last_name, role')\
            .eq('auth_id', supabase_user.get('id'))\
            .single()\
            .execute()
        
        if profile.data:
            user_data = {
                "user_id": profile.data['user_id'],
                "auth_id": supabase_user.get('id'),
                "first_name": profile.data.get('first_name', ''),
                "last_name": profile.data.get('last_name', ''),
                "role": profile.data.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
        else:
            user_data = {
                "user_id": supabase_user.get('id'),
                "auth_id": supabase_user.get('id'),
                "first_name": "",
                "last_name": "",
                "role": supabase_user.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        user_data = {
            "user_id": supabase_user.get('id'),
            "auth_id": supabase_user.get('id'),
            "first_name": "",
            "last_name": "",
            "role": supabase_user.get('role', 'admin'),
            "email": supabase_user.get('email')
        }
    
    return render(request, 'dashboard/reports.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": settings.SUPABASE_SERVICE_ROLE_KEY,
        "user": user_data 
    })

@login_required
def site_settings(request):
    supabase_user = request.session.get('supabase_user', {})
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    
    try:
        profile = supabase.table('users')\
            .select('user_id, first_name, last_name, role')\
            .eq('auth_id', supabase_user.get('id'))\
            .single()\
            .execute()
        
        if profile.data:
            user_data = {
                "user_id": profile.data['user_id'],
                "auth_id": supabase_user.get('id'),
                "first_name": profile.data.get('first_name', ''),
                "last_name": profile.data.get('last_name', ''),
                "role": profile.data.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
        else:
            user_data = {
                "user_id": supabase_user.get('id'),
                "auth_id": supabase_user.get('id'),
                "first_name": "",
                "last_name": "",
                "role": supabase_user.get('role', 'admin'),
                "email": supabase_user.get('email')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        user_data = {
            "user_id": supabase_user.get('id'),
            "auth_id": supabase_user.get('id'),
            "first_name": "",
            "last_name": "",
            "role": supabase_user.get('role', 'admin'),
            "email": supabase_user.get('email')
        }
    
    return render(request, 'dashboard/settings.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": settings.SUPABASE_SERVICE_ROLE_KEY,
        "user": user_data 
    })

def logout_view(request):
    request.session.flush()
    return redirect('login')

def landing_page(request):
    return render(request, 'dashboard/landing_page.html')

def login_view(request):
    # Handle AJAX login
    if request.method == 'POST' and request.headers.get('Content-Type') == 'application/json':
        try:
            data = json.loads(request.body)
            
            # Authenticate with Supabase using existing credentials
            auth_response = supabase.auth.sign_in_with_password({
                "email": data.get('email'),
                "password": data.get('password')
            })
            
            if auth_response.user:
                # Check if user has admin role in your users table
                profile = supabase.table('users')\
                    .select('role')\
                    .eq('auth_id', auth_response.user.id)\
                    .single()\
                    .execute()
                
                if profile.data and profile.data['role'] == 'admin':
                    # Store minimal user info in session
                    request.session['supabase_user'] = {
                        'id': auth_response.user.id,
                        'email': auth_response.user.email,
                        'role': profile.data['role']
                    }
                    
                    return JsonResponse({
                        'success': True,
                        'redirect': '/dashboard/'
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'Access denied. Admin account required.'
                    }, status=403)
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid email or password'
                }, status=401)
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    # Regular GET request - show login page
    return render(request, 'login/login.html')

# =============================================================================
# LIVE DETECTION PAGE FUNCTIONS
# =============================================================================

# Global AI queue and state for live monitoring
live_ai_queue = queue.Queue(maxsize=2)
live_detection_state = {}  # shared between live_ai_worker and generate_frames

def live_ai_worker():
    """AI worker for live monitoring page — same pattern as camera AI worker"""
    global is_monitoring
    print("🤖 Live monitoring AI worker started")

    while is_monitoring:
        try:
            frame = live_ai_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            small_frame = cv2.resize(frame, (320, 240))
            results = model(small_frame, imgsz=320, conf=0.5, verbose=False)

            crash_detected = False
            detected_classes = []
            detected_confidences = []
            detected_boxes = []

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id   = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detected_boxes.append({
                        'x1':      int(x1 * 2),
                        'y1':      int(y1 * 2),
                        'x2':      int(x2 * 2),
                        'y2':      int(y2 * 2),
                        'class':   class_name,
                        'conf':    confidence,
                        'is_crash': any(k in class_name.lower() for k in ['crash', 'accident'])
                    })

                    detected_classes.append(class_name)
                    detected_confidences.append(confidence)

                    if any(k in class_name.lower() for k in ['crash', 'accident']):
                        crash_detected = True

            latency = live_detection_state.get('last_inference_start', 0)
            latency_ms = int((time.time() - latency) * 1000) if latency else 0

            live_detection_state.update({
                'crash_detected':   crash_detected,
                'classes':          detected_classes,
                'confidences':      detected_confidences,
                'boxes':            detected_boxes,
                'timestamp':        time.time(),
                'latency_ms':       latency_ms,
                'frame_count':      live_detection_state.get('frame_count', 0) + 1,
            })

            # Update global latest_detection_result so /live/status/ still works
            global latest_detection_result
            latest_detection_result = {
                'crash_detected': crash_detected,
                'crash_classes':  detected_classes,
                'timestamp':      time.time(),
                'latency_ms':     latency_ms,
                'frame_count':    live_detection_state.get('frame_count', 0),
            }

            # Send Supabase alert with 30s cooldown
            if crash_detected:
                last_alert = live_detection_state.get('last_alert_time', 0)
                if time.time() - last_alert > 30:
                    live_detection_state['last_alert_time'] = time.time()
                    alert_thread = threading.Thread(
                        target=send_crash_alert_to_supabase,
                        kwargs={'accident_id': None, 'location': f'Live RTSP: {current_rtsp_url}'}
                    )
                    alert_thread.daemon = True
                    alert_thread.start()

        except Exception as e:
            print(f"❌ Live AI worker error: {e}")

    print("🛑 Live monitoring AI worker stopped")


def generate_frames():
    """Generate frames from RTSP — AI runs on separate thread, stream never blocks"""
    global is_monitoring, current_rtsp_url

    if not current_rtsp_url:
        print("❌ No RTSP URL configured")
        return

    cap = cv2.VideoCapture(current_rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # Reset state
    live_detection_state.clear()

    # Start AI worker thread
    ai_thread = threading.Thread(target=live_ai_worker, daemon=True)
    ai_thread.start()

    print(f"🎥 Starting RTSP stream: {current_rtsp_url}")

    while is_monitoring:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from RTSP")
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            # Send to AI worker (non-blocking)
            live_detection_state['last_inference_start'] = time.time()
            try:
                live_ai_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            # Draw last known boxes
            detected_boxes  = live_detection_state.get('boxes', [])
            crash_detected  = live_detection_state.get('crash_detected', False)
            det_classes     = live_detection_state.get('classes', [])
            det_confidences = live_detection_state.get('confidences', [])
            latency_ms      = live_detection_state.get('latency_ms', 0)
            frame_count     = live_detection_state.get('frame_count', 0)
            last_ts         = live_detection_state.get('timestamp', 0)
            last_alert      = live_detection_state.get('last_alert_time', 0)

            # ── Bounding boxes ────────────────────────────────────────────
            for det in detected_boxes:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                is_crash = det['is_crash']
                label    = f"{det['class']} {det['conf']:.0%}"
                color    = (0, 0, 255) if is_crash else (0, 255, 0)

                label_bg_y = max(y1 - 22, 0)
                cv2.rectangle(display_frame, (x1, label_bg_y),
                             (x1 + len(label) * 9, y1), color, -1)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1 + 3, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if is_crash:
                    cv2.rectangle(display_frame, (0, 0),
                                 (display_frame.shape[1]-1, display_frame.shape[0]-1),
                                 (0, 0, 255), 4)

            # ── Debug panel ───────────────────────────────────────────────
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

            if crash_detected:
                cv2.putText(display_frame, "!! CRASH DETECTED !!", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "AI ACTIVE - Monitoring", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if det_classes:
                classes_str = ', '.join(
                    f"{c}({conf:.0%})" for c, conf in zip(det_classes, det_confidences)
                )
                cv2.putText(display_frame, f"Detected: {classes_str}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                cv2.putText(display_frame, "Detected: nothing", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            if last_ts:
                age_ms = int((time.time() - last_ts) * 1000)
                cv2.putText(display_frame, f"Last inference: {age_ms}ms ago", (10, 72),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            else:
                cv2.putText(display_frame, "Last inference: waiting...", (10, 72),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(display_frame, f"Frames analyzed: {frame_count}", (10, 94),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if last_alert:
                cooldown = max(0, 30 - int(time.time() - last_alert))
                cd_color = (0, 165, 255) if cooldown > 0 else (0, 255, 0)
                cv2.putText(display_frame, f"Alert cooldown: {cooldown}s", (10, 116),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, cd_color, 1)
            else:
                cv2.putText(display_frame, "Alert cooldown: ready", (10, 116),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            ts = time.strftime("%H:%M:%S")
            cv2.putText(display_frame, f"Time: {ts}", (10, 138),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Encode and yield immediately
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            time.sleep(0.1)
            continue

    cap.release()
    print("🛑 RTSP stream capture stopped")

def rtsp_stream(request):
    """Stream RTSP feed with live crash detection"""
    global is_monitoring, current_rtsp_url
    
    try:
        if not current_rtsp_url:
            return HttpResponse("No RTSP URL configured", status=400)
            
        print(f"🎬 Stream endpoint called for: {current_rtsp_url}")
        is_monitoring = True
        
        response = StreamingHttpResponse(
            generate_frames(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
        
    except Exception as e:
        print(f"❌ Stream error: {e}")
        is_monitoring = False
        return HttpResponse("Stream error", status=500)

@login_required
def start_rtsp_monitoring(request):
    """Start RTSP monitoring with dynamic URL"""
    global is_monitoring, current_rtsp_url
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            rtsp_url = data.get('rtsp_url', '').strip()
            
            # Validate RTSP URL
            if not rtsp_url:
                return JsonResponse({'error': 'RTSP URL is required'}, status=400)
            
            if not rtsp_url.startswith(('rtsp://', 'http://', 'https://')):
                return JsonResponse({'error': 'Invalid URL format. Must start with rtsp://, http://, or https://'}, status=400)
            
            # Set the current RTSP URL
            current_rtsp_url = rtsp_url
            is_monitoring = True
            
            # Store in session for persistence
            request.session['current_rtsp_url'] = rtsp_url
            
            print(f"▶ RTSP monitoring started: {rtsp_url}")
            return JsonResponse({
                'status': 'success', 
                'message': 'RTSP monitoring started',
                'rtsp_url': rtsp_url
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def stop_rtsp_monitoring(request):
    """Stop RTSP monitoring"""
    global is_monitoring
    
    is_monitoring = False
    print("⏹ RTSP monitoring stopped")
    return HttpResponse("RTSP monitoring stopped")

@login_required
def get_detection_status(request):
    """Get current detection status"""
    global latest_detection_result
    
    if latest_detection_result:
        latency = latest_detection_result.get('latency_ms', 0)
        crash_status = "True" if latest_detection_result['crash_detected'] else "False"
        frame_count = latest_detection_result.get('frame_count', 0)
        return HttpResponse(f"Crash:{crash_status}|Latency:{latency}ms|Frames:{frame_count}")
    
    return HttpResponse("Crash:False|Latency:0ms|Frames:0")

@login_required
def get_rtsp_status(request):
    """Get current RTSP URL status"""
    global current_rtsp_url
    
    # Try to get from session if not set globally
    if not current_rtsp_url:
        current_rtsp_url = request.session.get('current_rtsp_url', '')
    
    return JsonResponse({'rtsp_url': current_rtsp_url})



# =============================================================================
# FIXED VIDEO UPLOAD AND PROCESSING FUNCTIONS
# =============================================================================

def process_video_thread(video_id):
    """
    Background thread for video processing
    """
    try:
        video_upload = VideoUpload.objects.get(id=video_id)
        result = process_video(video_upload)
        
        if result:
            print(f"✅ Video {video_id} processed successfully. Crash detected: {result['crash_detected']}")
        else:
            print(f"❌ Video {video_id} processing failed")
            
    except Exception as e:
        print(f"❌ Error in process_video_thread for video {video_id}: {e}")

def process_video(video_upload):
    """
    Process uploaded video with YOLOv8 model for crash detection
    Ensures proper video encoding and file integrity
    """
    CRASH_CONFIDENCE_THRESHOLD = 0.60
    CRASH_KEYWORDS = ['crash', 'accident', 'collision', 'wreck', 'smash']
    
    try:
        # Update status
        video_upload.processing_complete = False
        video_upload.save()
        
        input_video_path = video_upload.video_file.path
        
        # Create output path with proper extension
        original_name = os.path.basename(input_video_path)
        name_without_ext = os.path.splitext(original_name)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f'processed_{name_without_ext}_{timestamp}.mp4'
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', output_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📹 Input video: {input_video_path}")
        print(f"📹 Output path: {output_path}")
        
        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 60:  # Handle invalid FPS
            fps = 30
            print(f"⚠️ Invalid FPS detected, using default: {fps}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure dimensions are valid
        if width <= 0 or height <= 0:
            # Try to read first frame to get dimensions
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                height, width = test_frame.shape[:2]
                # Reset capture to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                raise Exception("Could not determine video dimensions")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Estimate by reading all frames (slow but accurate)
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            # Reset capture to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"📹 Processing video: {width}x{height} at {fps} FPS, {total_frames} total frames")
        
        # Try different codecs in order of compatibility
        codecs_to_try = [
            ('avc1', 'H.264'),
            ('H264', 'H.264 (alt)'),
            ('mp4v', 'MPEG-4'),
            ('X264', 'x264'),
            ('MJPG', 'Motion JPEG'),
            ('DIVX', 'DivX'),
            ('XVID', 'Xvid')
        ]
        
        out = None
        used_codec = None
        
        for fourcc_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    used_codec = fourcc_str
                    print(f"✅ Successfully opened VideoWriter with codec: {codec_name} ({fourcc_str})")
                    break
                else:
                    print(f"⚠️ Failed with codec: {codec_name} ({fourcc_str})")
            except Exception as e:
                print(f"⚠️ Error with codec {fourcc_str}: {e}")
                continue
        
        if out is None or not out.isOpened():
            # Last resort: try without specifying codec (let OpenCV choose)
            try:
                out = cv2.VideoWriter(output_path, 0, fps, (width, height))
                if out.isOpened():
                    used_codec = "default"
                    print("✅ Using default codec")
                else:
                    raise Exception("Could not open VideoWriter with any codec")
            except Exception as e:
                raise Exception(f"Failed to create output video writer: {e}")
        
        print(f"✅ Using codec: {used_codec}")
        
        crash_detected_any_frame = False
        frame_count = 0
        crash_frames = 0
        crash_events = []
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"📊 Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print(f"⚠️ Invalid frame at {frame_count}, skipping")
                continue
            
            # Run YOLO detection
            try:
                results = model(frame, conf=CRASH_CONFIDENCE_THRESHOLD, verbose=False)
            except Exception as e:
                print(f"⚠️ YOLO error on frame {frame_count}: {e}")
                # Continue with unannotated frame
                results = None
            
            frame_has_crash = False
            crash_details = []
            
            # Check if crash is detected
            if results and len(results) > 0:
                for result in results:
                    if result is not None and result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Check if this class indicates a crash
                            if any(keyword in class_name.lower() for keyword in CRASH_KEYWORDS):
                                frame_has_crash = True
                                crash_details.append({
                                    'class': class_name,
                                    'confidence': confidence
                                })
                                
                                print(f"🚨 Crash detected in frame {frame_count}: {class_name} ({confidence:.2f})")
            
            if frame_has_crash:
                crash_frames += 1
                crash_detected_any_frame = True
                
                # Store crash event timestamp
                timestamp_seconds = frame_count / fps if fps > 0 else 0
                crash_events.append({
                    'frame': frame_count,
                    'timestamp': timestamp_seconds,
                    'details': crash_details
                })
            
            # Visualize results
            if results and len(results) > 0:
                try:
                    annotated_frame = results[0].plot()
                except:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
            
            # Add crash detection info
            if frame_has_crash:
                cv2.putText(annotated_frame, "🚨 CRASH DETECTED! 🚨", (50, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Add processing info
            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timestamp
            time_seconds = frame_count / fps if fps > 0 else 0
            time_str = time.strftime('%H:%M:%S', time.gmtime(time_seconds))
            cv2.putText(annotated_frame, f"Time: {time_str}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write the frame
            try:
                out.write(annotated_frame)
            except Exception as e:
                print(f"❌ Error writing frame {frame_count}: {e}")
        
        # Properly release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Wait a moment for file to be fully written
        time.sleep(1)
        
        # Verify the output file
        if not os.path.exists(output_path):
            raise Exception("Output file was not created")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("Output file is empty")
        
        print(f"✅ Output file created: {file_size / (1024*1024):.2f} MB")
        
        # Try to verify with OpenCV
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_fps = test_cap.get(cv2.CAP_PROP_FPS)
            test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            test_cap.release()
            
            print(f"✅ Verification successful:")
            print(f"   - Frames: {test_frames}")
            print(f"   - FPS: {test_fps}")
            print(f"   - Resolution: {test_width}x{test_height}")
            
            if test_frames == 0:
                raise Exception("Output file has 0 readable frames")
        else:
            raise Exception("Output file cannot be read by OpenCV")
        
        # Calculate final results
        crash_percentage = (crash_frames / frame_count * 100) if frame_count > 0 else 0
        final_crash_detected = crash_detected_any_frame and (crash_percentage > 5.0)
        
        print(f"✅ Processing complete:")
        print(f"   - Total frames: {frame_count}")
        print(f"   - Crash frames: {crash_frames} ({crash_percentage:.2f}%)")
        print(f"   - Crash events: {len(crash_events)}")
        print(f"   - Final decision: {'CRASH DETECTED' if final_crash_detected else 'NO CRASH'}")
        
        # Save to database
        relative_path = f'processed_videos/{output_filename}'
        video_upload.processed_video.name = relative_path
        video_upload.crash_detected = final_crash_detected
        video_upload.processing_complete = True
        video_upload.save()
        
                # ✅ ADD THIS: Send Supabase notification if crash detected
        if final_crash_detected and crash_events:
            # Extract a 10-second clip centred on the first crash event
            first_crash_time = crash_events[0]['timestamp']   # seconds into video
            clip_start = max(0.0, first_crash_time - 5.0)
            clip_end   = first_crash_time + 5.0

            clip_path = extract_clip_from_video(
                input_video_path,
                clip_start,
                clip_end,
                fps=fps
            )

            # Upload clip to Supabase Storage
            clip_url = None
            if clip_path:
                clip_url = upload_clip_to_supabase(clip_path, f'upload_{video_upload.id}')
                if clip_path and os.path.exists(clip_path):
                    os.remove(clip_path)

            send_crash_alert_to_supabase(
                accident_id=video_upload.id,
                location=f"Uploaded video: {os.path.basename(input_video_path)}",
                clip_url=clip_url
            )
        elif final_crash_detected:
            send_crash_alert_to_supabase(
                accident_id=video_upload.id,
                location=f"Uploaded video: {os.path.basename(input_video_path)}"
            )
        
    
        return {
            'crash_detected': final_crash_detected,
            'frames_processed': frame_count,
            'crash_frames': crash_frames,
            'crash_percentage': crash_percentage,
            'file_size': file_size,
            'output_path': output_path
        }
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        
        # Still mark as complete but with error
        try:
            video_upload.processing_complete = True
            video_upload.save()
        except:
            pass
            
        return None

@login_required
def process_uploaded_video(request):
    """
    Handle video upload and processing via AJAX
    """
    if request.method == 'POST' and request.FILES.get('video_file'):
        try:
            # Save the uploaded video
            video_upload = VideoUpload.objects.create(
                video_file=request.FILES['video_file']
            )
            
            # Start processing in background
            thread = threading.Thread(target=process_video_thread, args=(video_upload.id,))
            thread.daemon = True
            thread.start()
            
            return JsonResponse({
                'success': True,
                'video_id': video_upload.id,
                'message': 'Video uploaded successfully. Processing started.'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def get_video_processing_status(request, video_id):
    """
    AJAX endpoint to check video processing status
    """
    try:
        video = VideoUpload.objects.get(id=video_id)
        
        # Get the processed video URL if available
        processed_url = None
        if video.processed_video and hasattr(video.processed_video, 'url'):
            processed_url = video.processed_video.url
        
        return JsonResponse({
            'processing_complete': video.processing_complete,
            'crash_detected': video.crash_detected,
            'has_processed_video': bool(video.processed_video),
            'processed_url': processed_url,
            'video_id': video.id
        })
    except VideoUpload.DoesNotExist:
        return JsonResponse({'error': 'Video not found'}, status=404)

@login_required
def get_video_history(request):
    """
    API endpoint to get video processing history
    """
    videos = VideoUpload.objects.all().order_by('-uploaded_at')[:50]
    data = {
        'videos': [{
            'id': v.id,
            'uploaded_at': v.uploaded_at.strftime('%Y-%m-%d %H:%M'),
            'crash_detected': v.crash_detected,
            'processed_url': v.processed_video.url if v.processed_video else None
        } for v in videos]
    }
    return JsonResponse(data)

@login_required
def get_video_stats(request):
    """
    API endpoint to get video analysis statistics
    """
    total = VideoUpload.objects.count()
    crashes = VideoUpload.objects.filter(crash_detected=True).count()
    return JsonResponse({
        'total': total,
        'crashes': crashes,
        'safe': total - crashes
    })

@login_required
def download_processed_video(request, video_id):
    """
    Download processed video file with proper headers for playback
    """
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    
    if not video_upload.processed_video:
        return HttpResponse("Processed video not found in database", status=404)
    
    file_path = video_upload.processed_video.path
    print(f"📥 Download request for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return HttpResponse("Processed video file not found on server", status=404)
    
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Get the filename
        original_filename = os.path.basename(file_path)
        safe_filename = get_valid_filename(original_filename)
        
        # Ensure .mp4 extension
        if not safe_filename.lower().endswith('.mp4'):
            safe_filename += '.mp4'
        
        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Create response with the file data
        response = HttpResponse(file_data, content_type='video/mp4')
        
        # Add headers for download
        response['Content-Disposition'] = f'attachment; filename="{safe_filename}"'
        response['Content-Length'] = file_size
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        
        print(f"✅ Serving file for download: {safe_filename} ({file_size} bytes)")
        return response
        
    except Exception as e:
        print(f"❌ File response error: {e}")
        import traceback
        traceback.print_exc()
        return HttpResponse(f"Error serving file: {str(e)}", status=500)

@login_required
def view_processed_video(request, video_id):
    """
    View processed video in browser with proper streaming support
    """
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    
    if not video_upload.processed_video:
        return HttpResponse("Processed video not found", status=404)
    
    # Get the absolute URL for the video
    video_url = video_upload.processed_video.url
    print(f"👀 View request - Video URL: {video_url}")
    
    # Get file info for debugging
    file_path = video_upload.processed_video.path
    file_exists = os.path.exists(file_path)
    file_size = os.path.getsize(file_path) if file_exists else 0
    
    print(f"📁 File path: {file_path}")
    print(f"📁 File exists: {file_exists}")
    print(f"📁 File size: {file_size} bytes")
    
    return render(request, 'dashboard/video_player.html', {
        'video_upload': video_upload,
        'video_url': video_url,
        'video_id': video_id,
        'crash_detected': video_upload.crash_detected
    })

@login_required
def stream_processed_video(request, video_id):
    """
    Stream video with support for byte-range requests (for seeking in video player)
    """
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    
    if not video_upload.processed_video:
        return HttpResponse("Processed video not found", status=404)
    
    file_path = video_upload.processed_video.path
    
    if not os.path.exists(file_path):
        return HttpResponse("File not found", status=404)
    
    file_size = os.path.getsize(file_path)
    
    # Handle Range header for video seeking
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse Range header
        try:
            # Extract the range value (e.g., "bytes=0-100")
            range_value = range_header.strip().split('=')[1]
            start_byte, end_byte = range_value.split('-')
            
            start_byte = int(start_byte) if start_byte else 0
            end_byte = int(end_byte) if end_byte else file_size - 1
            
            # Ensure we don't go beyond file size
            if end_byte >= file_size:
                end_byte = file_size - 1
            
            length = end_byte - start_byte + 1
            
            # Open file and seek to start position
            with open(file_path, 'rb') as f:
                f.seek(start_byte)
                data = f.read(length)
            
            # Create partial content response
            response = HttpResponse(data, status=206, content_type='video/mp4')
            response['Content-Range'] = f'bytes {start_byte}-{end_byte}/{file_size}'
            response['Accept-Ranges'] = 'bytes'
            response['Content-Length'] = str(length)
            response['Cache-Control'] = 'no-cache'
            
            return response
            
        except Exception as e:
            print(f"Error handling range request: {e}")
            # Fall back to serving entire file
            pass
    
    # No range header or error, serve entire file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    response = HttpResponse(data, content_type='video/mp4')
    response['Accept-Ranges'] = 'bytes'
    response['Content-Length'] = file_size
    response['Cache-Control'] = 'no-cache'
    
    return response

# Helper function for safe filenames
def get_valid_filename(name):
    """Return the given string converted to a string that can be used for a clean filename."""
    import re
    s = str(name).strip().replace(' ', '_')
    s = re.sub(r'(?u)[^-\w.]', '', s)
    return s

# =============================================================================
# CCTV MONITORING FUNCTIONS
# =============================================================================

active_camera_streams = defaultdict(dict)
stream_lock = threading.Lock()
camera_ai_queues = {}
camera_latest_frame = {}

def send_cctv_crash_alert(cam_id):
    """Send crash alert to Supabase — runs in background thread"""
    try:
        from supabase import create_client
        admin_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )

        # 1. Create accident detection record
        accident_response = admin_client\
            .table('accident_detections')\
            .insert({
                'camera_id': int(cam_id),
                'detection_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'status': 'Pending'
            }).execute()

        if not accident_response.data:
            print(f"❌ Failed to create accident record for camera {cam_id}")
            return

        new_accident_id = accident_response.data[0]['accident_id']
        cam_name = active_camera_streams.get(cam_id, {}).get('camera_name', f'Camera {cam_id}')

        # 2. Get all admins
        admins = admin_client\
            .table('users')\
            .select('user_id')\
            .eq('role', 'admin')\
            .execute()

        # 3. Send alert to each admin
        if admins.data:
            alerts = [{
                'detection_id': new_accident_id,
                'sent_to': admin['user_id'],
                'message': f'🚨 Crash detected on {cam_name}',
                'response_status': 'Unacknowledged',
                'alert_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            } for admin in admins.data]

            admin_client.table('alerts').insert(alerts).execute()
            print(f"✅ Alert sent for camera {cam_id} to {len(admins.data)} admin(s)")
        else:
            print(f"⚠️ No admins found to notify for camera {cam_id}")

    except Exception as e:
        print(f"❌ Failed to send crash alert: {e}")


def ai_worker(camera_id):
    """Runs YOLO in a separate thread — never blocks the stream."""
    print(f"🤖 AI worker started for camera {camera_id}")

    DISPLAY_W = 640
    DISPLAY_H = 480
    SMALL_W   = 320
    SMALL_H   = 240
    MIN_BOX_PCT = 0.02   # 2% minimum box size
    MAX_BOX_PCT = 0.45   # 45% maximum box size
    VEHICLE_CLASSES = ['car', 'truck', 'motorcycle', 'tricycle', 'jeep']

    while active_camera_streams.get(camera_id, {}).get('active', False):

        if not active_camera_streams.get(camera_id, {}).get('ai_enabled', False):
            time.sleep(0.1)
            continue

        try:
            frame = camera_ai_queues[camera_id].get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            small_frame = cv2.resize(frame, (SMALL_W, SMALL_H))
            results = model(small_frame, imgsz=320, conf=0.50, verbose=False)

            crash_detected       = False
            detected_classes     = []
            detected_confidences = []
            detected_boxes       = []
            crash_boxes_raw      = []
            vehicle_boxes_raw    = []

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id   = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # ── Min box size check on small frame ──────────────
                    small_box_area   = (x2 - x1) * (y2 - y1)
                    small_frame_area = SMALL_W * SMALL_H
                    if small_box_area / small_frame_area < MIN_BOX_PCT:
                        continue  # too small, likely noise

                    # ── Scale coordinates to display frame ─────────────
                    x1_s = max(0, min(int(x1 * 2), DISPLAY_W - 1))
                    y1_s = max(0, min(int(y1 * 2), DISPLAY_H - 1))
                    x2_s = max(0, min(int(x2 * 2), DISPLAY_W - 1))
                    y2_s = max(0, min(int(y2 * 2), DISPLAY_H - 1))

                    # ── Max box size check on display frame ────────────
                    display_box_area = (x2_s - x1_s) * (y2_s - y1_s)
                    display_area     = DISPLAY_W * DISPLAY_H
                    if display_box_area / display_area > MAX_BOX_PCT:
                        continue  # suspiciously large, skip

                    is_crash   = any(k in class_name.lower() for k in ['vehiclecrash', 'crash', 'accident'])
                    is_vehicle = any(v in class_name.lower() for v in VEHICLE_CLASSES)

                    # ── Per-class confidence threshold ─────────────────
                    min_conf = 0.75 if is_crash else 0.50
                    if confidence < min_conf:
                        continue

                    detected_boxes.append({
                        'x1':      x1_s,
                        'y1':      y1_s,
                        'x2':      x2_s,
                        'y2':      y2_s,
                        'class':   class_name,
                        'conf':    confidence,
                        'is_crash': is_crash
                    })

                    detected_classes.append(class_name)
                    detected_confidences.append(confidence)

                    if is_crash:
                        crash_boxes_raw.append([x1, y1, x2, y2])
                    elif is_vehicle:
                        vehicle_boxes_raw.append([x1, y1, x2, y2])

            # ── Require BOTH crash AND vehicle in same frame ───────────
            crash_raw = len(crash_boxes_raw) > 0 and len(vehicle_boxes_raw) > 0

            # ── Duration-based confirmation (1.5 seconds) ──────────────
            now = time.time()
            if crash_raw:
                if not active_camera_streams[camera_id].get('crash_first_seen'):
                    active_camera_streams[camera_id]['crash_first_seen'] = now
                duration = now - active_camera_streams[camera_id]['crash_first_seen']
                crash_detected = duration >= 1.5
            else:
                active_camera_streams[camera_id]['crash_first_seen'] = None
                crash_detected = False

            # ── Update shared state ────────────────────────────────────
            with stream_lock:
                if camera_id in active_camera_streams:
                    active_camera_streams[camera_id]['last_detection'] = {
                        'crash_detected':  crash_detected,
                        'timestamp':       now,
                        'classes':         detected_classes,
                        'confidences':     detected_confidences,
                        'boxes':           detected_boxes,
                    }
                    prev = active_camera_streams[camera_id].get('frames_analyzed', 0)
                    active_camera_streams[camera_id]['frames_analyzed'] = prev + 1

            # ── Send alert (30s cooldown) ──────────────────────────────
            if crash_detected:
                last_alert        = active_camera_streams.get(camera_id, {}).get('last_alert_time', 0)
                already_capturing = camera_id in camera_post_crash_capture

                if time.time() - last_alert > 30 and not already_capturing:
                    with stream_lock:
                        if camera_id in active_camera_streams:
                            active_camera_streams[camera_id]['last_alert_time'] = time.time()

                    before_frames = []
                    if camera_id in camera_frame_buffers:
                        before_frames = [f.copy() for _, f in list(camera_frame_buffers[camera_id])]

                    print(f"🚨 Crash confirmed on camera {camera_id} — "
                          f"{len(before_frames)} pre-crash frames captured")

                    clip_thread = threading.Thread(
                        target=create_accident_and_start_capture,
                        args=(camera_id, before_frames),
                        daemon=True
                    )
                    clip_thread.start()

        except Exception as e:
            print(f"❌ AI worker error for camera {camera_id}: {e}")

    print(f"🛑 AI worker stopped for camera {camera_id}")


def generate_camera_frames(camera_id, rtsp_url):
    """Generate frames for a specific camera stream — AI runs on separate thread"""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"📹 Camera {camera_id} - Stream started, AI is OFF by default")

    # Set up the AI queue and start the AI worker thread for this camera
    camera_ai_queues[camera_id] = queue.Queue(maxsize=2)
    camera_latest_frame[camera_id] = None

    ai_thread = threading.Thread(target=ai_worker, args=(camera_id,), daemon=True)
    ai_thread.start()

    while active_camera_streams.get(camera_id, {}).get('active', False):
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Camera {camera_id}: Failed to read frame")
                time.sleep(1)
                continue

            # Resize for performance
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            # Check if AI is enabled
            ai_enabled = active_camera_streams.get(camera_id, {}).get('ai_enabled', False)

            if camera_id not in camera_frame_buffers:
                camera_frame_buffers[camera_id] = deque(maxlen=CLIP_BEFORE_FRAMES)

            # Sample every Nth frame to hit ~CLIP_FPS without storing every frame
            frame_index = active_camera_streams.get(camera_id, {}).get('frames_analyzed', 0)
            if frame_index % SAMPLE_EVERY == 0:
                camera_frame_buffers[camera_id].append((time.time(), frame.copy()))

            # ── Post-crash frame collection ────────────────────────────────
            if camera_id in camera_post_crash_capture:
                capture = camera_post_crash_capture[camera_id]
                if frame_index % SAMPLE_EVERY == 0:
                    capture['after'].append(frame.copy())

                if len(capture['after']) >= capture['target']:
                    # Enough after-frames collected — encode in background
                    before       = capture['before']
                    after        = capture['after']
                    accident_id  = capture['accident_id']
                    del camera_post_crash_capture[camera_id]

                    encode_thread = threading.Thread(
                        target=encode_and_upload_clip,
                        args=(before + after, camera_id, accident_id),
                        daemon=True
                    )
                    encode_thread.start()

            # ── Send frame to AI worker (non-blocking) ─────────────────────
            if ai_enabled:
                try:
                    camera_ai_queues[camera_id].put_nowait(frame.copy())
                except queue.Full:
                    pass
                # Get last detection result
                last_detection   = active_camera_streams.get(camera_id, {}).get('last_detection') or {}
                crash_detected   = last_detection.get('crash_detected', False)
                last_det_time    = last_detection.get('timestamp', 0)
                det_classes      = last_detection.get('classes', [])
                det_confidences  = last_detection.get('confidences', [])
                detected_boxes   = last_detection.get('boxes', [])
                frames_analyzed  = active_camera_streams.get(camera_id, {}).get('frames_analyzed', 0)
                last_alert       = active_camera_streams.get(camera_id, {}).get('last_alert_time', 0)

                # ── Draw bounding boxes ────────────────────────────────────
                for det in detected_boxes:
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    is_crash = det['is_crash']
                    label    = f"{det['class']} {det['conf']:.0%}"
                    color    = (0, 0, 255) if is_crash else (0, 255, 0)

                    # Skip if box somehow still covers too much of frame
                    box_area    = (x2 - x1) * (y2 - y1)
                    frame_area  = display_frame.shape[0] * display_frame.shape[1]
                    if box_area / frame_area > 0.45:
                        continue  # safety net — don't draw oversized boxes

                    # Label background
                    label_bg_y = max(y1 - 22, 0)
                    cv2.rectangle(display_frame,
                                (x1, label_bg_y),
                                (x1 + len(label) * 9, y1),
                                color, -1)

                    # Bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    # Label text
                    cv2.putText(display_frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # ── Red border only if crash confirmed (not just detected) ────
                if crash_detected:
                    cv2.rectangle(display_frame, (2, 2),
                                (display_frame.shape[1]-2, display_frame.shape[0]-2),
                                (0, 0, 255), 2)  # thin border, not full 4px   

                # ── Debug panel background ─────────────────────────────────
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

                # Line 1 — AI status
                if crash_detected:
                    cv2.putText(display_frame, "!! CRASH DETECTED !!", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "AI ACTIVE - Monitoring", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

                # Line 2 — what classes YOLO sees
                if det_classes:
                    classes_str = ', '.join(
                        f"{c}({conf:.0%})" for c, conf in zip(det_classes, det_confidences)
                    )
                    det_label = f"Detected: {classes_str}"
                else:
                    det_label = "Detected: nothing"
                cv2.putText(display_frame, det_label, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                # Line 3 — how long ago AI last ran
                if last_det_time:
                    age_ms = int((time.time() - last_det_time) * 1000)
                    cv2.putText(display_frame, f"Last inference: {age_ms}ms ago", (10, 72),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                else:
                    cv2.putText(display_frame, "Last inference: waiting...", (10, 72),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Line 4 — total frames analyzed
                cv2.putText(display_frame, f"Frames analyzed: {frames_analyzed}", (10, 94),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Line 5 — alert cooldown
                if last_alert:
                    cooldown_remaining = max(0, 30 - int(time.time() - last_alert))
                    cooldown_color = (0, 165, 255) if cooldown_remaining > 0 else (0, 255, 0)
                    cv2.putText(display_frame, f"Alert cooldown: {cooldown_remaining}s", (10, 116),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, cooldown_color, 1)
                else:
                    cv2.putText(display_frame, "Alert cooldown: ready", (10, 116),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # Line 6 — timestamp
                ts = time.strftime("%H:%M:%S")
                cv2.putText(display_frame, f"Time: {ts}", (10, 138),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            else:
                cv2.putText(display_frame, "AI OFF", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # Encode frame to JPEG and yield immediately — never waits for AI
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"❌ Camera {camera_id} stream error: {e}")
            time.sleep(1)

    # Cleanup
    cap.release()
    if camera_id in camera_ai_queues:
        del camera_ai_queues[camera_id]
    if camera_id in camera_latest_frame:
        del camera_latest_frame[camera_id]
    print(f"🛑 Camera {camera_id} stream stopped")

def create_accident_and_start_capture(camera_id, before_frames):
    """
    1. Creates the accident_detections record in Supabase
    2. Registers the camera for post-crash frame collection
    3. Sends initial alerts to all admins

    Called from ai_worker() in a background thread immediately on crash detection.
    The clip is encoded later by encode_and_upload_clip() once after-frames are ready.
    """
    try:
        from supabase import create_client as _create_client
        admin_client = _create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        cam_name = active_camera_streams.get(camera_id, {}).get('camera_name', f'Camera {camera_id}')

        # 1. Create accident record (no video_clip yet — will be updated after encoding)
        accident_response = admin_client.table('accident_detections').insert({
            'camera_id':      int(camera_id),
            'detection_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status':         'Pending'
        }).execute()

        if not accident_response.data:
            print(f"❌ Failed to create accident record for camera {camera_id}")
            # Still send alert without a clip
            send_cctv_crash_alert(camera_id)
            return

        accident_id = accident_response.data[0]['accident_id']
        print(f"✅ Accident record created: #{accident_id} for camera {camera_id}")

        # 2. Register for post-crash frame collection
        #    generate_camera_frames() will feed frames into capture['after']
        camera_post_crash_capture[camera_id] = {
            'before':      before_frames,
            'after':       [],
            'target':      CLIP_AFTER_FRAMES,
            'accident_id': accident_id,
        }

        # 3. Send alerts immediately (clip will arrive a few seconds later via update)
        admins = admin_client.table('users').select('user_id').eq('role', 'admin').execute()
        if admins.data:
            alerts = [{
                'detection_id':    accident_id,
                'sent_to':         admin['user_id'],
                'message':         f'🚨 Crash detected on {cam_name} — review clip generating…',
                'response_status': 'Unacknowledged',
                'alert_time':      time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            } for admin in admins.data]
            admin_client.table('alerts').insert(alerts).execute()
            print(f"✅ Alerts sent to {len(admins.data)} admin(s) for accident #{accident_id}")
        else:
            print(f"⚠️ No admins found for accident #{accident_id}")

    except Exception as e:
        print(f"❌ create_accident_and_start_capture error: {e}")
        import traceback; traceback.print_exc()
        # Fall back to old alert with no clip
        send_cctv_crash_alert(camera_id)


def encode_and_upload_clip(frames, camera_id, accident_id):
    """
    Encodes a list of OpenCV frames into an MP4, uploads to Supabase Storage
    (bucket: 'accident-clips'), then updates accident_detections.video_clip
    and the corresponding alert message.

    Called from generate_camera_frames() in a background thread once enough
    post-crash frames have been collected.
    """
    if not frames:
        print(f"⚠️ encode_and_upload_clip: no frames for accident #{accident_id}")
        return

    tmp_path = None
    try:
        print(f"🎬 Encoding {len(frames)} frames for accident #{accident_id}…")

        h, w   = frames[0].shape[:2]
        clip_filename = f'crash_{accident_id}_{uuid.uuid4().hex[:6]}.mp4'
        tmp_path = os.path.join(settings.MEDIA_ROOT, 'clips', clip_filename)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        # Try avc1 first (H.264, best browser compat), fall back to mp4v
        out = None
        for fourcc_str in ('avc1', 'H264', 'mp4v'):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out    = cv2.VideoWriter(tmp_path, fourcc, CLIP_FPS, (w, h))
            if out.isOpened():
                print(f"✅ Using codec: {fourcc_str}")
                break
            out = None

        if out is None or not out.isOpened():
            raise RuntimeError("Could not open VideoWriter with any codec")

        for frame in frames:
            out.write(frame)
        out.release()

        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            raise RuntimeError("Encoded file is empty")

        print(f"✅ Clip encoded: {file_size / 1024:.1f} KB")

        # ── Upload to Supabase Storage ──────────────────────────────────────
        from supabase import create_client as _create_client
        admin_client = _create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        storage_key = f'clips/crash_{accident_id}.mp4'

        with open(tmp_path, 'rb') as f:
            admin_client.storage.from_('accident-clips').upload(
                path=storage_key,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )

        public_url = admin_client.storage.from_('accident-clips').get_public_url(storage_key)
        print(f"✅ Clip uploaded: {public_url}")

        # ── Update accident record with clip URL ────────────────────────────
        admin_client.table('accident_detections').update({
            'video_clip': public_url
        }).eq('accident_id', accident_id).execute()

        # ── Update alert message so admins know the clip is ready ───────────
        cam_name = active_camera_streams.get(camera_id, {}).get('camera_name', f'Camera {camera_id}')
        admin_client.table('alerts').update({
            'message': f'🚨 Crash on {cam_name} — clip ready for review'
        }).eq('detection_id', accident_id).execute()

        print(f"✅ accident_detections #{accident_id} updated with video_clip URL")

    except Exception as e:
        print(f"❌ encode_and_upload_clip error for accident #{accident_id}: {e}")
        import traceback; traceback.print_exc()
    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def extract_clip_from_video(video_path, start_sec, end_sec, fps=30):
    """
    Re-opens a video file and extracts the segment [start_sec, end_sec]
    into a temporary MP4. Returns the temp file path, or None on failure.

    No ring buffer needed — we already have the source file on disk.
    """
    tmp_path = None
    cap      = None
    out      = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ extract_clip: cannot open {video_path}")
            return None

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = fps

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * actual_fps)
        end_frame   = int(end_sec   * actual_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_filename = f'upload_clip_{uuid.uuid4().hex[:8]}.mp4'
        tmp_path = os.path.join(settings.MEDIA_ROOT, 'clips', clip_filename)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        out = None
        for fourcc_str in ('avc1', 'H264', 'mp4v'):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out    = cv2.VideoWriter(tmp_path, fourcc, actual_fps, (width, height))
            if out.isOpened():
                break
            out = None

        if out is None or not out.isOpened():
            raise RuntimeError("Could not open VideoWriter")

        frames_written = 0
        current_frame  = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
            current_frame  += 1

        cap.release()
        out.release()

        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError("Output clip is empty")

        print(f"✅ Extracted clip: {frames_written} frames → {tmp_path}")
        return tmp_path

    except Exception as e:
        print(f"❌ extract_clip_from_video error: {e}")
        if cap:  cap.release()
        if out:  out.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def upload_clip_to_supabase(clip_path, label='clip'):
    """
    Uploads a local MP4 to Supabase Storage bucket 'accident-clips'.
    Returns the public URL string, or None on failure.
    """
    try:
        from supabase import create_client as _create_client
        admin_client = _create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        storage_key = f'clips/{label}_{uuid.uuid4().hex[:8]}.mp4'

        with open(clip_path, 'rb') as f:
            admin_client.storage.from_('accident-clips').upload(
                path=storage_key,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )

        public_url = admin_client.storage.from_('accident-clips').get_public_url(storage_key)
        print(f"✅ Clip uploaded to Supabase: {public_url}")
        return public_url

    except Exception as e:
        print(f"❌ upload_clip_to_supabase error: {e}")
        return None

def camera_stream(request, camera_id):
    """Stream video for a specific camera"""
    try:
        with stream_lock:
            camera_data = active_camera_streams.get(camera_id, {})
            if not camera_data.get('active', False):
                return HttpResponse("Camera not active", status=404)
        
        response = StreamingHttpResponse(
            generate_camera_frames(camera_id, camera_data['rtsp_url']),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
        
    except Exception as e:
        print(f"❌ Camera {camera_id} stream endpoint error: {e}")
        return HttpResponse("Stream error", status=500)

@login_required
@login_required
def start_camera_stream(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            rtsp_url = data.get('rtsp_url')
            camera_name = data.get('camera_name', 'Camera')
            
            with stream_lock:
                # ← If already running, just return success instead of erroring
                if camera_id in active_camera_streams and active_camera_streams[camera_id].get('active'):
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Stream already running for {camera_name}',
                        'stream_url': f'/camera-stream/{camera_id}/'
                    })
                
                active_camera_streams[camera_id] = {
                    'rtsp_url': rtsp_url,
                    'camera_name': camera_name,
                    'active': True,
                    'ai_enabled': False,
                    'last_detection': None
                }
                
                return JsonResponse({
                    'status': 'success',
                    'message': f'Stream started for {camera_name}',
                    'stream_url': f'/camera-stream/{camera_id}/'
                })
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def stop_camera_stream(request):
    """Stop video stream for a camera"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            
            with stream_lock:
                if camera_id in active_camera_streams:
                    active_camera_streams[camera_id]['active'] = False
                    del active_camera_streams[camera_id]
                    
                                # Clean up AI resources for this camera
                if camera_id in camera_ai_queues:
                    del camera_ai_queues[camera_id]
                if camera_id in camera_latest_frame:
                    del camera_latest_frame[camera_id]
                    
                    return JsonResponse({
                        'status': 'success', 
                        'message': f'Stream stopped for camera {camera_id}'
                    })
            
            return JsonResponse({'error': 'Camera stream not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def toggle_camera_ai(request):
    """Toggle AI detection for a camera"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            ai_enabled = data.get('ai_enabled')
            
            with stream_lock:
                if camera_id in active_camera_streams:
                    active_camera_streams[camera_id]['ai_enabled'] = ai_enabled
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': f'AI {"enabled" if ai_enabled else "disabled"} for camera {camera_id}'
                    })
            
            return JsonResponse({'error': 'Camera stream not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def get_camera_status(request, camera_id):
    """Get status for a specific camera"""
    with stream_lock:
        camera_data = active_camera_streams.get(camera_id, {})
    
    if camera_data:
        last_detection = camera_data.get('last_detection', {})
        crash_status = "True" if last_detection.get('crash_detected') else "False"
        ai_status = "True" if camera_data.get('ai_enabled') else "False"
        return HttpResponse(f"Crash:{crash_status}|AI:{ai_status}|Camera:{camera_id}")
    
    return HttpResponse("Crash:False|AI:False|Camera:inactive")

def reports(request):
    return render(request, 'dashboard/reports.html', {
        'SUPABASE_URL': settings.SUPABASE_URL,
        'SUPABASE_ANON_KEY': settings.SUPABASE_ANON_KEY,
    })

def send_crash_alert_to_supabase(accident_id, camera_id=None, location="Video Upload", clip_url=None):
    """Send crash alert to all admins via Supabase. Now accepts an optional clip_url."""
    try:
        from supabase import create_client as _create_client
        admin_client = _create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        accident_data = {
            'detection_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': 'Pending'
        }
        if camera_id:
            accident_data['camera_id'] = camera_id
        if clip_url:
            accident_data['video_clip'] = clip_url

        accident_response = admin_client.table('accident_detections').insert(accident_data).execute()

        if not accident_response.data:
            print("❌ Failed to create accident detection record")
            return

        new_accident_id = accident_response.data[0]['accident_id']

        admins = admin_client.table('users').select('user_id').eq('role', 'admin').execute()

        if admins.data:
            clip_note = ' — clip attached' if clip_url else ''
            alerts = [{
                'detection_id':    new_accident_id,
                'sent_to':         admin['user_id'],
                'message':         f'🚨 Crash detected in uploaded video at {location}{clip_note}',
                'response_status': 'Unacknowledged',
                'alert_time':      time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            } for admin in admins.data]

            admin_client.table('alerts').insert(alerts).execute()
            print(f"✅ Crash alert sent to {len(admins.data)} admin(s)")
        else:
            print("⚠️ No admins found to notify")

    except Exception as e:
        print(f"❌ Error sending crash alert: {e}")


def get_active_streams(request):
    """Return which cameras are currently streaming and their AI state"""
    with stream_lock:
        active = {
            cam_id: {
                'active': data.get('active', False),
                'ai_enabled': data.get('ai_enabled', False),
            }
            for cam_id, data in active_camera_streams.items()
            if data.get('active', False)
        }
    return JsonResponse({'active_streams': active})

@login_required
def mock_test_camera(request):
    """Send a mock crash alert for a specific camera"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            camera_name = data.get('camera_name')
            ip_address = data.get('ip_address')
            
            from supabase import create_client
            admin_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
            
            # Create a mock accident detection record
            accident_response = admin_client.table('accident_detections').insert({
                'camera_id': int(camera_id),
                'detection_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'status': 'Pending'
            }).execute()
            
            if not accident_response.data:
                return JsonResponse({'error': 'Failed to create mock accident record'}, status=500)
            
            new_accident_id = accident_response.data[0]['accident_id']
            
            # Get all admins
            admins = admin_client.table('users').select('user_id').eq('role', 'admin').execute()
            
            if admins.data:
                alerts = [{
                    'detection_id': new_accident_id,
                    'sent_to': admin['user_id'],
                    'message': f'🧪 [MOCK TEST] Crash detected on {camera_name} | IP: {ip_address} | Camera ID: {camera_id}',
                    'response_status': 'Unacknowledged',
                    'alert_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                } for admin in admins.data]
                
                admin_client.table('alerts').insert(alerts).execute()
                
                return JsonResponse({
                    'status': 'success',
                    'message': f'Mock test alert sent for {camera_name}',
                    'details': {
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'ip_address': ip_address,
                        'admins_notified': len(admins.data),
                        'accident_id': new_accident_id
                    }
                })
            else:
                return JsonResponse({'error': 'No admins found to notify'}, status=404)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# Add at the top with other imports
import glob

# Add these global variables near other globals
demo_video_streams = {}  # { camera_id: VideoCapture object }
demo_mode_active = {}    # { camera_id: bool }

def generate_demo_camera_frames(camera_id, video_path):
    """
    Stream a pre-recorded video file in a loop as if it were a live camera.
    Perfect for demos without actual RTSP cameras.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Demo video failed to open: {video_path}")
        return
    
    # Store in global dict for cleanup
    demo_video_streams[camera_id] = cap
    demo_mode_active[camera_id] = True
    
    print(f"🎬 Demo mode: Streaming {video_path} for camera {camera_id}")
    
    # Set up AI queue if needed
    if camera_id not in camera_ai_queues:
        camera_ai_queues[camera_id] = queue.Queue(maxsize=2)
    
    # Start AI worker if not running
    if camera_id not in [t.name for t in threading.enumerate()]:
        ai_thread = threading.Thread(
            target=ai_worker, 
            args=(camera_id,), 
            daemon=True,
            name=f"ai_worker_{camera_id}"
        )
        ai_thread.start()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while active_camera_streams.get(camera_id, {}).get('active', False):
        ret, frame = cap.read()
        
        # Loop video when it ends
        if not ret or frame_count >= total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print(f"🔄 Demo video loop restart for camera {camera_id}")
            continue
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()
        
        # Check if AI is enabled
        ai_enabled = active_camera_streams.get(camera_id, {}).get('ai_enabled', False)
        
        # Add to frame buffer for clip generation
        if camera_id not in camera_frame_buffers:
            camera_frame_buffers[camera_id] = deque(maxlen=CLIP_BEFORE_FRAMES)
        
        frame_index = active_camera_streams.get(camera_id, {}).get('frames_analyzed', 0)
        if frame_index % SAMPLE_EVERY == 0:
            camera_frame_buffers[camera_id].append((time.time(), frame.copy()))
        
        # Post-crash frame collection (same as real RTSP)
        if camera_id in camera_post_crash_capture:
            capture = camera_post_crash_capture[camera_id]
            if frame_index % SAMPLE_EVERY == 0:
                capture['after'].append(frame.copy())
            
            if len(capture['after']) >= capture['target']:
                before = capture['before']
                after = capture['after']
                accident_id = capture['accident_id']
                del camera_post_crash_capture[camera_id]
                
                encode_thread = threading.Thread(
                    target=encode_and_upload_clip,
                    args=(before + after, camera_id, accident_id),
                    daemon=True
                )
                encode_thread.start()
        
        # Send to AI worker
        if ai_enabled:
            try:
                camera_ai_queues[camera_id].put_nowait(frame.copy())
            except queue.Full:
                pass
            
            # Get detection results and draw (same as real RTSP)
            last_detection = active_camera_streams.get(camera_id, {}).get('last_detection') or {}
            crash_detected = last_detection.get('crash_detected', False)
            detected_boxes = last_detection.get('boxes', [])
            det_classes = last_detection.get('classes', [])
            det_confidences = last_detection.get('confidences', [])
            frames_analyzed = active_camera_streams.get(camera_id, {}).get('frames_analyzed', 0)
            last_alert = active_camera_streams.get(camera_id, {}).get('last_alert_time', 0)
            
            # Draw bounding boxes (no labels, as requested)
            for det in detected_boxes:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                is_crash = det['is_crash']
                color = (0, 0, 255) if is_crash else (0, 255, 0)
                
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = display_frame.shape[0] * display_frame.shape[1]
                if box_area / frame_area > 0.45:
                    continue
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            if crash_detected:
                cv2.rectangle(display_frame, (2, 2),
                            (display_frame.shape[1]-2, display_frame.shape[0]-2),
                            (0, 0, 255), 2)
            
            # Debug panel
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
            if crash_detected:
                cv2.putText(display_frame, "!! CRASH DETECTED !!", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "AI ACTIVE - Monitoring", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            
            # Add DEMO MODE watermark
            cv2.putText(display_frame, "DEMO MODE", (display_frame.shape[1]-150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            if det_classes:
                classes_str = ', '.join(f"{c}({conf:.0%})" for c, conf in zip(det_classes, det_confidences))
                cv2.putText(display_frame, f"Detected: {classes_str}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                cv2.putText(display_frame, "Detected: nothing", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            
            cv2.putText(display_frame, f"Frames analyzed: {frames_analyzed}", (10, 94),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            if last_alert:
                cooldown = max(0, 30 - int(time.time() - last_alert))
                cd_color = (0, 165, 255) if cooldown > 0 else (0, 255, 0)
                cv2.putText(display_frame, f"Alert cooldown: {cooldown}s", (10, 116),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, cd_color, 1)
            
            ts = time.strftime("%H:%M:%S")
            cv2.putText(display_frame, f"Time: {ts}", (10, 138),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        else:
            cv2.putText(display_frame, "AI OFF", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.putText(display_frame, "DEMO MODE", (display_frame.shape[1]-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Control frame rate (simulate 15fps)
        time.sleep(1/15)
    
    # Cleanup
    cap.release()
    if camera_id in demo_video_streams:
        del demo_video_streams[camera_id]
    if camera_id in demo_mode_active:
        del demo_mode_active[camera_id]
    print(f"🛑 Demo mode stopped for camera {camera_id}")

    # Add this import at the top of views.py with other imports
import subprocess

# Add these two functions anywhere before start_demo_camera_stream

def is_video_compatible(video_path):
    """
    Check if a video file can be opened and read by OpenCV.
    Returns: (is_compatible: bool, message: str)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open file"
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Cannot read frames from video"
        
        return True, "Compatible"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def convert_to_mp4(input_path):
    """
    Convert video to MP4 with H.264 codec for maximum compatibility.
    Returns path to converted file or None if conversion fails.
    Requires FFmpeg to be installed on the server.
    """
    try:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.mp4'
        
        # Skip if already converted
        if os.path.exists(output_path):
            print(f"✅ Using existing converted file: {output_path}")
            return output_path
        
        print(f"🔄 Converting {input_path} to MP4...")
        
        # FFmpeg command for fast, compatible conversion
        cmd = [
            'ffmpeg',
            '-i', input_path,           # Input file
            '-c:v', 'libx264',          # H.264 video codec
            '-preset', 'fast',          # Fast encoding
            '-crf', '23',               # Quality (lower = better, 23 is good)
            '-c:a', 'aac',              # AAC audio codec
            '-b:a', '128k',             # Audio bitrate
            '-movflags', '+faststart',  # Web-optimized
            '-y',                       # Overwrite output
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"✅ Conversion successful: {output_path}")
            return output_path
        else:
            print(f"❌ Conversion failed: {result.stderr.decode()}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timeout for {input_path}")
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None


@login_required
def start_demo_camera_stream(request):
    """
    Start a demo stream using a video file instead of RTSP.
    Automatically assigns different videos to different cameras.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            camera_name = data.get('camera_name', 'Demo Camera')
            video_file = data.get('video_file')  # Optional: specify exact video
            
            if not video_file:
                demo_dir = os.path.join(settings.MEDIA_ROOT, 'demo_videos')
                os.makedirs(demo_dir, exist_ok=True)
                
                # Find all video files
                video_files = (
                    glob.glob(os.path.join(demo_dir, '*.mp4')) +
                    glob.glob(os.path.join(demo_dir, '*.avi')) +
                    glob.glob(os.path.join(demo_dir, '*.mov')) +
                    glob.glob(os.path.join(demo_dir, '*.mkv'))
                )
                
                # Sort for consistent order
                video_files.sort()
                
                if not video_files:
                    return JsonResponse({
                        'error': 'No demo videos found. Add video files to media/demo_videos/'
                    }, status=400)
                
                # ✅ ASSIGN DIFFERENT VIDEOS TO DIFFERENT CAMERAS
                # Use camera_id as index (with modulo to cycle through videos)
                try:
                    camera_index = int(camera_id) - 1  # Assuming IDs start at 1
                except (ValueError, TypeError):
                    camera_index = hash(str(camera_id))  # Fallback for non-numeric IDs
                
                video_index = camera_index % len(video_files)
                video_file = video_files[video_index]
                
                print(f"📹 Camera {camera_id} assigned video #{video_index + 1}/{len(video_files)}: {os.path.basename(video_file)}")
            
            if not os.path.exists(video_file):
                return JsonResponse({
                    'error': f'Demo video not found: {video_file}'
                }, status=400)
            
            # Check compatibility
            is_ok, msg = is_video_compatible(video_file)
            if not is_ok:
                print(f"⚠️ Video incompatible: {msg}. Attempting conversion...")
                converted_file = convert_to_mp4(video_file)
                if converted_file:
                    video_file = converted_file
                else:
                    return JsonResponse({
                        'error': f'Video incompatible and conversion failed: {msg}'
                    }, status=400)
            
            with stream_lock:
                if camera_id in active_camera_streams and active_camera_streams[camera_id].get('active'):
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Demo stream already running for {camera_name}',
                        'stream_url': f'/demo-camera-stream/{camera_id}/',
                        'demo_mode': True
                    })
                
                active_camera_streams[camera_id] = {
                    'rtsp_url': video_file,
                    'camera_name': camera_name,
                    'active': True,
                    'ai_enabled': False,
                    'last_detection': None,
                    'demo_mode': True
                }
                return JsonResponse({
                    'status': 'success',
                    'message': f'Demo stream started for {camera_name}',
                    'stream_url': f'/demo-camera-stream/{camera_id}/',
                    'demo_mode': True,
                    'video_file': os.path.basename(video_file)
                })
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def demo_camera_stream(request, camera_id):
    """Stream endpoint for demo mode"""
    try:
        with stream_lock:
            camera_data = active_camera_streams.get(camera_id, {})
            if not camera_data.get('active', False):
                return HttpResponse("Camera not active", status=404)
            
            video_path = camera_data.get('rtsp_url')  # It's actually a file path in demo mode
        
        response = StreamingHttpResponse(
            generate_demo_camera_frames(camera_id, video_path),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
        
    except Exception as e:
        print(f"❌ Demo camera {camera_id} stream error: {e}")
        return HttpResponse("Stream error", status=500)