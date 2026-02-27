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
# DYNAMIC RTSP MONITORING FUNCTIONS
# =============================================================================

def generate_frames():
    """Generate frames directly from RTSP stream"""
    global is_monitoring, latest_detection_result, current_rtsp_url
    
    if not current_rtsp_url:
        print("❌ No RTSP URL configured")
        return
    
    cap = cv2.VideoCapture(current_rtsp_url)
    
    # Low latency settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print(f"🎥 Starting RTSP stream capture: {current_rtsp_url}")
    frame_count = 0
    
    while is_monitoring:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from RTSP")
                time.sleep(1)
                continue
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO detection
            start_time = time.time()
            results = model(frame, imgsz=320, conf=0.5, verbose=False)
            
            # Check for crashes
            crash_detected = False
            crash_classes = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if any(keyword in class_name.lower() for keyword in ['crash', 'accident']):
                            crash_detected = True
                            crash_classes.append(class_name)
                            print(f"🚨 Crash detected: {class_name} ({confidence:.2f})")
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Add status text
            status_text = "🚨 CRASH DETECTED!" if crash_detected else "✅ Monitoring"
            color = (0, 0, 255) if crash_detected else (0, 255, 0)
            cv2.putText(annotated_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add latency info
            latency = int((time.time() - start_time) * 1000)
            cv2.putText(annotated_frame, f"Latency: {latency}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update detection result
            latest_detection_result = {
                'crash_detected': crash_detected,
                'crash_classes': crash_classes,
                'timestamp': time.time(),
                'latency_ms': latency,
                'frame_count': frame_count
            }
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80
            ])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"📊 Frames processed: {frame_count}, Latency: {latency}ms")
            
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
        if final_crash_detected:
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
# CAMERA STREAM FUNCTIONS
# =============================================================================

active_camera_streams = defaultdict(dict)
stream_lock = threading.Lock()

def generate_camera_frames(camera_id, rtsp_url):
    """Generate frames for a specific camera stream"""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"📹 Camera {camera_id} - Stream started, AI is OFF by default")
    
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
            
            # Check if AI is enabled for this camera
            ai_enabled = active_camera_streams.get(camera_id, {}).get('ai_enabled', False)
            
            # ONLY run AI if explicitly enabled
            if ai_enabled:
                # Run YOLO detection on a smaller frame for speed
                small_frame = cv2.resize(frame, (320, 240))
                results = model(small_frame, imgsz=320, conf=0.5, verbose=False)
                
                # Check for crashes
                crash_detected = False
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        if any(keyword in class_name.lower() for keyword in ['crash', 'accident']):
                            crash_detected = True
                            break
                    
                    # Scale boxes back to original frame size
                    if crash_detected:
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 240
                        
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "CRASH", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update detection status
                with stream_lock:
                    if camera_id in active_camera_streams:
                        active_camera_streams[camera_id]['last_detection'] = {
                            'crash_detected': crash_detected,
                            'timestamp': time.time()
                        }
                
                # ✅ SEND SUPABASE ALERT IF CRASH DETECTED (with 30s cooldown)
                if crash_detected:
                    last_alert = active_camera_streams.get(camera_id, {}).get('last_alert_time', 0)
                    
                    if time.time() - last_alert > 30:
                        # Update cooldown immediately to prevent duplicate alerts
                        with stream_lock:
                            if camera_id in active_camera_streams:
                                active_camera_streams[camera_id]['last_alert_time'] = time.time()
                        
                        # Run in background so it doesn't block the video stream
                        def send_cctv_crash_alert(cam_id):
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
                                    
                            except Exception as e:
                                print(f"❌ Failed to send crash alert: {e}")
                        
                        alert_thread = threading.Thread(target=send_cctv_crash_alert, args=(camera_id,))
                        alert_thread.daemon = True
                        alert_thread.start()
                
                # Add AI status text
                status_text = "CRASH DETECTED!" if crash_detected else "AI ACTIVE"
                status_color = (0, 0, 255) if crash_detected else (0, 255, 0)
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                cv2.putText(display_frame, "AI OFF", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"❌ Camera {camera_id} stream error: {e}")
            time.sleep(1)
    
    cap.release()
    print(f"🛑 Camera {camera_id} stream stopped")

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

# At the top of views.py, add this helper function
def send_crash_alert_to_supabase(accident_id, camera_id=None, location="Video Upload"):
    """Send crash alert to all admins via Supabase"""
    try:
        from supabase import create_client
        admin_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        
        # Create accident detection record
        accident_data = {
            'detection_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': 'Pending'
        }
        if camera_id:
            accident_data['camera_id'] = camera_id
            
        accident_response = admin_client.table('accident_detections').insert(accident_data).execute()
        
        if not accident_response.data:
            print("❌ Failed to create accident detection record")
            return
            
        new_accident_id = accident_response.data[0]['accident_id']
        
        # Get all admins
        admins = admin_client.table('users').select('user_id').eq('role', 'admin').execute()
        
        if admins.data:
            alerts = [{
                'detection_id': new_accident_id,
                'sent_to': admin['user_id'],
                'message': f'🚨 Crash detected in uploaded video at {location}',
                'response_status': 'Unacknowledged',
                'alert_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
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