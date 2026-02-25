import os
import cv2
import time
import threading
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.conf import settings
from django.conf import settings as django_settings  
from django.http import HttpResponse, FileResponse, StreamingHttpResponse
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
                "user_id": supabase_user.get('id'),  # Fallback
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
    return render(request, 'dashboard/cctv_monitoring.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
    })

@login_required
def live_monitoring(request):
    return render(request, 'dashboard/live_monitoring.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
    })

@login_required
def reports(request):
    return render(request, 'dashboard/reports.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
    })

@login_required
def site_settings(request):
    return render(request, 'dashboard/settings.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_ANON_KEY": settings.SUPABASE_ANON_KEY,
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

@login_required
def live_monitoring(request):
    """Live monitoring dashboard with dynamic RTSP URL input"""
    global current_rtsp_url
    
    # Get current RTSP URL from session if available
    if not current_rtsp_url:
        current_rtsp_url = request.session.get('current_rtsp_url', '')
    
    return render(request, 'dashboard/live_monitoring.html', {
        'RTSP_URL': current_rtsp_url
    })

# =============================================================================
# EXISTING VIDEO PROCESSING FUNCTIONS (UNCHANGED)
# =============================================================================

def process_video(video_upload):
    """Process uploaded video with YOLOv8 model for crash detection"""
    CRASH_CONFIDENCE_THRESHOLD = 0.60
    
    try:
        input_video_path = video_upload.video_file.path
        
        # Create output path with proper extension
        original_name = os.path.basename(input_video_path)
        name_without_ext = os.path.splitext(original_name)[0]
        output_filename = f'processed_{name_without_ext}.mp4'  # Force .mp4 extension
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', output_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Handle invalid FPS
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Processing video: {width}x{height} at {fps} FPS")
        
        # Use MP4V codec for maximum compatibility (works on most systems)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Failed to create output video writer")
            cap.release()
            return None
        
        crash_detected_any_frame = False
        frame_count = 0
        crash_frames = 0
        total_frames = 0
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, conf=CRASH_CONFIDENCE_THRESHOLD)
            
            frame_has_crash = False
            
            # Check if crash is detected
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        crash_keywords = ['crash', 'accident']
                        
                        if any(keyword in class_name.lower() for keyword in crash_keywords):
                            frame_has_crash = True
                            print(f"Crash detected in frame {frame_count}: {class_name} with confidence: {confidence:.2f}")
            
            if frame_has_crash:
                crash_frames += 1
            
            total_frames += 1
            
            # Visualize results
            annotated_frame = results[0].plot()
            
            # Add detection info
            if frame_has_crash:
                cv2.putText(annotated_frame, "CRASH DETECTED!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                crash_detected_any_frame = True
            
            # Add processing info to frame
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write the frame to output video
            out.write(annotated_frame)
            frame_count += 1
            
            # Progress update every 100 frames
            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count} frames...")
        
        # Release everything
        cap.release()
        out.release()
        
        # Final decision
        final_crash_detected = crash_detected_any_frame and (crash_frames / total_frames > 0.05) if total_frames > 0 else False
        
        print(f"✅ Processing complete: {crash_frames}/{total_frames} frames had crash detection")
        print(f"🎯 Final crash decision: {final_crash_detected}")
        print(f"💾 Output saved to: {output_path}")
        
        # Verify the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📁 Output file size: {file_size:.2f} MB")
        else:
            print("❌ Output file was not created!")
            return None
        
        # Save processed video to model
        video_upload.processed_video.name = f'processed_videos/{output_filename}'
        video_upload.crash_detected = final_crash_detected
        video_upload.processing_complete = True
        video_upload.save()
        
        return {
            'crash_detected': final_crash_detected,
            'frames_processed': frame_count,
            'crash_frames': crash_frames
        }
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        video_upload.processing_complete = True
        video_upload.save()
        return None

def home(request):
    """Home page with video upload form"""
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_upload = form.save()
            
            # Process video
            result = process_video(video_upload)
            
            if result:
                # Get the actual URL for the processed video
                if video_upload.processed_video:
                    processed_video_url = video_upload.processed_video.url
                    print(f"🌐 Processed video URL: {processed_video_url}")
                else:
                    processed_video_url = None
                    print("❌ No processed video URL available")
                
                return render(request, 'dashboard/result.html', {
                    'result': result, 
                    'video_upload': video_upload,
                    'crash_detected': result['crash_detected'],
                    'processed_video_url': processed_video_url
                })
            else:
                return render(request, 'dashboard/upload.html', {'form': form, 'error': 'Processing failed. Check console for details.'})
    else:
        form = VideoUploadForm()
    
    return render(request, 'dashboard/upload.html', {'form': form})

def download_processed_video(request, video_id):
    """Download processed video file"""
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    
    if video_upload.processed_video:
        file_path = video_upload.processed_video.path
        
        print(f"📥 Download request for: {file_path}")
        
        if os.path.exists(file_path):
            # Get safe filename for download
            original_filename = os.path.basename(file_path)
            safe_filename = get_valid_filename(original_filename)
            
            # Ensure it has .mp4 extension
            if not safe_filename.lower().endswith('.mp4'):
                safe_filename += '.mp4'
            
            print(f"📄 Serving file: {file_path} as {safe_filename}")
            
            try:
                response = FileResponse(
                    open(file_path, 'rb'),
                    content_type='video/mp4'
                )
                response['Content-Disposition'] = f'attachment; filename="{safe_filename}"'
                response['Content-Length'] = os.path.getsize(file_path)
                return response
            except Exception as e:
                print(f"❌ File response error: {e}")
                return HttpResponse("Error serving file", status=500)
        else:
            print(f"❌ File not found: {file_path}")
            return HttpResponse("Processed video file not found on server", status=404)
    
    return HttpResponse("Processed video not found in database", status=404)

def view_processed_video(request, video_id):
    """View processed video in browser"""
    video_upload = get_object_or_404(VideoUpload, id=video_id)
    
    if video_upload.processed_video:
        video_url = video_upload.processed_video.url
        print(f"👀 View request - Video URL: {video_url}")
        
        return render(request, 'dashboard/video_player.html', {
            'video_upload': video_upload,
            'video_url': video_url
        })
    
    return HttpResponse("Processed video not found", status=404)

# Helper function for safe filenames
def get_valid_filename(name):
    """Return the given string converted to a string that can be used for a clean filename."""
    import re
    s = str(name).strip().replace(' ', '_')
    s = re.sub(r'(?u)[^-\w.]', '', s)
    return s

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
            display_frame = frame.copy()  # Frame to display (may be annotated)
            
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
                            # Scale coordinates
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            
                            # Draw bounding box on display frame
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "🚨 CRASH", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update detection status
                with stream_lock:
                    if camera_id in active_camera_streams:
                        active_camera_streams[camera_id]['last_detection'] = {
                            'crash_detected': crash_detected,
                            'timestamp': time.time()
                        }
                
                # Add AI status text
                status_text = "🚨 CRASH DETECTED!" if crash_detected else "🤖 AI ACTIVE"
                status_color = (0, 0, 255) if crash_detected else (0, 255, 0)
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                # AI is OFF - just show plain video
                # Optional: add a small "AI OFF" indicator
                cv2.putText(display_frame, "⚫ AI OFF", (10, 30),
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
def start_camera_stream(request):
    """Start video stream for a camera"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            rtsp_url = data.get('rtsp_url')
            camera_name = data.get('camera_name', 'Camera')
            
            with stream_lock:
                if camera_id in active_camera_streams:
                    return JsonResponse({'error': 'Camera stream already running'}, status=400)
                
                # Start camera stream with AI explicitly OFF
                active_camera_streams[camera_id] = {
                    'rtsp_url': rtsp_url,
                    'camera_name': camera_name,
                    'active': True,
                    'ai_enabled': False,  # ← Explicitly set to False
                    'last_detection': None
                }
                
                print(f"🎥 Started stream for {camera_name} ({camera_id}) - AI: OFF")
                
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

@login_required
def toggle_camera_ai(request):
    """Toggle AI detection for a camera"""
    if request.method == 'POST':
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

