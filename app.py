print("\n=== STARTING APP ===")

# ============================
#   ENVIRONMENT SETTINGS
# ============================
import os
os.environ["ULTRALYTICS_HUB"]   = "False"
os.environ["ULTRALYTICS_CHECK"] = "False"
os.environ["WANDB_DISABLED"]    = "true"
os.environ["YOLO_VERBOSE"]      = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_DEVICE"]      = "cpu"

import cv2
from turbojpeg import TurboJPEG
import torch
from ultralytics import YOLO
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, send_file
import threading
import time
import datetime
import io
import collections

app = Flask(__name__)

# ============================
#   CONFIG
# ============================
RTSP_URL     = "rtsp://192.168.2.10:554/stream1"
FRAME_W      = 1280
FRAME_H      = 720
JPEG_QUALITY = 82

jpeg = TurboJPEG()

latest_frame         = None
latest_annotated     = None
frame_lock           = threading.Lock()

alarm_box_enabled          = False
alarm_box                  = None
alarm_sound                = "default.wav"
show_alarm_box             = True
movement_detection_enabled = True
yolo_boxes_enabled         = True

detection_log      = collections.deque(maxlen=50)
detection_log_lock = threading.Lock()

stats = {
    "fps": 0,
    "detections_today": 0,
    "last_trigger": None,
    "alarm_active": False,
    "connected": False,
}

ALARM_FOLDER    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarms")
SNAPSHOT_FOLDER = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(ALARM_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

# ============================
#   CAMERA THREAD
# ============================
_fps_counter   = 0
_fps_last_time = time.time()

def camera_reader():
    global latest_frame, _fps_counter, _fps_last_time
    cap = None
    while True:
        try:
            if cap is None or not cap.isOpened():
                stats["connected"] = False
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)
            ret, frame = cap.read()
            if ret:
                stats["connected"] = True
                with frame_lock:
                    latest_frame = frame
                _fps_counter += 1
                now = time.time()
                if now - _fps_last_time >= 1.0:
                    stats["fps"] = _fps_counter
                    _fps_counter = 0
                    _fps_last_time = now
            else:
                stats["connected"] = False
                cap.release()
                cap = None
        except Exception:
            stats["connected"] = False
            time.sleep(1)

threading.Thread(target=camera_reader, daemon=True).start()

# ============================
#   ALARM
# ============================
alarm_cooldown = 0

def trigger_alarm():
    global alarm_cooldown
    now = time.time()
    if now < alarm_cooldown:
        return
    alarm_path = os.path.join(ALARM_FOLDER, alarm_sound)
    if os.path.exists(alarm_path):
        os.system(f"ffplay -nodisp -autoexit -loglevel quiet '{alarm_path}' &")
    alarm_cooldown = now + 30
    stats["last_trigger"] = datetime.datetime.now().strftime("%H:%M:%S")
    stats["alarm_active"] = True
    threading.Timer(5, lambda: stats.update({"alarm_active": False})).start()

# ============================
#   FRAME GENERATOR
# ============================
CLASS_COLORS = {
    "person":  (0, 220, 120),
    "car":     (0, 160, 255),
    "truck":   (255, 160, 0),
    "bicycle": (200, 80, 255),
    "dog":     (255, 220, 0),
}
DEFAULT_COLOR = (180, 180, 180)
prev_person_centers = []

def draw_box(frame, x1, y1, x2, y2, label, color):
    L = 16
    t = 2
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx*L, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy*L), color, t)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

def generate_frames():
    global latest_annotated, prev_person_centers
    while True:
        with frame_lock:
            raw = latest_frame
        if raw is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(raw, (FRAME_W, FRAME_H))
        current_centers = []

        if alarm_box_enabled and movement_detection_enabled and alarm_box is not None:
            results = model(frame, conf=0.20, iou=0.45, verbose=False)[0]
            bx1, by1, bx2, by2 = alarm_box
            for box in results.boxes:
                cls  = int(box.cls[0])
                name = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(name, DEFAULT_COLOR)
                if yolo_boxes_enabled:
                    draw_box(frame, x1, y1, x2, y2, f"{name} {conf:.0%}", color)
                if name not in ["person", "car"]:
                    continue
                # Trigger if the detected box overlaps the zone at all (not just fully inside)
                if not (x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1):
                    continue
                current_centers.append(((x1+x2)//2, (y1+y2)//2))

            # Trigger only if a person/car inside the box has MOVED since last frame
            moved = (
                len(current_centers) > 0 and
                len(prev_person_centers) > 0 and
                any(
                    min(np.hypot(cx-px, cy-py) for px,py in prev_person_centers) > 10
                    for cx, cy in current_centers
                )
            )
            prev_person_centers = current_centers
            if moved:
                trigger_alarm()
                stats["detections_today"] += 1
                with detection_log_lock:
                    detection_log.appendleft({
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "objects": len(current_centers),
                        "type": "person/vehicle",
                    })
        else:
            prev_person_centers = []

        # Alarm box
        if alarm_box is not None and show_alarm_box:
            bx1,by1,bx2,by2 = alarm_box
            cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,60,255),1)
            for px,py in [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)]:
                cv2.circle(frame,(px,py),5,(0,80,255),-1)
            cv2.putText(frame,"WATCH ZONE",(bx1+6,by1+18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,120,255),1,cv2.LINE_AA)

        # Timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.rectangle(frame,(0,FRAME_H-28),(290,FRAME_H),(0,0,0),-1)
        cv2.putText(frame,ts,(8,FRAME_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(160,255,160),1,cv2.LINE_AA)

        # FPS
        cv2.rectangle(frame,(0,0),(80,24),(0,0,0),-1)
        cv2.putText(frame,f"{stats['fps']} FPS",(6,17),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,200,255),1,cv2.LINE_AA)

        # Alarm flash
        if stats["alarm_active"]:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(FRAME_W,FRAME_H),(0,0,180),-1)
            cv2.addWeighted(ov,0.13,frame,0.87,0,frame)
            cv2.putText(frame,"!  MOTION DETECTED  !",(FRAME_W//2-210,54),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),2,cv2.LINE_AA)

        with frame_lock:
            latest_annotated = frame.copy()

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.encode(frame, quality=JPEG_QUALITY) + b"\r\n"

# ============================
#   ROUTES
# ============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot")
def snapshot():
    with frame_lock:
        frame = latest_annotated if latest_annotated is not None else latest_frame
    if frame is None:
        return "No frame", 503
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_path = os.path.join(SNAPSHOT_FOLDER, f"snap_{ts}.jpg")
    with open(snap_path,"wb") as f: f.write(buf.tobytes())
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg",
                     as_attachment=True, download_name=f"snapshot_{ts}.jpg")

@app.route("/status")
def status():
    return jsonify({**stats,
        "alarm_box_enabled": alarm_box_enabled,
        "yolo_boxes_enabled": yolo_boxes_enabled,
        "movement_detection_enabled": movement_detection_enabled,
        "show_alarm_box": show_alarm_box,
        "alarm_sound": alarm_sound,
        "resolution": f"{FRAME_W}x{FRAME_H}",
    })

@app.route("/detection_log")
def get_detection_log():
    with detection_log_lock:
        return jsonify(list(detection_log))

@app.route("/save_box", methods=["POST"])
def save_box():
    global alarm_box
    data = request.get_json()
    nx1,ny1,nx2,ny2 = data.get("box_norm")
    alarm_box = [int(nx1*FRAME_W),int(ny1*FRAME_H),int(nx2*FRAME_W),int(ny2*FRAME_H)]
    return jsonify({"status":"ok"})

@app.route("/toggle_yolo_boxes", methods=["POST"])
def toggle_yolo_boxes():
    global yolo_boxes_enabled
    yolo_boxes_enabled = not yolo_boxes_enabled
    return jsonify({"enabled":yolo_boxes_enabled})

@app.route("/toggle_alarm_box", methods=["POST"])
def toggle_alarm_box():
    global alarm_box_enabled
    alarm_box_enabled = not alarm_box_enabled
    return jsonify({"enabled":alarm_box_enabled})

@app.route("/list_alarms")
def list_alarms():
    files = [f for f in os.listdir(ALARM_FOLDER) if f.endswith((".wav",".mp3"))]
    return jsonify(files)

@app.route("/set_alarm", methods=["POST"])
def set_alarm():
    global alarm_sound
    data = request.get_json()
    alarm_sound = data.get("filename")
    return jsonify({"status":"ok"})

@app.route("/toggle_box_visibility", methods=["POST"])
def toggle_box_visibility():
    global show_alarm_box
    show_alarm_box = not show_alarm_box
    return jsonify({"visible":show_alarm_box})

@app.route("/toggle_movement_detection", methods=["POST"])
def toggle_movement_detection():
    global movement_detection_enabled
    movement_detection_enabled = not movement_detection_enabled
    return jsonify({"enabled":movement_detection_enabled})

if __name__ == "__main__":
    print("=== SERVER STARTING ===")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
