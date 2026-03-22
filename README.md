# 🎥 SecureView — AI-Powered Security Camera Dashboard

A self-hosted, browser-based security camera system powered by **YOLOv8** object detection. Streams live RTSP footage, detects motion in user-defined watch zones, triggers audio alarms, records video, and serves a real-time dashboard — all from a single Python file.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Using the Dashboard](#-using-the-dashboard)
- [Folder Structure](#-folder-structure)
- [API Reference](#-api-reference)
- [Adding Alarm Sounds](#-adding-alarm-sounds)
- [Running on Boot (systemd)](#-running-on-boot-systemd)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)

---

## ✨ Features

- 📡 **Live RTSP stream** displayed in the browser via MJPEG
- 🤖 **YOLOv8 object detection** — detects people, cars, trucks, bicycles, dogs
- 🟧 **Draw custom watch zones** — drag a rectangle on the video, only triggers inside that area
- 🔔 **Audio alarm** — plays a `.wav` or `.mp3` when motion is detected (30-second cooldown)
- ⏺️ **Video recording** — start/stop recording directly from the UI, saved as `.mp4`
- 📷 **Snapshots** — save annotated JPEG frames instantly
- 📊 **Live stats** — FPS counter, detection count, last trigger time, connection status
- 📋 **Detection log** — scrollable history of recent detections with timestamps
- 🌐 **Fully browser-based** — no app install needed, works on phone, tablet, desktop
- 🎨 **Military-style dark UI** — warm amber HUD aesthetic with Bebas Neue + IBM Plex Mono

---

## 🧠 How It Works

```
RTSP Camera → camera_reader() thread → latest_frame (shared memory)
                                              ↓
                                     generate_frames() loop
                                              ↓
                              resize → YOLOv8 inference (if alarm enabled)
                                              ↓
                         check if detected box overlaps watch zone
                                              ↓
                        compare centers with previous frame (movement check)
                                              ↓
                   trigger_alarm() + log entry + write to VideoWriter
                                              ↓
                         MJPEG stream → browser <img> tag
```

- The camera is read in a **dedicated background thread** so frame capture never blocks the web server.
- **YOLOv8 inference runs only when the alarm zone is enabled**, saving CPU.
- **Movement detection** prevents false triggers from a person standing still — the detected center point must move more than 10 pixels between frames.
- The alarm has a **30-second cooldown** to prevent alarm spam.
- Recording writes the **annotated frame** (with boxes, timestamp, overlays) directly to an MP4 file.

---

## 📦 Requirements

### System

| Requirement | Notes |
|---|---|
| Python 3.9 or later | 3.10+ recommended |
| `ffplay` (FFmpeg) | Used to play alarm sounds |
| A working RTSP camera | Any IP camera with an RTSP stream |
| Linux or macOS | Windows works but `ffplay` command may differ |

### Python packages

```
flask
opencv-python
pyturbojpeg
torch
ultralytics
numpy
```

> **Note:** This app runs on **CPU only** by default (`CUDA_VISIBLE_DEVICES=-1`). If you have a GPU and want to use it, remove that line from the environment settings at the top of `app.py`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/secureview.git
cd secureview
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install flask opencv-python pyturbojpeg torch torchvision ultralytics numpy
```

> If `pyturbojpeg` fails to install, install the system dependency first:
> ```bash
> # Ubuntu/Debian
> sudo apt-get install libturbojpeg0-dev
>
> # macOS
> brew install jpeg-turbo
> ```

### 4. Install FFmpeg (for alarm sounds)

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 5. Download YOLOv8 model

The app downloads `yolov8n.pt` automatically on first run. To pre-download it:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Available model sizes (tradeoff: speed vs accuracy):

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fastest | Basic |
| `yolov8s.pt` | 22 MB | ⚡⚡ Fast | Good |
| `yolov8m.pt` | 50 MB | ⚡ Medium | Better |
| `yolov8l.pt` | 87 MB | 🐢 Slow | Best |

To use a different model, change this line in `app.py`:
```python
model = YOLO("yolov8n.pt")  # change to yolov8s.pt, yolov8m.pt, etc.
```

---

## ⚙️ Configuration

Open `app.py` and edit the **CONFIG** section near the top:

```python
# ============================
#   CONFIG
# ============================
RTSP_URL     = "rtsp://192.168.2.10:554/stream1"   # ← Your camera's RTSP URL
FRAME_W      = 1280                                  # ← Output resolution width
FRAME_H      = 720                                   # ← Output resolution height
JPEG_QUALITY = 82                                    # ← Stream quality (1-100)
```

### Finding your RTSP URL

Most IP cameras follow one of these formats:

```
rtsp://username:password@192.168.x.x:554/stream1
rtsp://192.168.x.x:554/Streaming/Channels/101
rtsp://192.168.x.x/live/ch00_0
```

Check your camera's manual or try the brand-specific format:

| Brand | Common RTSP URL format |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Amcrest | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@IP/axis-media/media.amp` |
| Generic ONVIF | `rtsp://user:pass@IP:554/stream1` |

### Adjusting detection sensitivity

In `generate_frames()` inside `app.py`:

```python
results = model(frame, conf=0.20, iou=0.45, verbose=False)[0]
```

- `conf=0.20` — confidence threshold (lower = more detections, more false positives)
- `iou=0.45` — overlap threshold for non-max suppression

For fewer false positives, raise `conf` to `0.40` or higher.

### Adjusting movement sensitivity

```python
min(np.hypot(cx-px, cy-py) for px,py in prev_person_centers) > 10
```

Change `10` to a higher number (e.g. `25`) to require more movement before triggering.

### Alarm cooldown

```python
alarm_cooldown = now + 30   # seconds between alarm re-triggers
```

Change `30` to however many seconds you want between alarm sounds.

---

## ▶️ Running the App

```bash
python3 app.py
```

You should see:

```
=== STARTING APP ===
=== SERVER STARTING ===
```

Then open your browser and go to:

```
http://localhost:5000
```

Or from another device on the same network:

```
http://YOUR_MACHINE_IP:5000
```

To find your machine's IP:
```bash
# Linux/macOS
ip addr show   # or: hostname -I

# Windows
ipconfig
```

---

## 🖥️ Using the Dashboard

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  HEADER — logo, connection status, clock                │
├──────────────┬──────────────────────────┬───────────────┤
│  LEFT PANEL  │                          │  RIGHT PANEL  │
│              │                          │               │
│  Controls    │      LIVE VIDEO FEED     │  Live Stats   │
│  Toggles     │                          │               │
│  Watch Zone  │                          │  Recordings   │
│  Alarm Sound │                          │  List         │
│  Log         │                          │               │
├──────────────┴──────────────────────────┴───────────────┤
│  FOOTER — Alarm | Motion | Boxes | Record | Snap | Full │
└─────────────────────────────────────────────────────────┘
```

### Step-by-step: Setting up motion detection

1. **Enable the alarm zone** — toggle "Alarm Zone" ON in the left panel
2. **Draw a watch zone** — click **"⊕ Draw Zone"**, then click and drag on the video to draw a rectangle over the area you want to monitor
3. **Enable Motion Detect** — make sure "Motion Detect" toggle is ON
4. **Select an alarm sound** — pick a `.wav` or `.mp3` from the dropdown (add files to the `alarms/` folder)
5. That's it — when a person or vehicle enters your zone and moves, the alarm will trigger

### Controls reference

| Control | What it does |
|---|---|
| **Alarm Zone** toggle | Arms/disarms the entire detection system |
| **Motion Detect** toggle | Enables/disables the movement check (if off, any overlap triggers) |
| **YOLO Boxes** toggle | Shows/hides the colored detection boxes on the stream |
| **Show Zone** toggle | Shows/hides the orange watch zone rectangle on the stream |
| **⊕ Draw Zone** button | Enter zone-drawing mode — click & drag on the video |
| **⏺ Record** footer button | Start/stop video recording to `recordings/` |
| **📷 Snapshot** footer button | Save the current annotated frame as a JPEG |
| **⛶ Fullscreen** footer button | Enter fullscreen mode for the video panel |

### Recordings panel

- Lists all `.mp4` files in the `recordings/` folder
- Shows file size in MB
- **DL** button — downloads the file to your computer
- **DEL** button — permanently deletes the recording (cannot delete an active recording)
- Refreshes automatically every 30 seconds, or click **↺ REFRESH** manually

---

## 📁 Folder Structure

```
secureview/
│
├── app.py                  # Main application — all backend logic
├── templates/
│   └── index.html          # Dashboard frontend (single-page)
│
├── alarms/                 # Put your alarm sound files here
│   └── default.wav         # Default alarm sound
│
├── snapshots/              # Saved snapshot JPEGs (auto-created)
│   └── snap_20240115_143022.jpg
│
├── recordings/             # Saved video recordings (auto-created)
│   └── rec_20240115_143500.mp4
│
└── yolov8n.pt              # YOLOv8 model weights (auto-downloaded)
```

All three output folders (`alarms/`, `snapshots/`, `recordings/`) are created automatically when the app starts if they don't exist.

---

## 🌐 API Reference

The backend exposes a simple REST API. All endpoints return JSON unless noted.

### Status & Data

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard HTML |
| `GET` | `/video` | MJPEG stream (used by the `<img>` tag) |
| `GET` | `/status` | Returns full system status (see below) |
| `GET` | `/detection_log` | Returns last 50 detection events |

**`/status` response:**
```json
{
  "fps": 24,
  "detections_today": 3,
  "last_trigger": "14:32:01",
  "alarm_active": false,
  "connected": true,
  "recording": false,
  "recording_file": null,
  "alarm_box_enabled": true,
  "yolo_boxes_enabled": true,
  "movement_detection_enabled": true,
  "show_alarm_box": true,
  "alarm_sound": "alarm.wav",
  "resolution": "1280x720"
}
```

### Controls

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/toggle_alarm_box` | Toggle alarm zone on/off |
| `POST` | `/toggle_movement_detection` | Toggle movement detection on/off |
| `POST` | `/toggle_yolo_boxes` | Toggle YOLO box overlay on/off |
| `POST` | `/toggle_box_visibility` | Toggle zone rectangle visibility |
| `POST` | `/save_box` | Save watch zone coordinates |
| `POST` | `/set_alarm` | Set the active alarm sound file |

**`/save_box` request body:**
```json
{
  "box_norm": [0.1, 0.2, 0.8, 0.9]
}
```
Coordinates are **normalized** (0.0–1.0) relative to the stream resolution.

**`/set_alarm` request body:**
```json
{
  "filename": "alarm.wav"
}
```

### Snapshots & Recordings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/snapshot` | Downloads the current annotated frame as a JPEG |
| `GET` | `/list_alarms` | Returns list of available alarm sound files |
| `POST` | `/start_recording` | Start a new MP4 recording |
| `POST` | `/stop_recording` | Stop the active recording |
| `GET` | `/list_recordings` | Returns list of recordings with sizes |
| `GET` | `/download_recording/<filename>` | Download a specific recording |
| `DELETE` | `/delete_recording/<filename>` | Delete a specific recording |

**`/list_recordings` response:**
```json
[
  {"file": "rec_20240115_143500.mp4", "size_mb": 142.3},
  {"file": "rec_20240115_120000.mp4", "size_mb": 87.1}
]
```

---

## 🔊 Adding Alarm Sounds

Drop any `.wav` or `.mp3` file into the `alarms/` folder:

```bash
cp my_alarm.wav alarms/
```

Then select it from the **Alarm Sound** dropdown in the dashboard. The sound plays using `ffplay` (part of FFmpeg), so FFmpeg must be installed.

To create a simple test tone on Linux:

```bash
ffmpeg -f lavfi -i "sine=frequency=1000:duration=2" alarms/beep.wav
```

---

## 🔁 Running on Boot (systemd)

To run SecureView automatically when your Linux server starts:

### 1. Create a service file

```bash
sudo nano /etc/systemd/system/secureview.service
```

Paste the following (adjust paths to match your setup):

```ini
[Unit]
Description=SecureView Camera Dashboard
After=network.target

[Service]
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/secureview
ExecStart=/home/YOUR_USERNAME/secureview/venv/bin/python3 app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### 2. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable secureview
sudo systemctl start secureview
```

### 3. Check status and logs

```bash
sudo systemctl status secureview
sudo journalctl -u secureview -f      # follow live logs
```

---

## 🛠️ Troubleshooting

### Camera stream shows nothing / "OFFLINE"

- Confirm the RTSP URL is correct — test it with VLC: `Media → Open Network Stream → paste URL`
- Make sure the camera and the server are on the same network
- Some cameras require the username/password in the URL: `rtsp://admin:password@192.168.1.100:554/stream1`
- Try changing `cv2.CAP_FFMPEG` to `cv2.CAP_GSTREAMER` in `camera_reader()` if FFmpeg backend has issues

### Alarm sound doesn't play

- Make sure `ffmpeg` / `ffplay` is installed: `ffplay -version`
- Check that the file exists in the `alarms/` folder
- Check that the filename in the dropdown matches exactly (case-sensitive on Linux)
- On some systems you may need to grant audio permissions

### High CPU usage

- Use a smaller YOLO model (`yolov8n.pt` is the smallest)
- Lower the resolution: set `FRAME_W = 640` and `FRAME_H = 360`
- The alarm zone detection only runs when **Alarm Zone is enabled** — keep it disabled when not needed
- Consider reducing `JPEG_QUALITY` to `60–70` to reduce encoding overhead

### `pyturbojpeg` install fails

```bash
# Ubuntu/Debian
sudo apt-get install libturbojpeg0-dev python3-dev

# Then retry
pip install pyturbojpeg
```

### Recording file is 0 bytes or won't play

- This can happen if the app crashes mid-recording. The `mp4v` codec writes an incomplete file.
- Try repairing it with FFmpeg: `ffmpeg -i rec_broken.mp4 -c copy rec_fixed.mp4`
- To avoid this, always click **Stop Rec** before closing the app

### `ModuleNotFoundError` on startup

Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

### Port 5000 already in use

Change the port at the bottom of `app.py`:
```python
app.run(host="0.0.0.0", port=8080, ...)   # change 5000 to anything free
```

---

## ❓ FAQ

**Q: Can I use multiple cameras?**
A: Not natively — the app is built for one RTSP source. To monitor multiple cameras, run multiple instances of the app on different ports.

**Q: Does it work without a GPU?**
A: Yes. It runs on CPU by default. On a modern CPU (`yolov8n.pt`) you can expect 5–15 FPS of inference depending on your hardware.

**Q: How do I change which object classes trigger the alarm?**
A: In `generate_frames()`, find this line and add/remove class names:
```python
if name not in ["person", "car"]:
    continue
```
Available classes include: `truck`, `bicycle`, `motorcycle`, `dog`, `cat`, `bus`, and [79 more](https://docs.ultralytics.com/datasets/detect/coco/).

**Q: Can I access the dashboard from outside my home network?**
A: Yes, but do not expose port 5000 directly to the internet without authentication. Use a VPN (like WireGuard or Tailscale) to access your home network remotely, then connect to the dashboard as if you were local.

**Q: Where are snapshots and recordings saved?**
A: In the `snapshots/` and `recordings/` folders next to `app.py`. They are not cleaned up automatically — manage disk space manually or set up a cron job to delete old files.

**Q: How do I reset the detection count?**
A: The count resets every time the app restarts. It is an in-memory counter only and is not persisted to disk.

**Q: The watch zone resets after I restart the app.**
A: The zone is stored in memory only. To persist it, you can add a JSON file save/load around the `alarm_box` variable — or simply redraw it after each restart.

---

## 📄 License

MIT — do whatever you want with it.
