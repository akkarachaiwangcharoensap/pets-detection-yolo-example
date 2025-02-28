import cv2
import base64
import io

import os

# Set YOLO to quiet mode 'True', else 'False' 
os.environ['YOLO_VERBOSE'] = 'True'

from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet

# For compatibility with SocketIO
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the YOLO model
model = YOLO("yolov8m-oiv7.pt")

def pets_detected_callback():
    print("Pets detected!")

def pets_not_detected_callback():
    print("No pets detected.")

def generate_frames():
    cap = cv2.VideoCapture("pets-video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        # classes: 
        # Run YOLO prediction on the current frame.
        results = model.predict(
            source=frame,   # Process current frame
            classes=[411, 160, 96, 8, 412, 488],  # Filter for Rabbit, Dog, Cat, Animal, Raccoon, Squrriel 
            show=False,
            save=False,
            imgsz=576,
            max_det=5,
            vid_stride=10,
            device="cpu"
        )

        r = results[0]

        # Trigger callbacks based on detections.
        if r.boxes is not None and len(r.boxes) > 0:
            pets_detected_callback()
        else:
            pets_not_detected_callback()

        # Obtain the annotated frame (with bounding boxes drawn).
        annotated_frame = r.plot()

        # Encode the frame as JPEG.
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Emit the frame over the "new_frame" event.
        socketio.emit('new_frame', {'image': jpg_as_text})
        socketio.sleep(0.03)  # Adjust sleep for desired frame rate

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print("Client connected.")

if __name__ == '__main__':
    socketio.start_background_task(generate_frames)
    socketio.run(app, debug=True, host='127.0.0.1', port=8082)