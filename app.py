import os
import cv2
import time
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, Response, flash
)
from ultralytics import YOLO

UPLOAD_FOLDER = "uploads"
STATIC_OUTPUT = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_OUTPUT, exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret123"

# Load YOLO models
detect_model = YOLO("yolov8n-oiv7.pt")
pose_model = YOLO("yolov8n-pose.pt")


def probe_cameras(max_idx=5):
    """Find available webcams"""
    cams = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cams.append(i)
        cap.release()
    return cams


def draw_detections(frame, boxes, scores, classes, names):
    for (xyxy, conf, cls) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

@app.route("/", methods=["GET"])
def index():
    cams = probe_cameras(5)
    return render_template("index.html", cameras=cams)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file selected", "danger")
        return redirect(url_for("index"))

    f = request.files["file"]
    filename = f.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    mode = request.form.get("mode", "detect")

    # YOLO processing
    if mode == "pose":
        result = pose_model(save_path)[0]
        output = result.plot()
    else:
        result = detect_model(save_path)[0]
        img = result.orig_img.copy()

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        draw_detections(img, boxes, scores, classes, detect_model.names)
        output = img

    out_path = os.path.join(STATIC_OUTPUT, f"output_{int(time.time())}.jpg")
    cv2.imwrite(out_path, output)

    return render_template(
        "index.html",
        uploaded_image=out_path,
        cameras=probe_cameras(5)
    )


# ---------- WEBCAM ----------
def generate_stream(cam_index, mode):
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "pose":
            result = pose_model(frame)[0]
            frame = result.plot()
        else:
            result = detect_model(frame)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            draw_detections(frame, boxes, scores, classes, detect_model.names)

        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

    cap.release()


@app.route("/webcam", endpoint="webcam_page")
def webcam_page():
    cams = probe_cameras(5)
    cam = request.args.get("cam", cams[0] if cams else 0)
    mode = request.args.get("mode", "detect")
    return render_template(
        "webcam.html",
        cameras=cams,
        selected_cam=int(cam),
        mode=mode
    )


@app.route("/webcam_feed")
def webcam_feed():
    cam = request.args.get("cam", 0)
    mode = request.args.get("mode", "detect")
    return Response(
        generate_stream(cam, mode),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
