# 🎯 Real-Time Object Detection using YOLOv8

A high-performance real-time object detection system built with YOLOv8, achieving **mAP of 0.99**. Detects and classifies multiple objects from webcam feed or uploaded images/videos.

## 🚀 Features

- Real-time object detection using YOLOv8
- Achieves **mAP@0.5 = 0.99** on custom dataset
- Supports webcam, image, and video input
- Bounding box visualization with class labels & confidence scores
- Flask-based web interface for easy access

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)** – Object detection model
- **OpenCV** – Video/image processing
- **Flask** – Web framework
- **PyTorch** – Deep learning backend

## 📁 Project Structure

```
object-detection/
├── app.py                  # Flask app entry point
├── detect.py               # YOLOv8 inference logic
├── train.py                # Model training script
├── models/                 # YOLOv8 weights (.pt files)
├── datasets/               # Training & validation data
├── static/                 # CSS, JS, assets
├── templates/              # HTML templates
└── requirements.txt        # Python dependencies
```

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/object-detection.git
   cd object-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. Open your browser at `http://localhost:5000`

## 🏋️ Training

To train YOLOv8 on your custom dataset:

```bash
python train.py --data dataset.yaml --epochs 50 --imgsz 640
```

## 📊 Model Performance

| Metric        | Score  |
|---------------|--------|
| mAP@0.5       | 0.99   |
| Precision     | High   |
| Recall        | High   |
| Inference FPS | Real-time |

## 📸 Demo

> Upload an image/video or use your webcam to detect objects in real time with bounding boxes and confidence scores.

## 📦 Requirements

- Python 3.8+
- Ultralytics (YOLOv8)
- OpenCV
- Flask
- PyTorch
- NumPy

## 🙋‍♂️ Author

**Harish Kumar**  
[GitHub](https://github.com/iharish17) • [LinkedIn](https://linkedin.com/in/harishk18)
