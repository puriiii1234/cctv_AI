# Real-Time Multi-Object Tracking (MOT) and Analytics Pipeline

This repository contains a high-performance computer vision pipeline designed for real-time traffic analysis and surveillance. It integrates object detection (YOLOv11) with a custom centroid-based tracking algorithm, Automatic Number Plate Recognition (ANPR) using a custom fine-tuned YOLOv11 model (best.pt) and EasyOCR, and an asynchronous visualization dashboard.

The system is engineered to decouple the heavy computational load of the inference engine from the lightweight I/O operations of the reporting dashboard, ensuring minimal latency in video processing.

---

## 📋 System Overview

The architecture consists of four primary modules:

### **Inference Engine**
Utilizes ultralytics YOLOv11s for detecting specific classes (Person, Car, Motorcycle, Bus, Truck) within the COCO dataset, coupled with a secondary model for license plate localization.

### **Custom Tracker**
A lightweight implementation of centroid tracking using Euclidean distance minimization to maintain object identity across temporal frames.

### **Data Persistence Layer**
A structured logging module that captures granular telemetry for every gate traversal event, exporting audit trails to CSV format for post-hoc analysis.

### **Analytics Server**
A Dash (Flask-based) web application running on a separate thread, consuming telemetry data via a thread-safe queue to render real-time time-series graphs.

---

## 🔑 Key Capabilities

- **Multi-Class Detection:** Filters and tracks 5 distinct vehicle/pedestrian classes.  
- **Vector-Based Counting:** Utilizes position history (deque) to determine direction vectors (Entry vs. Exit) rather than simple line intersection, reducing false positives caused by jitter.  
- **Automated Reporting:** Generates comprehensive CSV logs containing timestamps, traversal direction, and OCR-extracted license plate data.  
- **Asynchronous Visualization:** Implements a Producer-Consumer pattern where the vision loop produces metrics and the Dash server consumes them.

---

## 🏗️ Technical Architecture

### **1. Object Tracking Algorithm (ObjectTracker)**

Unlike heavy trackers like DeepSORT, this implementation uses a spatial coherence approach optimized for fixed-camera setups.

- **Centroid Calculation:** Computes the geometric center of bounding boxes.  
- **Distance Matrix:** Constructs a Euclidean distance matrix between existing track centroids and new detection centroids.  
- **Assignment:** Solves the linear assignment problem (greedy approach) to associate detections with IDs.  
- **Lifecycle Management:** Handles deregistration of lost objects after `max_disappeared` frames (set to 30).

---

### **2. Event Logic & Reporting (GateCounter)**

To ensure accuracy in counting and logging:

- **Hysteresis:** Tracks y-coordinate history to classify entry vs exit based on direction.  
- **Deduplication:** Prevents double-counting using a set of completed IDs.  
- **Structured Export (CSV):** Serializes event records into `vehicle_log.csv`.

#### **Schema:**

| Timestamp | Object ID | Class | Direction | License Plate |
| :--- | :--- | :--- | :--- | :--- |
| YYYY-MM-DD HH:MM:SS | 102 | Car | IN | ABC-1234 |
| YYYY-MM-DD HH:MM:SS | 103 | Person | OUT | N/A |

---

### **3. Concurrency Model**

The application runs in a multi-threaded environment:

- **Main Thread:** Executes the OpenCV loop, GPU inference, and CSV writing.  
- **Daemon Thread:** Runs the Dash server.  
- **Data Exchange:** Uses a `queue.Queue` for non-blocking metric transfer.

---

## 📦 Installation & Setup

### **Recommended**
- Python 3.8+  
- NVIDIA GPU with CUDA 

### **Dependencies**
Install packages:

```bash
pip install -r requirements.txt
```

### **Configuration**

Update your RTSP or video source in `main.py`:

```python
# In main.py
rtsp = "rtsp://admin:password@192.168.1.XX:554/stream"  # IP Camera
# OR
rtsp = 0  # Local Webcam
```

---

## 🚀 Usage

Run the main script:

```bash
python main.py
```

You will see three outputs:

1. **Computer Vision Feed** – stream with annotated detections and gate.  
2. **Analytics Dashboard** – open at: `http://localhost:8050`.  
3. **Data Log** – `vehicle_log.csv` updated in real-time.

---

## 📊 Analytics Dashboard

The dashboard is built with Plotly and Dash, providing:

- **Temporal Analysis:** Real-time line charts of entry volume per class.  
- **Responsive Layout:** Subplots for each category.  
- **Data Persistence:** Rolling window of last 100 points.

---

## 🤝 Contribution

Contributions regarding the optimization of the tracking algorithm (e.g., implementing Kalman Filters for state estimation) or database integration (SQLite/PostgreSQL) are welcome.

---

## 📜 License

MIT
