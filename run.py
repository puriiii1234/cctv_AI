import cv2
import numpy as np
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import threading
import time
import json
from collections import defaultdict, deque
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import queue

# outside
#rtsp = "rtsp://admin:admin@10.10.62.230:554/media/video1"

#inside main
rtsp = "rtsp://admin:AdmiN%40123@10.10.62.156:554/media/video1"

#RTSP inside sub-stream 
#rtsp = "rtsp://admin:AdmiN%40123@10.10.62.156:554/media/video2"
class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = 30
        self.max_distance = 50

    def register(self, centroid):
        self.tracks[self.next_id] = {
            'centroid': centroid,
            'disappeared': 0,
            'positions': deque(maxlen=10)
        }
        self.tracks[self.next_id]['positions'].append(centroid)
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.tracks[object_id]

    def update(self, objects):
        if len(objects) == 0:
            for object_id in list(self.tracks.keys()):
                self.tracks[object_id]['disappeared'] += 1
                if self.tracks[object_id]['disappeared'] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        if len(self.tracks) == 0:
            for obj in objects:
                self.register(obj)
        else:
            object_ids = list(self.tracks.keys())
            object_centroids = [self.tracks[id]['centroid'] for id in object_ids]

            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(objects), axis=2)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.tracks[object_id]['centroid'] = objects[col]
                self.tracks[object_id]['positions'].append(objects[col])
                self.tracks[object_id]['disappeared'] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.tracks[object_id]['disappeared'] += 1
                    if self.tracks[object_id]['disappeared'] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(objects[col])

        return self.tracks


class GateCounter:
    def __init__(self, gate_y_position):
        self.gate_y = gate_y_position
        self.crossed_objects = set()
        self.entry_counts = defaultdict(int)
        self.exit_counts = defaultdict(int)
        self.total_counts = defaultdict(int)

    def check_crossing(self, object_id, positions, object_class):
        if len(positions) < 2:
            return None

        prev_y = positions[-2][1]
        curr_y = positions[-1][1]

        # Check if object crossed the gate line
        if prev_y > self.gate_y and curr_y <= self.gate_y:
            if object_id not in self.crossed_objects:
                self.crossed_objects.add(object_id)
                self.entry_counts[object_class] += 1
                self.total_counts[object_class] += 1
                return 'exit'
        elif prev_y <self.gate_y and curr_y >= self.gate_y:
            if object_id not in self.crossed_objects:
                self.crossed_objects.add(object_id)
                self.exit_counts[object_class] += 1
                self.total_counts[object_class] -= 1
                return 'entry'

        return None


class VideoAnalytics:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.model = YOLO('yolov11s.pt')

        # Class mappings for COCO dataset
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        self.trackers = {class_name: ObjectTracker() for class_name in self.target_classes.values()}

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Gate position (80% of height)
        self.gate_y = int(0.8 * self.frame_height)
        self.gate_counter = GateCounter(self.gate_y)

        # Data storage for graphs
        self.data_queue = queue.Queue()
        self.time_data = deque(maxlen=100)
        self.entry_data = {
            'person': deque(maxlen=100),
            'car': deque(maxlen=100),
            'motorcycle': deque(maxlen=100),
            'bus': deque(maxlen=100)
        }

        # Colors for different classes
        self.colors = {
            'person': (255, 100, 100),
            'car': (0, 0, 255),
            'motorcycle': (100, 100, 255),
            'bus': (255, 255, 100),
            'truck': (255, 100, 255)
        }

        self.running = False

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Run YOLO detection
        results = self.model(frame)

        # Process detections
        detections = defaultdict(list)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    if class_id in self.target_classes:
                        confidence = float(box.conf)
                        if confidence > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            class_name = self.target_classes[class_id]
                            detections[class_name].append((center_x, center_y))

                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          self.colors[class_name], 2)

                            # Draw confidence score
                            cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, self.colors[class_name], 2)

        # Update trackers and check gate crossings
        current_time = datetime.now()

        for class_name, objects in detections.items():
            tracks = self.trackers[class_name].update(objects)

            for object_id, track_info in tracks.items():
                positions = list(track_info['positions'])

                # Check gate crossing
                crossing_type = self.gate_counter.check_crossing(
                    f"{class_name}_{object_id}", positions, class_name
                )

                # Draw tracking info
                center = track_info['centroid']
                cv2.circle(frame, tuple(map(int, center)), 5, self.colors[class_name], -1)
                cv2.putText(frame, f'ID: {object_id}',
                            (int(center[0]), int(center[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_name], 2)

        # Draw gate line
        cv2.line(frame, (0, self.gate_y), (self.frame_width, self.gate_y),
                 (0, 255, 255), 3)
        cv2.putText(frame, 'GATE', (10, self.gate_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw statistics with aesthetic styling
        self.draw_statistics(frame)

        # Update data for graphs
        self.update_graph_data(current_time)

        return frame

    def draw_statistics(self, frame):
        # Create aesthetic overlay
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (485, 174), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, 'GLOBAL COLLEGE GATE ANALYTICS', (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        y_pos = 70
        for class_name in ['person', 'car', 'motorcycle', 'bus']:
            total = self.gate_counter.total_counts[class_name]
            entries = self.gate_counter.entry_counts[class_name]
            exits = self.gate_counter.exit_counts[class_name]

            text = f'{class_name.upper()}: {total} (Entry:{entries} | Exit:{exits})'
            cv2.putText(frame, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[class_name], 2)
            y_pos += 25

    def update_graph_data(self, current_time):
        # Add current time and entry counts to data
        self.time_data.append(current_time)

        for class_name in self.entry_data.keys():
            self.entry_data[class_name].append(self.gate_counter.entry_counts[class_name])

        # Send data to graph queue
        data_point = {
            'time': current_time,
            'person': self.gate_counter.entry_counts['person'],
            'car': self.gate_counter.entry_counts['car'],
            'motorcycle': self.gate_counter.entry_counts['motorcycle'],
            'bus': self.gate_counter.entry_counts['bus']
        }

        if not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass

        self.data_queue.put(data_point)

    def run(self):
        self.running = True

        while self.running:
            frame = self.process_frame()
            if frame is not None:
                cv2.imshow('College Gate Analytics', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Dash app for real-time graphs
def create_dash_app(analytics):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("College Gate Analytics Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px',
                       'fontFamily': 'Arial, sans-serif', 'fontSize': '36px'}),

        dcc.Graph(id='live-graph'),

        dcc.Interval(
            id='graph-update',
            interval=1000,  # Update every second
            n_intervals=0
        ),

        html.Div(id='stats-display', style={'marginTop': '20px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px'})

    @app.callback(Output('live-graph', 'figure'),
                  Input('graph-update', 'n_intervals'))
    def update_graph(n):
        if analytics.data_queue.empty():
            return create_empty_figure()

        # Get latest data
        try:
            data = analytics.data_queue.get_nowait()
        except queue.Empty:
            return create_empty_figure()

        # Create DataFrame from recent data
        if len(analytics.time_data) > 1:
            df = pd.DataFrame({
                'Time': list(analytics.time_data),
                'Person': list(analytics.entry_data['person']),
                'Car': list(analytics.entry_data['car']),
                'Motorcycle': list(analytics.entry_data['motorcycle']),
                'Bus': list(analytics.entry_data['bus'])
            })

            # Create stunning plotly figure
            fig = create_aesthetic_figure(df)
            return fig

        return create_empty_figure()

    return app


def create_aesthetic_figure(df):
    # Create subplots with futuristic styling
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Person Entries', 'Car Entries', 'Motorcycle Entries', 'Bus Entries'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Color scheme - futuristic neon colors
    colors = {
        'Person': '#00ff9f',
        'Car': '#00d4ff',
        'Motorcycle': '#ff0080',
        'Bus': '#ffaa00'
    }

    # Add traces for each category
    categories = ['Person', 'Car', 'Motorcycle', 'Bus']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for i, (category, pos) in enumerate(zip(categories, positions)):
        fig.add_trace(
            go.Scatter(
                x=df['Time'],
                y=df[category],
                mode='lines+markers',
                name=category,
                line=dict(color=colors[category], width=3, shape='spline'),
                marker=dict(size=8, color=colors[category],
                            line=dict(width=2, color='white')),
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=f'rgba({",".join(str(int(colors[category][i:i + 2], 16)) for i in (1, 3, 5))}, 0.3)'
            ),
            row=pos[0], col=pos[1]
        )

    # Update layout with futuristic styling
    fig.update_layout(
        title={
            'text': 'Real-Time College Gate Analytics',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.9)',
        font=dict(color='white', size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.2)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255,255,255,0.5)'
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.2)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255,255,255,0.5)'
    )

    return fig


def create_empty_figure():
    fig = go.Figure()
    fig.update_layout(
        title="Waiting for data...",
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.9)',
        font=dict(color='white')
    )
    return fig


def main():
    # Initialize video analytics
    analytics = VideoAnalytics(rtsp)  # Use 0 for webcam or path to video file

    # Create and run dash app in separate thread
    dash_app = create_dash_app(analytics)

    def run_dash():
        dash_app.run_server(debug=False, port=8050)

    # Start dash app in background
    dash_thread = threading.Thread(target=run_dash)
    dash_thread.daemon = True
    dash_thread.start()

    print("Starting video analytics...")
    print("Dashboard available at: http://localhost:8050")
    print("Press 'q' to quit video window")

    # Start video processing
    analytics.run()


if __name__ == "__main__":
    main()