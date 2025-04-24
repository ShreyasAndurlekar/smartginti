import tkinter as tk
from tkinter import filedialog
import cv2
import os
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

API_URL = "https://detect.roboflow.com"
API_KEY = "LDZkJOKciCnM6TrrDgXT"
MODEL_ID = "classroom-head-h0mey"
VERSION = "2"
CONFIDENCE = 0.5
MAX_OVERLAP = 0.3

def detect_heads(image):
    """Send the frame to Roboflow API for head detection."""
    api_url = f"{API_URL}/{MODEL_ID}/{VERSION}?api_key={API_KEY}&format=json&confidence={CONFIDENCE}&overlap={MAX_OVERLAP}"

    _, img_encoded = cv2.imencode(".jpg", image)
    response = requests.post(api_url, files={"file": img_encoded.tobytes()})

    if response.status_code == 200:
        results = response.json()
        heads = []
        for prediction in results.get("predictions", []):
            x, y, width, height = (
                prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            )
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            heads.append([x1, y1, x2, y2])
        return heads
    else:
        print("Error:", response.status_code, response.text)
        return []

def process_video(file_path):
    """Extracts frames and processes them with head detection."""
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        label.config(text="Error: Couldn't open video file.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  
    
    frame_interval = int(fps * 60)  # Capture every 60 seconds
    frames_to_capture = [
        total_frames - frame_interval * 3,  
        total_frames - frame_interval * 2,  
        total_frames - frame_interval      
    ]
    
    frames = []
    for frame_number in frames_to_capture:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            label.config(text="Error: Couldn't read frame.")
            cap.release()
            return
    
    cap.release()  
    
    head_counts = []
    for i, frame in enumerate(frames):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        heads = detect_heads(rgb_image)
        head_counts.append(len(heads))
        
        for x, y, x2, y2 in heads:
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)  

    max_head_count_index = head_counts.index(max(head_counts))
    selected_frame = frames[max_head_count_index]
    selected_frame_rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)

    output_dir = os.getcwd()  
    filename = os.path.join(output_dir, "max_head_count_frame.jpg")
    cv2.imwrite(filename, selected_frame)  
    
    label.config(text="Frame extracted and saved successfully!")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(selected_frame_rgb)
    plt.title(f'Heads Detected: {head_counts[max_head_count_index]}')
    plt.axis("off")
    plt.show()

def open_file():
    """Opens a file dialog to select a video."""
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*"))
    )
    
    if file_path:  
        label.config(text=f"Selected File: {file_path}")
        process_video(file_path)
    else:
        label.config(text="No file selected")

# GUI Setup
root = tk.Tk()
root.title("Video Frame Extractor and Head Detection")
root.geometry("400x300")  

label = tk.Label(root, text="Select a video file to extract frames", font=("Arial", 14))
label.pack(pady=20)

button = tk.Button(root, text="Open Video File", command=open_file)
button.pack(pady=10)

root.mainloop()
