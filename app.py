import tkinter as tk
from tkinter import filedialog
import cv2
import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO


model_path = "medium.pt"
model = YOLO(model_path)


def detect_heads(image):
  
    results = model(image)
    heads = []
    for result in results:
        for box in result.boxes.xyxy:
            x, y, x2, y2 = map(int, box[:4])
            heads.append([x, y, x2, y2])
    return heads


def process_video(file_path):

    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        label.config(text="Error: Couldn't open video file.")
        return
    
    # Get the total number of frames in the video and the frame rate (fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    
    # Calculate the frame number for 1 minute, 2 minutes, and 3 minutes before the end
    frame_interval = int(fps * 60)  
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
    
    # Detect heads in the frames
    head_counts = []
    for i, frame in enumerate(frames):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        heads = detect_heads(rgb_image)
        head_counts.append(len(heads))
        
        # Draw green bounding boxes around the heads
        for x, y, x2, y2 in heads:
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)  

    # Select the frame with the maximum head count
    max_head_count_index = head_counts.index(max(head_counts))
    selected_frame = frames[max_head_count_index]
    selected_frame_rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)

    # Save the selected frame with maximum heads (with bounding boxes)
    output_dir = os.getcwd()  # Current directory
    filename = os.path.join(output_dir, f"max_head_count_frame.jpg")
    cv2.imwrite(filename, selected_frame)  # Save the selected frame as an image
    
    label.config(text="Frame extracted and saved successfully!")
    
    # Display the selected frame with maximum heads (with bounding boxes)
    plt.figure(figsize=(8, 6))
    plt.imshow(selected_frame_rgb)
    plt.title(f'Heads Detected: {head_counts[max_head_count_index]}')
    plt.axis("off")
    plt.show()

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*"))
    )
    
    if file_path:  
        label.config(text=f"Selected File: {file_path}")
        process_video(file_path)
    else:
        label.config(text="No file selected")


root = tk.Tk()
root.title("Video Frame Extractor and Head Detection")
root.geometry("400x300")  # Set the size of the window


label = tk.Label(root, text="Select a video file to extract frames", font=("Arial", 14))
label.pack(pady=20)


button = tk.Button(root, text="Open Video File", command=open_file)
button.pack(pady=10)


root.mainloop()
