import cv2
import os
from ultralytics import YOLO

import json 
from azure.iot.device import IoTHubDeviceClient, Message

# Load a COCO-pretrained YOLO model
model = YOLO("best.pt")

# Azure IoT Hub Connection String
connection_string = "HostName=Kapsel.azure-devices.net;DeviceId=ESP;SharedAccessKey=NWf7pUIGx5Td4icqbE+Qne9wHhODZ/yv790knrwKLrg="
# Initialize Azure IoT client
client = IoTHubDeviceClient.create_from_connection_string(connection_string)

# Function to send data to IoT Hub
def send_data_to_iothub(client, message):
    try:
        msg = Message(message)
        client.send_message(msg)
        print(f"Message successfully sent: {message}")
    except Exception as e:
        print(f"Error sending message to IoT Hub: {e}")

# Mapping YOLO categories to custom categories
organic_categories = {"paper_recycle", "organic", "incinerable"}

def classify_waste(detected_label):
    if detected_label in organic_categories:
        return "organic"
    return "inorganic"

# Open the webcam (0 is the default webcam index)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a unique output folder for each run
base_output_dir = "runs/detect"
run_number = 1

while os.path.exists(os.path.join(base_output_dir, f"predict{run_number}")):
    run_number += 1

output_dir = os.path.join(base_output_dir, f"predict{run_number}")
os.makedirs(output_dir)

# Define the video writer for saving the output
output_video_path = os.path.join(output_dir, "output.avi")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if not available

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Saving video to: {output_video_path}")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Run inference with the YOLO model on the current frame
    results = model(frame)

    # Iterate over detections
    for detection in results[0].boxes:
        # Convert class index to label
        detected_label_index = int(detection.cls)  # YOLO provides the class index
        detected_label = model.names[detected_label_index]  # Map index to class name

        # Classify waste
        waste_type = classify_waste(detected_label)

        # Create a JSON message and send to IoT Hub
        message = json.dumps({"type": waste_type, "label": detected_label})
        send_data_to_iothub(client, message)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the video
    video_writer.write(annotated_frame)

    # Display the frame
    cv2.imshow("YOLO Webcam Output", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Saved frames are in the '{output_dir}' directory.")