from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

results = model.train(data="real-time-waste-classification-2/data.yaml", epochs=5, imgsz=640)

# Evaluate the model's performance on the validation set
results = model.val()

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("telor.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")