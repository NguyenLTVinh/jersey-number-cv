from ultralytics import YOLO
import os
import torch


# Load the model
model = YOLO("yolov8m.pt")

# Train the model
results = model.train(
    data="./data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    name="yolov8_custom_train",
    optimizer="AdamW",
    lr0=0.01,
    patience=50,
    device="0",
    workers=8,
    conf=0.1
)

# Validate the model on the validation set
val_metrics = model.val(
    data="./data.yaml",
    split="val",
    batch=16,
    imgsz=640,
    conf=0.1,
    iou=0.6,
    device="0",
    name="yolov8_custom_val",
)

# Run inference on the test set
test_results = model.predict(
    source="./test/images",
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=True,
    imgsz=640,
    device="0",
    name="yolov8_custom_predict",
)

# Evaluate the model on the test set
test_metrics = model.val(
    data="./data.yaml",
    split="test",  # Use the test set
    batch=16,
    imgsz=640,
    conf=0.25,
    iou=0.6,
    device="0",
    name="yolov8_custom_test",
)

# Export the model
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    device="0",
)

# Print results
print("Training Results:", results)
print("Validation Metrics:", val_metrics)
print("Test Metrics:", test_metrics)

# Test accuracy (mAP@0.5)
test_accuracy = test_metrics.box.map50
print("Test Accuracy (mAP@0.5):", test_accuracy)
