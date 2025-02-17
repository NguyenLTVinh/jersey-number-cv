from ultralytics import YOLO
import torch

model = YOLO("yolov8m.pt")

# Train the model for a single class
results = model.train(
    data="./data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    name="yolov8_jersey_number",
    optimizer="AdamW",
    lr0=0.005,
    patience=30,
    device="cuda" if torch.cuda.is_available() else "cpu",
    workers=8,
    conf=0.05,
    single_cls=True,
)

# Validate on the validation set
val_metrics = model.val(
    data="./data.yaml",
    split="val",
    batch=16,
    imgsz=640,
    conf=0.1,
    iou=0.6,
    device="cuda",
    name="yolov8_jersey_val",
)

# Run inference on the test set
test_results = model.predict(
    source="./test/images",
    conf=0.2,
    save=True,
    save_txt=True,
    save_conf=True,
    imgsz=640,
    device="cuda",
    name="yolov8_jersey_predict",
)

# Evaluate on the test set
test_metrics = model.val(
    data="./data.yaml",
    split="test",
    batch=16,
    imgsz=640,
    conf=0.2,
    iou=0.6,
    device="cuda",
    name="yolov8_jersey_test",
)

# Export the trained model
model.export(
    format="torchscript",
    imgsz=640,
    simplify=True,
    device="cuda",
)

# Print results
print("Training Results:", results)
print("Validation Metrics:", val_metrics)
print("Test Metrics:", test_metrics)

# Print test accuracy (mAP@0.5)
test_accuracy = test_metrics.box.map50
print("Test Accuracy (mAP@0.5):", test_accuracy)
