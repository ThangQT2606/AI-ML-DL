import os, onnx
from ultralytics import YOLO

# Đường dẫn model YOLO
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(model_path)

# convert Yolo to onnx
model.export(format="onnx")