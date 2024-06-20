import cv2
import os
from imread_from_url import imread_from_url

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/last.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
img_path = os.path.join(os.path.dirname(__file__), 'images', 'Cars259.png')
# print(img_path)
img = cv2.imread(img_path)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)
print(boxes)
print(scores)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.waitKey(0)