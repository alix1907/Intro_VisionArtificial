from ultralytics import YOLO

modelo = YOLO("yolov8n.pt")

modelo.predict(
source=0,
show=True,
conf=0.5,
save=True
)