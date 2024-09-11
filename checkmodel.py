from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO("./best.pt")

results = model.predict("./test-image.jpg", imgsz=640,save=True, show=True)