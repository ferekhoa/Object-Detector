import os
import io
import uuid
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLO_MODEL", "models/best.pt")

RESULTS_DIR = Path("static") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if not RESULTS_DIR.is_dir():
    raise RuntimeError("static/results exists but is not a directory. Delete it and recreate the folder.")

model = YOLO(MODEL_PATH)

def detect(image_bytes: bytes, conf: float = 0.25, iou: float = 0.45):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(img, conf=conf, iou=iou, verbose=False)
    res = results[0]

    boxes = []
    names = model.names
    if res.boxes is not None:
        xyxy = res.boxes.xyxy.tolist()
        cls = res.boxes.cls.tolist()
        confs = res.boxes.conf.tolist()
        for b, c, cf in zip(xyxy, cls, confs):
            x1, y1, x2, y2 = [float(v) for v in b]
            boxes.append({"label": names[int(c)], "confidence": float(cf), "box": [x1, y1, x2, y2]})

    arr_bgr = res.plot()
    arr_rgb = arr_bgr[:, :, ::-1]

    out_path = RESULTS_DIR / f"{str(uuid.uuid4())[:8]}.jpg"
    Image.fromarray(arr_rgb).save(out_path, "JPEG", quality=90)

    return boxes, str(out_path)
