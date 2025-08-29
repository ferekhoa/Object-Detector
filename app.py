import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from inference import detect

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  
ALLOWED = {"jpg", "jpeg", "png", "webp"}


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return render_template("index.html", conf=0.25, iou=0.45)


@app.post("/predict")
def predict():
    if 'image' not in request.files:
        return jsonify(error="No file field named 'image'"), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify(error="No file selected"), 400

    ext = f.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type: .{ext}"), 400

    try:
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
    except ValueError:
        conf, iou = 0.25, 0.45

    img_bytes = f.read()
    boxes, out_path = detect(img_bytes, conf=conf, iou=iou)

    if request.accept_mimetypes.accept_html and not request.is_json:
        return render_template(
            "index.html",
            result_image="/" + out_path,
            boxes=boxes,
            conf=conf,
            iou=iou,
        )

    return jsonify({"boxes": boxes, "result_image": "/" + out_path})


@app.get("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory("static/results", filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
