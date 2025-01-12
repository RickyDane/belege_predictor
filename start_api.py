from flask import request
from ultralytics.utils import cv2
from yolov8_utils import Detector
from flask.app import Flask


app = Flask(__name__)

detector = Detector()

@app.route("/api/v1/predictBeleg", methods=["POST"])
def predict():
    image = request.files["image"]
    image.save(f"temp_image/{image.filename}")
    real_image = cv2.imread(f"temp_image/{image.filename}")
    print(image)
    results = detector.predictBelege(real_image, image.filename)
    return results

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=True)
