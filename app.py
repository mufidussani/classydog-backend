from flask import Flask, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
import pickle

# from numba import jit

# Declare a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/")
def index():
    return "CLASSYDOG!"


@app.route("/upload", methods=["POST"])
def handle_form():
    try:
        files = request.files
        file = files.get("file")

        model = load_model("./model")

        with open("breed_label.pkl", "rb") as f:
            breed_label = pickle.load(f)

        def crop_center(pil_img):
            img_w, img_h = pil_img.size
            hw = min(pil_img.size)
            return pil_img.crop(
                (
                    (img_w - hw) // 2,
                    (img_h - hw) // 2,
                    (img_w + hw) // 2,
                    (img_h + hw) // 2,
                )
            )

        inception_bottleneck = InceptionResNetV2(
            weights="imagenet", include_top=False, input_shape=(299, 299, 3)
        )
        # Resize & reshape image array, extract features and return prediction

        def predict(pil_img):
            img_array = np.array(crop_center(pil_img))
            image_array = cv2.resize(img_array, (299, 299)) / 255
            image_array = np.reshape(image_array, (1, 299, 299, 3))
            features = inception_bottleneck.predict(image_array)
            X = features.reshape((1, -1))
            prediction = model.predict(X)
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index]
            return {
                "prediction": breed_label[prediction.argmax()],
                "confidence": float(confidence),
            }

        image = Image.open(file)
        # Predict
        prediction = predict(image)
        value = prediction
        if value["confidence"] > 0.8:
            return {"predict": value["prediction"], "confidence": value["confidence"]}
        else:
            return {"predict": "Bukan Anjing", "confidence": value["confidence"]}

    except ValueError:
        value = {"predict": "bukan anjing"}
        return value


# @app.route('/getData', methods=['GET'])
# def getData():
#     handle_form()


@app.route("/data", methods=["POST"])
# @jit
def main():
    model = load_model("./model")

    with open("breed_label.pkl", "rb") as f:
        breed_label = pickle.load(f)

    def crop_center(pil_img):
        img_w, img_h = pil_img.size
        hw = min(pil_img.size)
        return pil_img.crop(
            ((img_w - hw) // 2, (img_h - hw) // 2, (img_w + hw) // 2, (img_h + hw) // 2)
        )

    inception_bottleneck = InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(299, 299, 3)
    )
    # Resize & reshape image array, extract features and return prediction

    def predict(pil_img):
        img_array = np.array(crop_center(pil_img))
        image_array = cv2.resize(img_array, (299, 299)) / 255
        image_array = np.reshape(image_array, (1, 299, 299, 3))
        features = inception_bottleneck.predict(image_array)
        X = features.reshape((1, -1))
        prediction = model.predict(X)
        return breed_label[prediction.argmax()]
        # return prediction

    image = Image.open("tes1.jpg")
    # Predict
    prediction = predict(image)

    # Get values through input bars
    # height = request.form.get("height")
    # weight = request.form.get("weight")

    # Put inputs to dataframe
    # X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])

    # Get prediction
    # prediction = model.predict(X)[0]

    # else:
    #     prediction = ""
    print(prediction)
    return {"predict": prediction}


# Running the app
if __name__ == "__main__":
    app.run(debug=True)
