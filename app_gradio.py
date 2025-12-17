import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ------------------ Load Model & Classes ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.h5")
classes_path = os.path.join(BASE_DIR, "class_names.json")

model = tf.keras.models.load_model(model_path)

with open(classes_path, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

# ------------------ Prediction Function ------------------
def predict_image(image: Image.Image):
    if image is None:
        return None, {}

    # Preprocess
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0)

    # Predict
    preds = model.predict(img_array)[0]

    predicted_index = int(np.argmax(preds))
    predicted_class = class_names[predicted_index]
    confidence = float(preds[predicted_index] * 100)

    # Probabilities
    prob_dict = {
        class_names[i]: float(preds[i])
        for i in range(len(class_names))
    }

    result_text = f"{predicted_class} ({confidence:.2f}%)"

    return result_text, prob_dict


# ------------------ Gradio UI ------------------
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a sports image"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Label(label="Class Probabilities")
    ],
    title="🏆 Sports Image Classifier",
    description="Upload a sports image and the CNN model will identify the sport with confidence.",
   
)

if __name__ == "__main__":
    demo.launch()
