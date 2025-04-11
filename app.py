from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (make sure the path is correct)
model = load_model('model/plant_disease_binary_model.keras')

# Preprocessing function
def preprocess_image(img_file):
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in request', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected', 400

    try:
        # Preprocess the image
        img_array = preprocess_image(file)

        # Make prediction
        prediction = model.predict(img_array)
        result = "Healthy" if prediction[0][0] > 0.5 else "Diseased"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

