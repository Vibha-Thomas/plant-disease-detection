Plant Disease Detection (Tri-Class) using Deep Learning

A Deep Learning-based web app that classifies leaf images into one of three categories:
- *Healthy*
- *Diseased*
- *Not a Leaf*

Deployed using *Flask + Render*, and trained using **MobileNetV2** for speed, accuracy, and mobile-readiness.

---

Dataset

The dataset was manually curated and organized into three classes with approximately 800–900 images per class:

- `Train/Healthy/`
- `Train/Diseased/`
- `Train/Not a Leaf/`

All images were resized to `224x224` and normalized before training.

---

Model Architecture

I used **MobileNetV2**, a lightweight and powerful **CNN model** pretrained on ImageNet. The custom layers added:

- GlobalAveragePooling2D
- Dense(64, relu)
- Dropout(0.3)
- Dense(3, softmax)

Training settings:
- EarlyStopping (patience=3)
- Data Augmentation
- Validation Split (20%)
- Categorical Crossentropy Loss
- Adam Optimizer

---

Performance

 Metric       | Value        
--------------|--------------
 Train Acc.   | ~92%         
 Val Acc.     | ~90%         
 Test Acc.    | ~91%         
 Overfitting  | Controlled   
 Inference    | ~99% Confidence (correct) 

---

Deployment (Render)

The Flask app is hosted on **Render.com** with the following features:

- Upload an image (JPEG/PNG)
- Model predicts class + confidence
- Live and mobile responsive

**Model Used**: `plant_disease_triclass_model.keras`  
**Deployment Type**: Flask Web App  
**Hosting Platform**: Render (Free Tier)

---

How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-detection
cd plant-disease-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
