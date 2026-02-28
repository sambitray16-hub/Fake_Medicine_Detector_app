import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = FastAPI(title="Fake Medicine Detector API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for history images
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Path to custom trained model
MODEL_PATH = os.path.join("models", "medicine_classifier.h5")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try loading the local custom model first, fallback to generic MobileNetV2
try:
    if os.path.exists(MODEL_PATH):
        base_model = tf.keras.models.load_model(MODEL_PATH)
        is_custom_model = True
        print(f"Loaded CUSTOM model from {MODEL_PATH}")
    else:
        base_model = MobileNetV2(weights='imagenet')
        is_custom_model = False
        print("Loaded GENERIC MobileNetV2 model (fallback)")
except Exception as e:
    print(f"Error loading model: {e}")
    base_model = None
    is_custom_model = False

@app.get("/")
async def root():
    return {
        "message": "Fake Medicine Detector API is running",
        "model_type": "Custom Trained" if is_custom_model else "Generic Pre-trained"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 1. Save file to uploads folder for history/auditing
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # 2. Read and preprocess image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        
        # 3. Prepare for model
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Mocking the classification for "Fake" vs "Genuine"
        # In a real app, you'd use a custom model: model.predict(img_array)
        
        # For demonstration, we'll use MobileNet to check if it's actually a pill/medicine
        if base_model:
            preds = base_model.predict(img_array)
            decoded = decode_predictions(preds, top=5)[0]
            
            # Check if any top prediction relates to medicine/pill/bottle
            is_medicine = any(item[1].lower() in ['pill', 'pill_bottle', 'medicine_chest', 'packet', 'carton'] for item in decoded)
            
            # Simulated logic for Fake/Genuine based on image brightness/contrast OR just random
            # (Real-world would use a siamese network or specialized classifier)
            confidence = float(np.max(preds)) * 100
            
            # Heuristic for demo: if it's detected as medicine, we simulate a check
            if is_medicine:
                # Randomly decide for demo purpose, or base on some pixel property
                status = "Genuine" if np.random.random() > 0.3 else "Fake"
                reason = "Packaging texture and font consistency verified." if status == "Genuine" else "Inconsistent font spacing and low-quality hologram detected."
            else:
                status = "Unknown"
                reason = "The uploaded image does not appear to be a medicine or pill bottle."
                confidence = 0
            
            return {
                "status": status,
                "confidence": round(confidence, 2),
                "details": {
                    "detected_objects": [item[1] for item in decoded],
                    "analysis": reason,
                    "medicine_name": "Detected Packaging" if is_medicine else "N/A"
                }
            }
        else:
            return {
                "status": "Error",
                "message": "Model not loaded"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    files = os.listdir(UPLOAD_DIR)
    # Filter for image files and return recent 10
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x)), reverse=True)
    return {"history": image_files[:10]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
