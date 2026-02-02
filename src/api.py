from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import torch
import io
from PIL import Image

# Import from our local modules
from .inference import get_model, predict
from .dataset import get_classes

# Global variable to hold the model
model_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the API starts, and clean up when it stops.
    """
    print("Loading model...")
    try:
        model_context["model"] = get_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_context["model"] = None
    yield
    model_context.clear()

app = FastAPI(title="Capstone2 Image Classifier", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Capstone2 API",
        "usage": "POST an image file to /predict to classify it."
    }

# --- THIS WAS MISSING ---
@app.get("/health")
def health_check():
    """Checks if the model is loaded and ready."""
    if model_context.get("model") is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy"}
# ------------------------

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and returns the classification.
    """
    # 1. Validate Model Availability
    model = model_context.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available yet.")

    # 2. Validate File Type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed.")

    # 3. Read and Predict
    try:
        contents = await file.read()
        prediction_class = predict(model, contents)
        return {
            "filename": file.filename,
            "prediction": prediction_class,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
