from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
import config
from models.cnn import SimpleCNN

# Initialize FastAPI app
app = FastAPI(
    title="AutoML Image Classifier API",
    description="Bayesian-optimized CIFAR-10 classifier",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIFAR-10 class names
CLASSES = config.CLASSES

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010))
])

# Load model
model = None
model_info = None

# Device detection (Mac-friendly). Move code to shared class
if torch.cuda.is_available():
    device = 'cuda'
    print('Using device: NVIDIA GPU (CUDA)')
elif torch.backends.mps.is_available():
    device = 'mps'
    print('Using device: Apple Silicon GPU (MPS)')
else:
    device = 'cpu'
    print('Using device: CPU')

def load_model():
    """Load the trained model"""
    global model, model_info
    
    # Load checkpoint
    checkpoint = torch.load('saved_models/final_model.pth', map_location=device)
    
    # Create model
    dropout = checkpoint['hyperparameters'].get('dropout', 0.5)
    model = SimpleCNN(num_classes=10, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    model_info = {
        'accuracy': checkpoint['accuracy'],
        'hyperparameters': checkpoint['hyperparameters']
    }
    
    print(f"Model loaded successfully! Accuracy: {model_info['accuracy']:.2f}%")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AutoML Image Classifier API",
        "status": "running",
        "model_accuracy": f"{model_info['accuracy']:.2f}%" if model_info else "N/A"
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model_info is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model_info

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with predicted class and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        top3_predictions = [
            {
                "class": CLASSES[idx.item()],
                "confidence": f"{prob.item() * 100:.2f}%"
            }
            for prob, idx in zip(top3_prob[0], top3_idx[0])
        ]
        
        return {
            "predicted_class": CLASSES[predicted.item()],
            "confidence": f"{confidence.item() * 100:.2f}%",
            "top_3_predictions": top3_predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get list of supported classes"""
    return {"classes": CLASSES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)