"""FastAPI server for medical image classification using the trained federated model."""
import io
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from typing import Dict, List, Optional
import uvicorn
from contextlib import asynccontextmanager

import json
from datetime import datetime

from medapp.task import Net
from medapp.service.llm import llm_service

# Model path constants
MODEL_PATHS = {
    "pathmnist": "models/path.pt",
    "retinamnist": "models/retina.pt", 
    "dermamnist": "models/derma.pt",
    "bloodmnist": "models/blood.pt"
}

# Model configurations
MODEL_CONFIGS = {
    "pathmnist": {
        "num_classes": 9,
        "class_names": [
            "adipose", "background", "debris", "lymphocytes", 
            "mucus", "smooth_muscle", "normal_colon_mucosa", 
            "cancer-associated_stroma", "colorectal_adenocarcinoma_epithelium"
        ],
        "description": "Colon pathology classification",
        "model_path": MODEL_PATHS["pathmnist"],
        "samples": 107180
    },
    "retinamnist": {
        "num_classes": 5,
        "class_names": [
            "normal", "diabetes", "glaucoma", "cataract", "amd"
        ],
        "description": "Retinal OCT classification", 
        "model_path": MODEL_PATHS["retinamnist"],
        "samples": 1600
    },
    "dermamnist": {
        "num_classes": 7,
        "class_names": [
            "actinic_keratoses", "basal_cell_carcinoma", "benign_keratosis",
            "dermatofibroma", "melanoma", "melanocytic_nevi", "vascular_lesions"
        ],
        "description": "Dermatology classification",
        "model_path": MODEL_PATHS["dermamnist"],
        "samples": 10015
    },
    "bloodmnist": {
        "num_classes": 8,
        "class_names": [
            "basophil", "eosinophil", "erythroblast", "immature_granulocytes",
            "lymphocyte", "monocyte", "neutrophil", "platelet"
        ],
        "description": "Blood cell classification",
        "model_path": MODEL_PATHS["bloodmnist"],
        "samples": 17092
    }
}

VALIDATION_STORAGE_PATH = "validation_data.json"

# Global variables
loaded_models = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_transforms():
    """Get image transforms."""
    return Compose([
        Resize((28, 28)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_model(dataset_name: str):
    """Load the trained model for a specific dataset."""
    if dataset_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if dataset_name not in loaded_models:
        config = MODEL_CONFIGS[dataset_name]
        model = Net(num_classes=config["num_classes"])
        
        print(f"Loading model for {config['num_classes']}...")
        model_path = config["model_path"]
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        loaded_models[dataset_name] = model
    
    return loaded_models[dataset_name]

def preprocess_image(image_file: UploadFile) -> torch.Tensor:
    """Preprocess uploaded image for model inference."""
    try:
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def classify_for_dataset(dataset_name: str, file: UploadFile):
    """Common classification logic."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        model = load_model(dataset_name)
        config = MODEL_CONFIGS[dataset_name]
        
        image_tensor = preprocess_image(file)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        class_probabilities = {}
        for i, class_name in enumerate(config["class_names"]):
            class_probabilities[class_name] = float(probabilities[0][i].item())
        
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "predicted_class": config["class_names"][predicted_class_idx],
            "confidence": float(confidence),
            "class_probabilities": class_probabilities,
            "filename": file.filename,
            "num_classes": config["num_classes"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    os.makedirs("models", exist_ok=True)
    yield
    loaded_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Medical Image Classification API",
    description="API for classifying medical images using federated learning models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Medical Image Classification API",
        "available_datasets": list(MODEL_CONFIGS.keys()),
        "version": "1.0.0"
    }

@app.get("/datasets")
async def get_datasets():
    """Get information about available datasets."""
    return {"datasets": MODEL_CONFIGS}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/llm/status")
async def llm_status():
    """Get LLM service status."""
    try:
        status = llm_service.get_service_status()
        return {
            "status": "success",
            "llm_service": status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/llm/summary")
async def generate_medical_summary(classification_result: Dict):
    """Generate medical summary from classification results."""
    try:
        # Validate classification result structure
        required_fields = ["dataset", "predicted_class", "confidence", "class_probabilities"]
        for field in required_fields:
            if field not in classification_result:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field: {field}"
                )
        
        # Generate summary using LLM service
        summary = llm_service.generate_medical_summary(classification_result)
        
        return {
            "status": "success",
            "summary": summary,
            "classification_result": classification_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating medical summary: {str(e)}"
        )

@app.post("/classify_with_summary/{dataset_name}")
async def classify_with_summary(dataset_name: str, file: UploadFile = File(...)):
    """Classify image and generate medical summary in one call."""
    try:
        # First, classify the image
        classification_result = classify_for_dataset(dataset_name, file)
        
        # Then, generate medical summary
        medical_summary = llm_service.generate_medical_summary(classification_result)
        
        # Combine results
        return {
            "status": "success",
            "classification": classification_result,
            "medical_summary": medical_summary,
            "timestamp": str(torch.tensor([]).new_empty(1).uniform_().item())  # Simple timestamp
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error in classification with summary: {str(e)}"
        )

# Individual dataset endpoints - existing ones
@app.post("/classify/pathmnist")
async def classify_pathmnist(file: UploadFile = File(...)):
    """Classify colon pathology image."""
    return classify_for_dataset("pathmnist", file)

@app.post("/classify/retinamnist")
async def classify_retinamnist(file: UploadFile = File(...)):
    """Classify retinal OCT image."""
    return classify_for_dataset("retinamnist", file)

@app.post("/classify/dermamnist")
async def classify_dermamnist(file: UploadFile = File(...)):
    """Classify dermatology image."""
    return classify_for_dataset("dermamnist", file)

@app.post("/classify/bloodmnist")
async def classify_bloodmnist(file: UploadFile = File(...)):
    """Classify blood cell image."""
    return classify_for_dataset("bloodmnist", file)

# New combined endpoints with LLM summary
@app.post("/classify_with_summary/pathmnist")
async def classify_pathmnist_with_summary(file: UploadFile = File(...)):
    """Classify colon pathology image with medical summary."""
    return await classify_with_summary("pathmnist", file)

@app.post("/classify_with_summary/retinamnist")
async def classify_retinamnist_with_summary(file: UploadFile = File(...)):
    """Classify retinal OCT image with medical summary."""
    return await classify_with_summary("retinamnist", file)

@app.post("/classify_with_summary/dermamnist")
async def classify_dermamnist_with_summary(file: UploadFile = File(...)):
    """Classify dermatology image with medical summary."""
    return await classify_with_summary("dermamnist", file)

@app.post("/classify_with_summary/bloodmnist")
async def classify_bloodmnist_with_summary(file: UploadFile = File(...)):
    """Classify blood cell image with medical summary."""
    return await classify_with_summary("bloodmnist", file)


@app.post("/validation/submit")
async def submit_validation(validation_data: Dict):
    """Store doctor validation for retraining."""
    try:
        # Add timestamp if not present
        if "timestamp" not in validation_data:
            validation_data["timestamp"] = datetime.now().isoformat()
        
        # Store validation data (append to JSONL file)
        os.makedirs(os.path.dirname(VALIDATION_STORAGE_PATH) if os.path.dirname(VALIDATION_STORAGE_PATH) else ".", exist_ok=True)
        
        with open(VALIDATION_STORAGE_PATH, "a") as f:
            f.write(json.dumps(validation_data) + "\n")
        
        return {
            "status": "success",
            "message": "Validation data stored successfully",
            "validation_id": validation_data["timestamp"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing validation: {str(e)}")

@app.get("/validation/stats")
async def get_validation_stats():
    """Get validation statistics for retraining insights."""
    try:
        if not os.path.exists(VALIDATION_STORAGE_PATH):
            return {"total_validations": 0, "accuracy_by_dataset": {}}
        
        validations = []
        with open(VALIDATION_STORAGE_PATH, "r") as f:
            for line in f:
                validations.append(json.loads(line.strip()))
        
        # Calculate statistics
        stats = {
            "total_validations": len(validations),
            "accuracy_by_dataset": {},
            "common_misclassifications": {}
        }
        
        for dataset in MODEL_CONFIGS.keys():
            dataset_validations = [v for v in validations if v["dataset"] == dataset]
            if dataset_validations:
                correct = len([v for v in dataset_validations if v["doctor_assessment"] == "Correct Diagnosis"])
                stats["accuracy_by_dataset"][dataset] = {
                    "total": len(dataset_validations),
                    "correct": correct,
                    "accuracy": correct / len(dataset_validations)
                }
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/validation/export")
async def export_validation_data():
    """Export validation data for retraining."""
    try:
        if not os.path.exists(VALIDATION_STORAGE_PATH):
            return {"message": "No validation data available"}
        
        validations = []
        with open(VALIDATION_STORAGE_PATH, "r") as f:
            for line in f:
                validations.append(json.loads(line.strip()))
        
        return {
            "total_records": len(validations),
            "validation_data": validations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "medapp.service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )