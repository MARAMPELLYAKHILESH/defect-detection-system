"""
FastAPI Application for Defect Detection System
Provides REST API endpoints for image upload and defect detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from typing import List, Optional
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
try:
    from src.preprocessing import ImagePreprocessor
    from src.defect_detection import DefectDetector
except ImportError:
    # Fallback: try direct import
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from preprocessing import ImagePreprocessor
    from defect_detection import DefectDetector

# Initialize FastAPI app
app = FastAPI(
    title="Defect Detection API",
    description="Computer Vision-based defect detection system for manufacturing quality control",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
preprocessor = ImagePreprocessor(target_size=(640, 480))
detector = DefectDetector(min_defect_area=100, max_defect_area=10000)

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)


class DetectionResponse(BaseModel):
    """Response model for defect detection"""
    success: bool
    image_id: str
    num_defects: int
    defect_percentage: float
    quality_assessment: str
    defects: List[dict]
    processed_image_url: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    try:
        with open("defect_detection_interface.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {
            "status": "online",
            "message": "Defect Detection API is running",
            "version": "1.0.0",
            "docs": "/docs"
        }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": "1.0.0"
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_defects(file: UploadFile = File(...)):
    """
    Detect defects in uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detection results with annotated image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save original
        original_path = f"data/raw/{image_id}_original.jpg"
        cv2.imwrite(original_path, original_image)
        
        # Preprocess image
        resized = preprocessor.resize_image(original_image)
        gray = preprocessor.convert_to_grayscale(resized)
        denoised = preprocessor.denoise(gray, method='gaussian')
        enhanced = preprocessor.enhance_contrast(denoised, method='clahe')
        thresh = preprocessor.threshold_image(enhanced, method='otsu')
        binary = preprocessor.apply_morphology(thresh, operation='closing')
        
        # Detect defects
        results = detector.detect_defects(resized, binary)
        
        # Save annotated image
        processed_path = f"results/{image_id}_annotated.jpg"
        cv2.imwrite(processed_path, results["annotated_image"])
        
        # Prepare response
        return {
            "success": True,
            "image_id": image_id,
            "num_defects": results["num_defects"],
            "defect_percentage": round(results["defect_percentage"], 2),
            "quality_assessment": results["quality_assessment"],
            "defects": results["defects"],
            "processed_image_url": f"/results/{image_id}_annotated.jpg",
            "timestamp": timestamp
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/results/{image_id}")
async def get_result_image(image_id: str):
    """
    Get processed result image
    
    Args:
        image_id: Unique image identifier
        
    Returns:
        Annotated image file
    """
    file_path = f"results/{image_id}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)


@app.post("/batch-detect")
async def batch_detect_defects(files: List[UploadFile] = File(...)):
    """
    Batch process multiple images
    
    Args:
        files: List of uploaded images
        
    Returns:
        List of detection results
    """
    results = []
    
    for file in files:
        try:
            result = await detect_defects(file)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"total": len(files), "results": results}


@app.get("/stats")
async def get_statistics():
    """
    Get system statistics
    
    Returns:
        Statistics about processed images
    """
    try:
        raw_count = len(os.listdir("data/raw"))
        results_count = len(os.listdir("results"))
        
        return {
            "total_images_processed": raw_count,
            "total_results_generated": results_count,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/cleanup")
async def cleanup_old_files():
    """
    Cleanup old processed files
    
    Returns:
        Cleanup status
    """
    try:
        # Clean raw data
        raw_files = os.listdir("data/raw")
        for file in raw_files:
            os.remove(os.path.join("data/raw", file))
        
        # Clean results
        result_files = os.listdir("results")
        for file in result_files:
            os.remove(os.path.join("results", file))
        
        return {
            "success": True,
            "message": f"Cleaned {len(raw_files)} raw files and {len(result_files)} result files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)