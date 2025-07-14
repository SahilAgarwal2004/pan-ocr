from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import shutil
import time
import json
import uuid
from pathlib import Path
from typing import List
import logging

from . import crud, models, schemas
from .database import SessionLocal, engine, get_db
from .config import settings

# Import your inference function
import sys
import os

# Get the path to ocr_project directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from the ocr_model folder
from ocr_model.inference import pan_ocr_inference

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create upload directory
Path(settings.upload_dir).mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PAN OCR API",
    description="API for PAN Card OCR processing with YOLO detection and text extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    if file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {settings.max_file_size} bytes"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type {file_ext} not supported. Allowed types: {settings.allowed_extensions}"
        )
    
    return True

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to disk"""
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    saved_filename = f"{file_id}{file_ext}"
    file_path = os.path.join(settings.upload_dir, saved_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )

@app.post("/ocr/process", response_model=schemas.OCRProcessResponse)
async def process_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Process uploaded PAN card image and extract text"""
    try:
        # Validate file
        validate_file(file)
        
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        
        # Process with OCR
        start_time = time.time()
        
        # Read file content for inference
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Run inference
        ocr_results = pan_ocr_inference(file_content)
        
        end_time = time.time()
        processing_time = f"{end_time - start_time:.3f}s"
        
        # Extract individual fields
        pan_number = ocr_results.get("PAN Number", {}).get("text", "")
        name = ocr_results.get("Name", {}).get("text", "")
        father_name = ocr_results.get("Father Name", {}).get("text", "")
        dob = ocr_results.get("DOB", {}).get("text", "")
        
        # Check if results are valid
        is_valid = bool(pan_number and name)
        
        # Create database entry
        db_result = schemas.OCRResultCreate(
            filename=file.filename,
            file_path=file_path,
            pan_number=pan_number,
            name=name,
            father_name=father_name,
            dob=dob,
            raw_results=json.dumps(ocr_results),
            processing_time=processing_time,
            is_valid=is_valid
        )
        
        # Save to database
        saved_result = crud.create_ocr_result(db, db_result)
        
        return schemas.OCRProcessResponse(
            id=saved_result.id,
            filename=file.filename,
            results=ocr_results,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@app.get("/ocr/results/{result_id}", response_model=schemas.OCRResultResponse)
def get_ocr_result(result_id: int, db: Session = Depends(get_db)):
    """Get OCR result by ID"""
    db_result = crud.get_ocr_result(db, result_id)
    if db_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR result not found"
        )
    return db_result

@app.get("/ocr/results", response_model=List[schemas.OCRResultResponse])
def get_ocr_results(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Get list of OCR results"""
    results = crud.get_ocr_results(db, skip=skip, limit=limit)
    return results

@app.get("/ocr/results/filename/{filename}", response_model=List[schemas.OCRResultResponse])
def get_results_by_filename(filename: str, db: Session = Depends(get_db)):
    """Get OCR results by filename"""
    results = crud.get_ocr_results_by_filename(db, filename)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found for this filename"
        )
    return results

@app.get("/ocr/download/{result_id}")
def download_processed_image(result_id: int, db: Session = Depends(get_db)):
    """Download the original uploaded image"""
    db_result = crud.get_ocr_result(db, result_id)
    if db_result is None or not db_result.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    if not os.path.exists(db_result.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File no longer exists on server"
        )
    
    return FileResponse(db_result.file_path, filename=db_result.filename)

@app.delete("/ocr/results/{result_id}")
def delete_ocr_result(result_id: int, db: Session = Depends(get_db)):
    """Delete OCR result and associated file"""
    db_result = crud.get_ocr_result(db, result_id)
    if db_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR result not found"
        )
    
    # Delete file if exists
    if db_result.file_path and os.path.exists(db_result.file_path):
        try:
            os.remove(db_result.file_path)
        except Exception as e:
            logger.warning(f"Could not delete file {db_result.file_path}: {str(e)}")
    
    # Delete database entry
    success = crud.delete_ocr_result(db, result_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete OCR result"
        )
    
    return {"message": "OCR result deleted successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PAN OCR API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
