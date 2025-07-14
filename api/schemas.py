from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class OCRResultBase(BaseModel):
    filename: str
    pan_number: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None
    is_valid: bool = False

class OCRResultCreate(OCRResultBase):
    file_path: Optional[str] = None
    raw_results: Optional[str] = None
    processing_time: Optional[str] = None

class OCRResultResponse(OCRResultBase):
    id: int
    file_path: Optional[str] = None
    processing_time: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class OCRProcessResponse(BaseModel):
    id: int
    filename: str
    results: Dict[str, Any]
    processing_time: str
    message: str = "OCR processing completed successfully"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
