from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from .database import Base

class OCRResult(Base):
    __tablename__ = "ocr_results"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    pan_number = Column(String(50), nullable=True)
    name = Column(String(255), nullable=True)
    father_name = Column(String(255), nullable=True)
    dob = Column(String(50), nullable=True)
    raw_results = Column(Text, nullable=True)  # JSON string of full results
    processing_time = Column(String(50), nullable=True)
    is_valid = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
