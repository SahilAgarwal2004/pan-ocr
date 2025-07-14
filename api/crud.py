from sqlalchemy.orm import Session
from . import models, schemas
from typing import List, Optional
import json

def create_ocr_result(db: Session, result: schemas.OCRResultCreate) -> models.OCRResult:
    db_result = models.OCRResult(**result.dict())
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result

def get_ocr_result(db: Session, result_id: int) -> Optional[models.OCRResult]:
    return db.query(models.OCRResult).filter(models.OCRResult.id == result_id).first()

def get_ocr_results(db: Session, skip: int = 0, limit: int = 100) -> List[models.OCRResult]:
    return db.query(models.OCRResult).offset(skip).limit(limit).all()

def get_ocr_results_by_filename(db: Session, filename: str) -> List[models.OCRResult]:
    return db.query(models.OCRResult).filter(models.OCRResult.filename == filename).all()

def update_ocr_result(db: Session, result_id: int, result_update: schemas.OCRResultCreate) -> Optional[models.OCRResult]:
    db_result = get_ocr_result(db, result_id)
    if db_result:
        for key, value in result_update.dict(exclude_unset=True).items():
            setattr(db_result, key, value)
        db.commit()
        db.refresh(db_result)
    return db_result

def delete_ocr_result(db: Session, result_id: int) -> bool:
    db_result = get_ocr_result(db, result_id)
    if db_result:
        db.delete(db_result)
        db.commit()
        return True
    return False
