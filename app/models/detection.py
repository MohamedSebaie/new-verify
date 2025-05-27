from pydantic import BaseModel # type: ignore
from typing import List, Dict, Any, Optional

class OCRResult(BaseModel):
    """Model for OCR processing result"""
    success: bool
    extracted_text: str
    confidence: Optional[float] = None
    error: Optional[str] = None
    image_path: Optional[str] = None
    prompt_used: Optional[str] = None
    model_response: Optional[Dict[str, Any]] = None

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    crop_path: Optional[str] = None  # Path to cropped detection
    ocr_result: Optional[OCRResult] = None  # OCR result for this detection

class DetectionResult(BaseModel):
    filename: str
    file_type: str
    output_folder: str
    pages_processed: int
    results: List[Dict[str, Any]]
    ocr_enabled: Optional[bool] = False  # Whether OCR was performed
    ocr_summary: Optional[Dict[str, Any]] = None  # Summary of OCR results
    processing_time_ms: Optional[float] = None

class BatchDetectionResult(BaseModel):
    results: List[Dict[str, Any]]
    total_files: int
    total_detections: int

class Base64File(BaseModel):
    base64: str
    filename: Optional[str] = "image.jpg"