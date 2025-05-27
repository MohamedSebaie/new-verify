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

class LayoutRegion(BaseModel):
    """Model for a detected region in an ID layout"""
    region_type: str  # e.g., "photo_zone", "mrz_zone", "signature_zone", etc.
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    crop_path: Optional[str] = None  # Path to the cropped image
    orientation: Optional[str] = None  # "portrait" or "landscape"
    quality_score: Optional[float] = None  # Quality assessment of the region
    ocr_result: Optional[OCRResult] = None  # OCR result for this region

class LayoutDetectionResult(BaseModel):
    """Model for ID layout detection result"""
    filename: str
    status: str  # "success" or "error"
    message: str
    layout_type: Optional[str] = None  # e.g., "standard_id", "old_format", "digital_id", etc.
    orientation: Optional[str] = None  # "portrait" or "landscape"
    regions: List[LayoutRegion] = []
    output_folder: Optional[str] = None
    annotated_image_path: Optional[str] = None
    processing_time_ms: Optional[float] = None
    ocr_enabled: Optional[bool] = False  # Whether OCR was performed
    ocr_summary: Optional[Dict[str, Any]] = None  # Summary of OCR results

class BatchLayoutResult(BaseModel):
    """Model for batch layout detection results"""
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]