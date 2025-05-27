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

class IDElementDetection(BaseModel):
    """Model for a detected ID element"""
    element_type: str  # e.g., "ID_Front", "ID_Back", "Agent_Signature", etc.
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    crop_path: Optional[str] = None  # Path to the cropped image
    ocr_result: Optional[OCRResult] = None  # OCR result for this element

class NationalIDResult(BaseModel):
    """Model for National ID processing result"""
    filename: str
    status: str  # "success" or "error"
    message: str
    id_type: Optional[str] = None  # "front", "back", or "both"
    elements: List[IDElementDetection] = []
    output_folder: Optional[str] = None
    annotated_image_path: Optional[str] = None
    processing_time_ms: Optional[float] = None
    ocr_enabled: Optional[bool] = False  # Whether OCR was performed
    ocr_summary: Optional[Dict[str, Any]] = None  # Summary of OCR results

class BatchNationalIDResult(BaseModel):
    """Model for batch processing results"""
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]