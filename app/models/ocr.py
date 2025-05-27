from pydantic import BaseModel, Field # type: ignore
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for OCR processing. If not provided, default prompt will be used."
    )
    enable_ocr: bool = Field(
        default=False,
        description="Whether to perform OCR on the processed images"
    )

class OCRResult(BaseModel):
    """Model for OCR processing result"""
    success: bool
    extracted_text: str
    confidence: Optional[float] = None
    error: Optional[str] = None
    image_path: Optional[str] = None
    prompt_used: Optional[str] = None
    model_response: Optional[Dict[str, Any]] = None

class RegionWithOCR(BaseModel):
    """Model for layout region with OCR results"""
    # Original region fields
    region_type: Optional[str] = None
    element_type: Optional[str] = None
    confidence: float
    bbox: List[float]
    crop_path: Optional[str] = None
    quality_score: Optional[float] = None
    
    # OCR result
    ocr_result: Optional[OCRResult] = None

class SingleImageOCRResult(BaseModel):
    """Result for single image OCR"""
    filename: str
    ocr_result: OCRResult
    processing_time_ms: Optional[float] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)

class BatchOCRResult(BaseModel):
    """Result for batch OCR processing"""
    results: List[SingleImageOCRResult]
    summary: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
    )
    processing_time_ms: Optional[float] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)

class LayoutDetectionWithOCRResult(BaseModel):
    """Enhanced layout detection result with OCR"""
    filename: str
    status: str
    message: str
    layout_type: Optional[str] = None
    orientation: Optional[str] = None
    regions: List[RegionWithOCR] = []
    output_folder: Optional[str] = None
    annotated_image_path: Optional[str] = None
    processing_time_ms: Optional[float] = None
    ocr_enabled: bool = False
    ocr_summary: Optional[Dict[str, Any]] = None

class NationalIDWithOCRResult(BaseModel):
    """Enhanced National ID result with OCR"""
    filename: str
    status: str
    message: str
    id_type: Optional[str] = None
    elements: List[RegionWithOCR] = []
    output_folder: Optional[str] = None
    annotated_image_path: Optional[str] = None
    processing_time_ms: Optional[float] = None
    ocr_enabled: bool = False
    ocr_summary: Optional[Dict[str, Any]] = None

class RegionSpecificPrompts(BaseModel):
    """Model for region-specific OCR prompts"""
    photo_zone: Optional[str] = None
    mrz_zone: Optional[str] = None
    signature_zone: Optional[str] = None
    personal_info_zone: Optional[str] = None
    document_number_zone: Optional[str] = None
    fingerprint_zone: Optional[str] = None
    barcode_zone: Optional[str] = None
    hologram_zone: Optional[str] = None
    header_zone: Optional[str] = None
    
    # National ID elements
    ID_Front: Optional[str] = None
    ID_Back: Optional[str] = None
    Agent_Signature: Optional[str] = None
    Store_Stamp: Optional[str] = None
    Photo: Optional[str] = None
    Name: Optional[str] = None
    Date_of_Birth: Optional[str] = None
    ID_Number: Optional[str] = None
    Signature: Optional[str] = None
    Address: Optional[str] = None

class OCRConfig(BaseModel):
    """OCR configuration model"""
    enabled: bool = Field(
        default=False,
        description="Enable OCR processing"
    )
    default_prompt: str = Field(
        default="You are an Arabic OCR expert. Extract all text from this image accurately:",
        description="Default prompt for OCR processing"
    )
    region_prompts: Optional[RegionSpecificPrompts] = Field(
        default=None,
        description="Region-specific prompts for different ID areas"
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum tokens for OCR response"
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for OCR model"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for OCR requests"
    )