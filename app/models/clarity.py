from pydantic import BaseModel, Field # type: ignore
from typing import List, Optional, Dict, Union
from datetime import datetime

class ClarityResult(BaseModel):
    is_clear: bool
    confidence: float
    message: str
    file_type: str
    output_file_path: Optional[str] = None
    base64_output: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None

class PageResult(ClarityResult):
    page_number: int
    page_confidence: float

class PDFClarityResult(BaseModel):
    overall_result: ClarityResult
    page_results: List[PageResult]
    file_type: str = "pdf"
    total_pages: int
    clear_pages: int
    unclear_pages: int

class BatchClarityResult(BaseModel):
    filename: str
    result: Union[ClarityResult, PDFClarityResult]
    error: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    file_size: int
    file_type: str

class BatchClaritySummary(BaseModel):
    results: List[BatchClarityResult]
    summary: Dict[str, Union[int, float]] = Field(
        default_factory=lambda: {
            "total_processed": 0,
            "clear_documents": 0,
            "unclear_documents": 0,
            "errors": 0,
            "average_confidence": 0.0
        }
    )
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float

class ClarityConfig(BaseModel):
    confidence_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Minimum confidence threshold for clarity detection"
    )
    image_size: int = Field(
        default=224,
        description="Image size for ViT model input"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for processing multiple images"
    )