from pydantic import BaseModel, Field # type: ignore
from typing import Dict, List, Optional, Union
from datetime import datetime

class ClassificationResult(BaseModel):
    class_name: str = Field(alias='class')
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    message: str
    file_type: str
    filename: Optional[str] = None
    page_number: Optional[int] = None

class PDFClassificationResult(BaseModel):
    overall_result: Dict
    page_results: List[Dict]
    file_type: str
    filename: Optional[str] = None

class BatchClassificationResult(BaseModel):
    results: List[Union[ClassificationResult, PDFClassificationResult, Dict]]
    summary: Dict
    processed_at: datetime = Field(default_factory=datetime.utcnow)

class ClassificationConfig(BaseModel):
    threshold: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Minimum confidence threshold for classification"
    )