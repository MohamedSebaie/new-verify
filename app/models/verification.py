from pydantic import BaseModel, Field # type: ignore
from typing import List, Dict, Optional, Union

# Common models
class Base64File(BaseModel):
    filename: str
    content: str
    file_type: str

# Verification models
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class VerificationResult(BaseModel):
    status: str
    message: str
    detections: List[Detection]
    file_type: str
    filename: Optional[str] = None

class BatchVerificationResult(BaseModel):
    total_files: int
    successful_files: int
    rejected_files: int
    failed_files: int
    results: List[Union[VerificationResult, Dict]]

# Clarity models
class ClarityResult(BaseModel):
    is_clear: bool
    confidence: float
    message: str
    file_type: str
    filename: Optional[str] = None

class PageResult(BaseModel):
    is_clear: bool
    confidence: float
    message: str
    page_number: int

class PDFClarityResult(BaseModel):
    overall_result: Dict
    page_results: List[Dict]
    total_pages: int
    clear_pages: int
    unclear_pages: int
    file_type: str
    filename: Optional[str] = None

class BatchClarityResult(BaseModel):
    results: List[Union[ClarityResult, PDFClarityResult, Dict]]
    summary: Dict

# Classification models
class ClassificationResult(BaseModel):
    class_name: str = Field(alias='class')
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    message: str
    file_type: str
    filename: Optional[str] = None

class ProcessingConfig(BaseModel):
   confidence_threshold: float = Field(default=0.25, ge=0, le=1.0)
   upper_conf_threshold: float = Field(default=0.5, ge=0, le=1.0)
   upper_part_ratio: float = Field(default=0.3, ge=0, le=1.0)
   image_size: int = Field(default=1120, ge=320, le=1920)

class PDFClassificationResult(BaseModel):
    overall_result: Dict
    page_results: List[Dict]
    file_type: str
    total_pages: int
    filename: Optional[str] = None

class BatchClassificationResult(BaseModel):
    results: List[Union[ClassificationResult, PDFClassificationResult, Dict]]
    summary: Dict