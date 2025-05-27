from pydantic import BaseModel, Field # type: ignore
from typing import List, Dict, Optional, Union, Any

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
    results: List[Union[VerificationResult, Dict[str, Any]]]
    summary: Optional[Dict[str, Any]] = None

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
    overall_result: Dict[str, Any]
    page_results: List[Dict[str, Any]]
    total_pages: int
    clear_pages: int
    unclear_pages: int
    file_type: str
    filename: Optional[str] = None

class BatchClarityResult(BaseModel):
    results: List[Union[ClarityResult, PDFClarityResult, Dict[str, Any]]]
    summary: Dict[str, Any]

# Classification models
class ClassificationResult(BaseModel):
    class_name: str = Field(alias='class')
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    message: str
    file_type: str
    filename: Optional[str] = None

class PDFClassificationResult(BaseModel):
    overall_result: Dict[str, Any]
    page_results: List[Dict[str, Any]]
    file_type: str
    total_pages: int
    filename: Optional[str] = None

class BatchClassificationResult(BaseModel):
    results: List[Union[ClassificationResult, PDFClassificationResult, Dict[str, Any]]]
    summary: Dict[str, Any]