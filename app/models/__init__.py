# Import verification models
from .verification import (
    Base64File,
    Detection,
    VerificationResult,  # Export the original name
    BatchVerificationResult,  # Export the original name
)

from .detection import Detection, DetectionResult, BatchDetectionResult, Base64File
# Create the aliases that other parts of the code might expect
ProcessingResult = VerificationResult
BatchProcessingResult = BatchVerificationResult

# Add the missing ProcessingConfig class definition
from pydantic import BaseModel, Field # type: ignore
from typing import Optional, Dict

class ProcessingConfig(BaseModel):
    confidence_threshold: float = Field(default=0.25, ge=0, le=1.0)
    upper_conf_threshold: float = Field(default=0.5, ge=0, le=1.0)
    upper_part_ratio: float = Field(default=0.3, ge=0, le=1.0)
    image_size: int = Field(default=1120, ge=320, le=1920)

# Add the missing BatchFileResult class definition
class BatchFileResult(BaseModel):
    filename: str
    file_type: str
    result: ProcessingResult
    error: Optional[str] = None


# Import clarity models
from .clarity import (
    ClarityResult,
    PageResult,
    PDFClarityResult,
    BatchClarityResult,
    BatchClaritySummary,
    ClarityConfig
)

# Import classification models
from .classification import (
    ClassificationResult,
    PDFClassificationResult,
    BatchClassificationResult,
    ClassificationConfig
)

# Define what should be importable from this module
__all__ = [
    # Verification models (both original names and aliases)
    "Base64File",
    "ProcessingConfig",
    "Detection",
    "VerificationResult",
    "BatchVerificationResult",
    "ProcessingResult",
    "BatchFileResult",
    "BatchProcessingResult",
    
    # Clarity models
    "ClarityResult",
    "PageResult",
    "PDFClarityResult",
    "BatchClarityResult",
    "BatchClaritySummary",
    "ClarityConfig",
    
    # Classification models
    "ClassificationResult",
    "PDFClassificationResult",
    "BatchClassificationResult",
    "ClassificationConfig"
]