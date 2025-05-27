from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Base Models
class PipelineStage(BaseModel):
    """Base model for pipeline stage results"""
    stage_name: str
    status: str  # "success", "error", "warning"
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

# KYC Pipeline Models
class SignatureVerification(BaseModel):
    """Model for KYC signature verification results"""
    agent_signature: bool = Field(description="Whether agent signature is detected")
    customer_signature: bool = Field(description="Whether customer signature is detected") 
    store_stamp: bool = Field(description="Whether store stamp is detected")
    all_signatures_verified: bool = Field(description="Whether all required signatures are present")

class KYCProcessingSummary(BaseModel):
    """Summary of KYC processing results"""
    signatures_found: int
    total_signatures_required: int
    fields_extracted: int
    ocr_available: bool

class KYCPipelineResult(BaseModel):
    """Complete KYC pipeline result"""
    status: str
    signature_verification: SignatureVerification
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    processing_summary: KYCProcessingSummary
    error: Optional[str] = None

# TC Pipeline Models
class TCPipelineResult(BaseModel):
    """Complete TC (Trade Certificate) pipeline result"""
    status: str
    is_clear: bool
    confidence: float = Field(ge=0.0, le=1.0)
    message: str
    file_type: str
    error: Optional[str] = None

# NID Pipeline Models
class ElementsDetected(BaseModel):
    """Model for detected NID elements"""
    id_front: bool = Field(description="Whether ID front is detected")
    id_back: bool = Field(description="Whether ID back is detected")
    signature: bool = Field(description="Whether signature is detected")
    stamp: bool = Field(description="Whether stamp is detected")

class NIDExtractedData(BaseModel):
    """Model for extracted NID data"""
    front_fields: Dict[str, str] = Field(default_factory=dict)
    back_fields: Dict[str, str] = Field(default_factory=dict)

class NIDProcessingSummary(BaseModel):
    """Summary of NID processing results"""
    layout_regions_found: int
    nid_elements_found: int
    front_fields_extracted: int
    back_fields_extracted: int
    ocr_available: bool

class NIDPipelineResult(BaseModel):
    """Complete NID pipeline result"""
    status: str
    elements_detected: ElementsDetected
    extracted_data: NIDExtractedData
    processing_summary: NIDProcessingSummary
    error: Optional[str] = None

# Validation Models
class FieldValidation(BaseModel):
    """Model for individual field validation result"""
    kyc_value: str
    nid_value: str
    is_match: bool
    similarity_score: float = Field(ge=0.0, le=1.0)

class ValidationResults(BaseModel):
    """Model for cross-validation results between KYC and NID"""
    validation_performed: bool
    overall_validation: bool = False
    field_validations: Dict[str, FieldValidation] = Field(default_factory=dict)
    discrepancies: List[str] = Field(default_factory=list)
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    match_percentage: Optional[float] = None
    reason: Optional[str] = None

# Classification Models
class ClassificationResult(BaseModel):
    """Model for document classification result"""
    class_name: str = Field(alias='class')
    class_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    message: str
    file_type: str

# Pipeline Results Container Models
class PipelineResults(BaseModel):
    """Container for all pipeline stage results"""
    verification: Optional[Dict[str, Any]] = None
    field_detection: Optional[Dict[str, Any]] = None
    clarity: Optional[Dict[str, Any]] = None
    id_layout: Optional[Dict[str, Any]] = None
    nid_detection: Optional[Dict[str, Any]] = None
    ocr: Optional[Dict[str, Any]] = None

class DocumentPipelineResult(BaseModel):
    """Result for a single document's pipeline processing"""
    document_type: str
    pipeline_stages: PipelineResults
    final_result: Union[KYCPipelineResult, TCPipelineResult, NIDPipelineResult]

# Master Pipeline Models
class ProcessingSummary(BaseModel):
    """Summary of overall processing"""
    total_processing_time_ms: float
    document_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    pipeline_success: bool
    ocr_enabled: bool
    validation_performed: bool
    error: Optional[str] = None

class MasterPipelineResult(BaseModel):
    """Complete result from master pipeline processing"""
    filename: str
    processing_timestamp: float
    classification: ClassificationResult
    pipeline_results: Dict[str, DocumentPipelineResult]
    validation_results: ValidationResults
    processing_summary: ProcessingSummary
    error: Optional[str] = None

# Multi-Document Models
class MultiDocumentProcessingSummary(BaseModel):
    """Summary for multi-document processing"""
    total_processing_time_ms: float
    successful_documents: int
    failed_documents: int
    validation_performed: bool
    overall_success: bool
    error: Optional[str] = None

class PairProcessingInfo(BaseModel):
    """Information about KYC-NID pair processing"""
    kyc_processed: bool
    nid_processed: bool
    both_successful: bool
    validation_available: bool

class MultiDocumentPipelineResult(BaseModel):
    """Result from processing multiple documents"""
    total_documents: int
    processing_timestamp: float
    document_results: Dict[str, MasterPipelineResult]
    cross_validation: ValidationResults
    processing_summary: MultiDocumentProcessingSummary
    pair_processing: Optional[PairProcessingInfo] = None  # For KYC-NID pairs
    error: Optional[str] = None

# Service Status Models
class ServiceStatus(BaseModel):
    """Status of individual service component"""
    available: bool
    model_loaded: Optional[bool] = None
    ocr_integration: Optional[bool] = None
    supported_elements: Optional[List[str]] = None
    supported_regions: Optional[List[str]] = None
    supported_classes: Optional[List[str]] = None
    required_elements: Optional[List[str]] = None
    provider: Optional[str] = None
    text_cleaning: Optional[bool] = None
    supported_validations: Optional[List[str]] = None

class PipelineStatus(BaseModel):
    """Complete pipeline status information"""
    pipeline_services: Dict[str, ServiceStatus]
    supported_formats: Dict[str, List[str]]
    pipeline_capabilities: Dict[str, bool]

# Document Type Information Models
class PipelineStageInfo(BaseModel):
    """Information about a pipeline stage"""
    stage: str
    description: str
    required_elements: Optional[List[str]] = None

class DocumentTypeInfo(BaseModel):
    """Information about a document type and its processing"""
    full_name: str
    pipeline_stages: List[PipelineStageInfo]
    output_fields: List[str]

class ValidationCapabilityInfo(BaseModel):
    """Information about validation capabilities"""
    description: str
    validated_fields: List[str]
    similarity_threshold: float

class SupportedDocumentTypes(BaseModel):
    """Complete information about supported document types"""
    document_types: Dict[str, DocumentTypeInfo]
    validation_capabilities: Dict[str, ValidationCapabilityInfo]

# Health Check Models
class ComponentHealth(BaseModel):
    """Health status of individual component"""
    status: str  # "healthy", "warning", "unhealthy", "unavailable"
    model_loaded: Optional[bool] = None
    available: Optional[bool] = None
    error: Optional[str] = None

class PipelineHealth(BaseModel):
    """Overall pipeline health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    service: str
    timestamp: float
    components: Dict[str, ComponentHealth]
    unhealthy_services: Optional[List[str]] = None
    error: Optional[str] = None

# Request Models
class Base64FileInput(BaseModel):
    """Model for base64 file input"""
    filename: str
    content: str
    file_type: str

class SingleDocumentRequest(BaseModel):
    """Request model for single document processing"""
    base64_file: Base64FileInput
    enable_validation: bool = False
    classification_only: bool = False

class MultiDocumentRequest(BaseModel):
    """Request model for multiple document processing"""
    base64_files: List[Base64FileInput]
    enable_validation: bool = True
    parallel_processing: bool = True

class KYCNIDPairRequest(BaseModel):
    """Request model for KYC-NID pair processing"""
    kyc_file: Base64FileInput
    nid_file: Base64FileInput

class ValidationRequest(BaseModel):
    """Request model for standalone validation"""
    kyc_data: Dict[str, Any]
    nid_data: Dict[str, Any]

# Response Models for API endpoints
class ClassificationOnlyResponse(BaseModel):
    """Response for classification-only processing"""
    mode: str
    filename: str
    classification: ClassificationResult
    message: str

class ValidationOnlyResponse(BaseModel):
    """Response for standalone validation"""
    validation_type: str
    timestamp: float
    validation_results: ValidationResults

# Error Models
class PipelineError(BaseModel):
    """Model for pipeline error responses"""
    error_type: str
    message: str
    stage: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: float

# Configuration Models
class PipelineConfig(BaseModel):
    """Configuration model for pipeline settings"""
    enable_parallel_processing: bool = True
    max_concurrent_documents: int = 5
    ocr_timeout_seconds: int = 30
    validation_similarity_threshold: float = 0.85
    classification_confidence_threshold: float = 0.5
    auto_cleanup_temp_files: bool = True
    enable_detailed_logging: bool = True

# Metrics Models
class ProcessingMetrics(BaseModel):
    """Model for processing performance metrics"""
    document_type: str
    stage_timings: Dict[str, float]  # stage_name -> time_ms
    total_processing_time: float
    ocr_processing_time: Optional[float] = None
    validation_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
class BatchMetrics(BaseModel):
    """Model for batch processing metrics"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    average_processing_time: float
    total_batch_time: float
    document_metrics: List[ProcessingMetrics]
    
# Export all models
__all__ = [
    # Base Models
    "PipelineStage",
    
    # KYC Models
    "SignatureVerification",
    "KYCProcessingSummary", 
    "KYCPipelineResult",
    
    # TC Models
    "TCPipelineResult",
    
    # NID Models
    "ElementsDetected",
    "NIDExtractedData",
    "NIDProcessingSummary",
    "NIDPipelineResult",
    
    # Validation Models
    "FieldValidation",
    "ValidationResults",
    
    # Classification Models  
    "ClassificationResult",
    
    # Pipeline Container Models
    "PipelineResults",
    "DocumentPipelineResult",
    
    # Master Pipeline Models
    "ProcessingSummary",
    "MasterPipelineResult",
    
    # Multi-Document Models
    "MultiDocumentProcessingSummary",
    "PairProcessingInfo", 
    "MultiDocumentPipelineResult",
    
    # Status Models
    "ServiceStatus",
    "PipelineStatus",
    "DocumentTypeInfo",
    "SupportedDocumentTypes",
    "ComponentHealth",
    "PipelineHealth",
    
    # Request Models
    "Base64FileInput",
    "SingleDocumentRequest", 
    "MultiDocumentRequest",
    "KYCNIDPairRequest",
    "ValidationRequest",
    
    # Response Models
    "ClassificationOnlyResponse",
    "ValidationOnlyResponse",
    
    # Error Models
    "PipelineError",
    
    # Configuration Models
    "PipelineConfig",
    
    # Metrics Models
    "ProcessingMetrics",
    "BatchMetrics"
]