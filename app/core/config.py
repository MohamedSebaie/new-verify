# Updated app/core/config.py - Enhanced configuration for Master Pipeline

from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # ============= API Configuration =============
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "KYC Processing API - Master Pipeline"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "Comprehensive KYC Processing API with intelligent document routing and cross-validation"

    # ============= CORS Configuration =============
    ALLOWED_ORIGINS: str = "*"
    ALLOWED_METHODS: str = "*"
    ALLOWED_HEADERS: str = "*"

    # ============= Model Paths =============
    MODELS_DIR: str = "/home/ec2-user/Sebaie/KYC_with_OCR/app/models"
    
    # Verification Models
    YOLO_MODEL_PATH: str = os.path.join(MODELS_DIR, "verfication_best.pt")
    
    # Classification Models
    VIT_MODEL_PATH: str = os.path.join(MODELS_DIR, "KYC_VIT_Model.pt")
    DOCUMENT_CLASS_MODEL_PATH: str = os.path.join(MODELS_DIR, "KYC_Document_VIT_Model.pt")
    VIT_PROCESSOR_NAME: str = "vit-base-patch16-224"

    # Detection Models
    DETECTION_MODEL_PATH: str = os.path.join(MODELS_DIR, "detectionKyc.pt")
    DETECTION_OUTPUT_DIR: str = "output"
    
    # National ID Models
    NATIONAL_ID_MODEL_PATH: str = os.path.join(MODELS_DIR, "NID.pt")
    NATIONAL_ID_OUTPUT_DIR: str = "national_id_output"

    # ID Layout Models
    ID_LAYOUT_MODEL_PATH: str = os.path.join(MODELS_DIR, "layout_model.pt")
    ID_LAYOUT_OUTPUT_DIR: str = "id_layout_output"

    # ============= Master Pipeline Configuration =============
    
    # Pipeline Processing Settings
    ENABLE_PARALLEL_PROCESSING: bool = True
    MAX_CONCURRENT_DOCUMENTS: int = 5
    PIPELINE_TIMEOUT_SECONDS: int = 300  # 5 minutes total timeout
    STAGE_TIMEOUT_SECONDS: int = 60      # 1 minute per stage timeout
    
    # Validation Settings
    VALIDATION_SIMILARITY_THRESHOLD: float = 0.85
    VALIDATION_ENABLED_BY_DEFAULT: bool = True
    VALIDATION_REQUIRED_FIELDS: List[str] = ["name", "id_number", "address"]
    
    # Classification Settings
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.5
    CLASSIFICATION_RETRY_COUNT: int = 2
    
    # Pipeline Caching
    ENABLE_RESULT_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000     # Maximum cached results
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = True
    PERFORMANCE_LOG_THRESHOLD_MS: float = 5000  # Log if processing takes > 5 seconds
    
    # ============= File Configurations =============
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # Increased to 20MB for pipeline
    SUPPORTED_IMAGE_TYPES: List[str] = [
        ".jpg", ".jpeg", ".png", ".tiff", ".bmp", 
        ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"
    ]
    SUPPORTED_DOC_TYPES: List[str] = [".pdf"]

    # Temporary File Configuration
    TEMP_DIR: str = "temp"
    AUTO_CLEANUP_TEMP_FILES: bool = True
    TEMP_FILE_RETENTION_HOURS: int = 24

    # ============= OCR Configuration =============
    
    # Provider Selection and Fallback
    OCR_PROVIDER: str = "VLLM"  # Primary provider
    OCR_FALLBACK_PROVIDER: str = "OCI"  # Fallback provider
    OCR_ENABLED: bool = True
    OCR_BATCH_SIZE: int = 10
    OCR_AUTO_FALLBACK: bool = True  # Automatically switch providers on failure

    # OCI OCR Configuration
    OCI_CONFIG_FILE: str = "config"
    OCI_OCR_ENDPOINT: str = "https://modeldeployment.uk-london-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.uk-london-1.amaaaaaamefhmxaazdpztqubjm6gi5fyuuzlgo3xe2wog2rloff3ikykirmq/predict"
    OCI_DEFAULT_OCR_PROMPT: str = "You are an expert Arabic OCR system. Extract all text from this image. Return only the text without any explanation, description, or translation. For numbers, be precise with Arabic numerals."
    OCI_OCR_MAX_TOKENS: int = 500
    OCI_OCR_TEMPERATURE: float = 0.0
    OCI_OCR_TOP_P: float = 0.9
    OCI_OCR_TIMEOUT: int = 30

    # VLLM OCR Configuration
    VLLM_BASE_URL: str = "http://localhost:4003/v1"
    VLLM_API_KEY: str = "EMPTY"
    VLLM_MODEL: str = "MBZUAI/AIN"
    VLLM_AVAILABLE_MODELS: List[str] = ["Qwen/Qwen2.5-VL-7B-Instruct", "MBZUAI/AIN"]
    VLLM_DEFAULT_OCR_PROMPT: str = "You are an expert OCR system. Extract all text from this image accurately. Return only the extracted text without any explanations or descriptions."
    VLLM_OCR_MAX_TOKENS: int = 1000
    VLLM_OCR_TEMPERATURE: float = 0.1
    VLLM_OCR_TIMEOUT: int = 60

    # General OCR Settings
    DEFAULT_OCR_PROMPT: str = "You are an expert Arabic OCR system. Extract all text from this image. Return only the text without any explanation, description, or translation. For numbers, be precise with Arabic numerals."
    OCR_MAX_TOKENS: int = 500
    OCR_TEMPERATURE: float = 0.0
    OCR_TOP_P: float = 0.9
    OCR_TIMEOUT: int = 30
    OCR_TEXT_CLEANING_ENABLED: bool = True

    # ============= Pipeline-Specific OCR Prompts =============
    
    # KYC Field-Specific Prompts
    KYC_FIELD_PROMPTS: Dict[str, str] = {
        "customer_name": "Extract the customer's full name from this field:",
        "national_id": "Extract the national ID number from this field:",
        "customer_address": "Extract the complete address from this field:",
        "phone_number": "Extract the phone number from this field:",
        "date_of_birth": "Extract the date of birth from this field:",
        "signature": "Extract any text or name from this signature area:"
    }
    
    # NID Element-Specific Prompts
    NID_ELEMENT_PROMPTS: Dict[str, str] = {
        "Name": "Extract the full name from this area of the National ID:",
        "ID_Number": "Extract the National ID number from this area:",
        "Date_of_Birth": "Extract the date of birth from this area:",
        "Address": "Extract the complete address from this area:",
        "Signature": "Extract any text or name from this signature area:",
        "Photo": "Describe what you see in this photo area:",
        "ID_Front": "Extract any visible text from the front of this ID:",
        "ID_Back": "Extract any visible text from the back of this ID:"
    }
    
    # ID Layout Region-Specific Prompts
    ID_LAYOUT_REGION_PROMPTS: Dict[str, str] = {
        "photo_zone": "Extract any text visible in this photo area:",
        "mrz_zone": "Extract the machine-readable zone (MRZ) text line by line:",
        "signature_zone": "Extract any text or name from this signature area:",
        "personal_info_zone": "Extract all personal information from this area:",
        "document_number_zone": "Extract the document number from this area:",
        "fingerprint_zone": "Extract any text from this fingerprint area:",
        "barcode_zone": "Extract any text or numbers from this barcode area:",
        "hologram_zone": "Extract any text from this hologram area:",
        "header_zone": "Extract any header text from this area:"
    }

    # ============= Logging Configuration =============
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_ENABLED: bool = True
    LOG_FILE_PATH: str = "logs/kyc_pipeline.log"
    LOG_FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    
    # Pipeline-specific logging
    LOG_PIPELINE_STAGES: bool = True
    LOG_PROCESSING_TIMES: bool = True
    LOG_VALIDATION_DETAILS: bool = True
    LOG_OCR_RESULTS: bool = False  # Set to False to avoid logging sensitive data
    # ============= Deployment Configuration =============
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 4007
    WORKERS: int = 1  # Increase for production
    RELOAD: bool = True  # Set to False for production
    
    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True  # Set to False for production
    
    # Resource Limits
    MAX_MEMORY_USAGE_MB: int = 8192  # 8GB
    MAX_CPU_USAGE_PERCENTAGE: float = 80.0
    
    # ============= Feature Flags =============
    
    # Pipeline Features
    ENABLE_CLASSIFICATION_CACHING: bool = True
    ENABLE_OCR_CACHING: bool = True
    ENABLE_RESULT_VALIDATION: bool = True
    ENABLE_PARALLEL_OCR: bool = True
    ENABLE_BATCH_OPTIMIZATION: bool = True
    
    # API Features
    ENABLE_SWAGGER_UI: bool = True
    ENABLE_REDOC: bool = True
    ENABLE_CORS: bool = True
    ENABLE_REQUEST_LOGGING: bool = True
    
    # Advanced Features
    ENABLE_WEBHOOK_NOTIFICATIONS: bool = False
    ENABLE_ASYNC_PROCESSING: bool = True
    ENABLE_RESULT_STREAMING: bool = False

    # ============= Model Configuration =============
    
    # Model Loading Settings
    MODEL_PRELOAD_ON_STARTUP: bool = True
    MODEL_LAZY_LOADING: bool = False
    MODEL_CACHE_SIZE: int = 3  # Number of models to keep in memory
    MODEL_WARMUP_ENABLED: bool = True
    
    # Inference Settings
    INFERENCE_BATCH_SIZE: int = 1
    INFERENCE_DEVICE: str = "auto"  # auto, cpu, cuda
    INFERENCE_PRECISION: str = "fp32"  # fp32, fp16
    
    # Model Fallback
    ENABLE_MODEL_FALLBACK: bool = True
    MODEL_HEALTH_CHECK_INTERVAL: int = 300  # 5 minutes

    class Config:
        case_sensitive = True
        env_prefix = "KYC_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# ============= Configuration Validation =============

def validate_configuration():
    """Validate configuration settings and log warnings for potential issues"""
    import logging
    logger = logging.getLogger(__name__)
    
    warnings = []
    errors = []
    
    # Check model paths
    model_paths = [
        settings.YOLO_MODEL_PATH,
        settings.VIT_MODEL_PATH,
        settings.DOCUMENT_CLASS_MODEL_PATH,
        settings.DETECTION_MODEL_PATH,
        settings.NATIONAL_ID_MODEL_PATH,
        settings.ID_LAYOUT_MODEL_PATH
    ]
    
    for model_path in model_paths:
        if not Path(model_path).exists():
            warnings.append(f"Model file not found: {model_path}")
    
    # Check OCR configuration
    if settings.OCR_ENABLED:
        if settings.OCR_PROVIDER not in ["OCI", "VLLM"]:
            errors.append(f"Invalid OCR provider: {settings.OCR_PROVIDER}")
        
        if settings.OCR_PROVIDER == "OCI" and not Path(settings.OCI_CONFIG_FILE).exists():
            warnings.append(f"OCI config file not found: {settings.OCI_CONFIG_FILE}")
    
    # Check directories
    directories = [
        settings.TEMP_DIR,
        settings.DETECTION_OUTPUT_DIR,
        settings.NATIONAL_ID_OUTPUT_DIR,
        settings.ID_LAYOUT_OUTPUT_DIR
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directory {directory}: {str(e)}")
    
    # Check resource limits
    if settings.MAX_CONCURRENT_DOCUMENTS > 20:
        warnings.append("High MAX_CONCURRENT_DOCUMENTS may cause resource exhaustion")
    
    if settings.PIPELINE_TIMEOUT_SECONDS < 60:
        warnings.append("Low PIPELINE_TIMEOUT_SECONDS may cause premature timeouts")
    
    # Production checks
    if settings.ENVIRONMENT == "production":
        production_warnings = []
        
        if settings.DEBUG:
            production_warnings.append("DEBUG should be False in production")
        
        if settings.RELOAD:
            production_warnings.append("RELOAD should be False in production")
        
        if not settings.ENABLE_API_KEY_AUTH:
            production_warnings.append("Consider enabling API key authentication")
        
        if not settings.ENABLE_DATA_ENCRYPTION:
            production_warnings.append("Consider enabling data encryption")
        
        if settings.LOG_OCR_RESULTS:
            production_warnings.append("OCR results logging may expose sensitive data")
        
        warnings.extend(production_warnings)
    
    # Log warnings and errors
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")
    
    for error in errors:
        logger.error(f"Configuration error: {error}")
    
    if errors:
        raise ValueError(f"Configuration validation failed with {len(errors)} errors")
    
    if warnings:
        logger.info(f"Configuration validated with {len(warnings)} warnings")
    else:
        logger.info("Configuration validation passed successfully")

# ============= Environment-Specific Configurations =============

class DevelopmentSettings(Settings):
    """Development environment specific settings"""
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    RELOAD: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENABLE_PERFORMANCE_MONITORING: bool = True
    ENABLE_METRICS_COLLECTION: bool = True
    MODEL_PRELOAD_ON_STARTUP: bool = False  # Faster startup in dev

class StagingSettings(Settings):
    """Staging environment specific settings"""
    ENVIRONMENT: str = "staging"
    DEBUG: bool = False
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    ENABLE_API_KEY_AUTH: bool = True
    RATE_LIMIT_ENABLED: bool = True
    ENABLE_AUDIT_LOGGING: bool = True

class ProductionSettings(Settings):
    """Production environment specific settings"""
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    RELOAD: bool = False
    LOG_LEVEL: str = "WARNING"
    ENABLE_API_KEY_AUTH: bool = True
    ENABLE_DATA_ENCRYPTION: bool = True
    ENABLE_FILE_VIRUS_SCAN: bool = True
    RATE_LIMIT_ENABLED: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    LOG_OCR_RESULTS: bool = False
    ANONYMIZE_LOGS: bool = True
    WORKERS: int = 4
    MODEL_PRELOAD_ON_STARTUP: bool = True
    ENABLE_ALERTING: bool = True

# ============= Configuration Factory =============

def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("KYC_ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "staging":
        return StagingSettings()
    else:
        return DevelopmentSettings()

# Use environment-specific settings
settings = get_settings()

# Validate configuration on import
if __name__ != "__main__":
    try:
        validate_configuration()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        # In development, continue with warnings
        # In production, you might want to exit here