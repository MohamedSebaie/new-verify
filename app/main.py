# Updated app/main.py - Integration with Master Pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .core.config import settings
import logging
import os
from pathlib import Path
import tempfile
import time

# Import existing routers
from .routers.verification import router as verification_router
from .routers.clarity import router as clarity_router
from .routers.classification import router as classification_router
from .routers import verification, classification, clarity, detection, national_id, id_layout, ocr

# Import NEW master pipeline router
from .routers.master_pipeline import router as master_pipeline_router

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{settings.PROJECT_NAME} - Master Pipeline",
    description="""
    Comprehensive KYC Processing API with Master Pipeline Integration
    
    ## Features
    - **Master Pipeline**: Intelligent document routing and processing
    - **3-Document Processing**: Specialized KYC+TC+NID validation
    - **Multi-Document Processing**: Handle any combination of documents
    - **Cross-Validation**: Automatic validation between KYC and NID data
    - **OCR Integration**: Advanced text extraction with Arabic support
    - **Parallel Processing**: Optimized batch processing capabilities
    
    ## Document Types Supported
    - **KYC**: Know Your Customer forms with signature verification
    - **NID**: National ID cards with field extraction
    - **TC**: Trade Certificates with clarity assessment
    
    ## Main Endpoints
    - `/api/v1/pipeline/process-document/` - Single document processing
    - `/api/v1/pipeline/process-documents/` - Multi-document processing (supports 3-doc validation)
    - `/api/v1/pipeline/process-three-document-set/` - Specialized 3-document KYC processing
    - `/api/v1/pipeline/process-kyc-nid-pair/` - KYC-NID pair validation
    
    ## 3-Document Validation Rules
    - **3 Images OR 3-page PDF** → Classify as KYC, TC, NID
    - **All 3 present** → Process all + validate KYC with NID
    - **TC missing only** → Process KYC+NID + validate (TC optional)
    - **KYC or NID missing** → Process available + warning (no validation)
    """,
    version=f"{settings.VERSION} - Master Pipeline",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# ============= MASTER PIPELINE ROUTES (NEW) =============
app.include_router(
    master_pipeline_router,
    prefix=f"{settings.API_V1_STR}/pipeline",
    tags=["🚀 Master Pipeline"]
)

# ============= EXISTING INDIVIDUAL SERVICE ROUTES =============
app.include_router(
    verification_router,
    prefix=f"{settings.API_V1_STR}/verification",
    tags=["Verification"]
)

app.include_router(
    clarity_router,
    prefix=f"{settings.API_V1_STR}/clarity",
    tags=["Clarity"]
)

app.include_router(
    classification_router,
    prefix=f"{settings.API_V1_STR}/classification",
    tags=["Classification"]
)

app.include_router(
    detection.router,
    prefix=f"{settings.API_V1_STR}/detection",
    tags=["Detection"]
)

app.include_router(
    national_id.router,
    prefix=f"{settings.API_V1_STR}/national-id",
    tags=["National ID"]
)

app.include_router(
    id_layout.router,
    prefix=f"{settings.API_V1_STR}/id-layout",
    tags=["ID Layout"]
)

app.include_router(
    ocr.router,
    prefix=f"{settings.API_V1_STR}/ocr",
    tags=["OCR"]
)

# ============= EXCEPTION HANDLERS =============
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error",
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
            "timestamp": time.time()
        }
    )

# ============= ROOT ENDPOINTS =============
@app.get("/")
async def root():
    return {
        "name": f"{settings.PROJECT_NAME} - Master Pipeline",
        "version": settings.VERSION,
        "description": "Comprehensive KYC Processing API with intelligent document routing",
        "features": [
            "Master Pipeline Processing",
            "Multi-Document Validation", 
            "Cross-Document Validation",
            "OCR Integration",
            "Parallel Processing"
        ],
        "main_endpoints": {
            "master_pipeline": f"{settings.API_V1_STR}/pipeline/",
            "docs": "/docs",
            "health": "/health",
            "status": f"{settings.API_V1_STR}/pipeline/pipeline-status/"
        },
        "supported_documents": ["KYC", "NID", "TC"],
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for all services"""
    health_status = {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check master pipeline service
    try:
        from .services.master_pipeline_service import MasterPipelineService
        pipeline_service = MasterPipelineService()
        health_status["services"]["master_pipeline"] = "active"
    except Exception as e:
        health_status["services"]["master_pipeline"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check individual services
    service_checks = {
        "verification": "active",
        "clarity": "active", 
        "classification": "active",
        "detection": "active",
        "national_id": "active",
        "id_layout": "active"
    }
    
    # Check OCR service
    try:
        from .services.ocr_service import OCRService
        ocr_service = OCRService()
        health_status["services"]["ocr"] = "active" if ocr_service.ocr_available else "unavailable"
    except Exception:
        health_status["services"]["ocr"] = "unavailable"
    
    health_status["services"].update(service_checks)
    
    # Overall health determination
    if any("error" in status for status in health_status["services"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/api-info")
async def api_info():
    """Detailed API information and capabilities"""
    return {
        "api_name": f"{settings.PROJECT_NAME} - Master Pipeline",
        "version": settings.VERSION,
        "pipeline_architecture": {
            "master_pipeline": {
                "description": "Intelligent document routing based on classification",
                "flow": "Classification → Routing → Processing → Validation",
                "capabilities": [
                    "Single document processing",
                    "Multi-document processing", 
                    "Cross-validation",
                    "Parallel processing"
                ]
            }
        },
        "document_processing_flows": {
            "KYC": [
                "1. Classification",
                "2. Signature Verification", 
                "3. Field Detection",
                "4. OCR Text Extraction"
            ],
            "NID": [
                "1. Classification",
                "2. ID Layout Detection",
                "3. Element Segmentation",
                "4. OCR Text Extraction"
            ],
            "TC": [
                "1. Classification",
                "2. Clarity Assessment"
            ]
        },
        "validation_capabilities": {
            "kyc_nid_validation": {
                "validated_fields": [
                    "Name matching",
                    "ID number matching",
                    "Address matching", 
                    "Date of birth matching"
                ],
                "similarity_threshold": 0.85,
                "validation_algorithm": "Text normalization + similarity scoring"
            }
        },
        "api_endpoints": {
            "master_pipeline": {
                "base_url": f"{settings.API_V1_STR}/pipeline",
                "main_endpoints": [
                    "POST /process-document/ - Single document processing",
                    "POST /process-documents/ - Multi-document processing",
                    "POST /process-kyc-nid-pair/ - KYC-NID pair validation",
                    "POST /validate-extracted-data/ - Standalone validation",
                    "GET /pipeline-status/ - Service status",
                    "GET /supported-document-types/ - Document type info"
                ]
            },
            "individual_services": {
                "verification": f"{settings.API_V1_STR}/verification",
                "detection": f"{settings.API_V1_STR}/detection", 
                "national_id": f"{settings.API_V1_STR}/national-id",
                "id_layout": f"{settings.API_V1_STR}/id-layout",
                "classification": f"{settings.API_V1_STR}/classification",
                "clarity": f"{settings.API_V1_STR}/clarity",
                "ocr": f"{settings.API_V1_STR}/ocr"
            }
        },
        "technical_specifications": {
            "supported_formats": {
                "images": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
                "documents": [".pdf"]
            },
            "input_methods": ["File upload", "Base64 encoding"],
            "ocr_providers": ["OCI", "VLLM"],
            "ai_models": ["YOLO", "ViT", "Custom classification"],
            "processing_modes": ["Synchronous", "Batch", "Parallel"]
        }
    }

# ============= STARTUP EVENT =============
@app.on_event("startup")
async def startup_event():
    logger.info("Starting KYC Processing API with Master Pipeline")
    
    # Initialize directories
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    temp_dir = Path(tempfile.gettempdir()) / "kyc_temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Initialize output directories
    output_dirs = [
        settings.DETECTION_OUTPUT_DIR,
        settings.NATIONAL_ID_OUTPUT_DIR, 
        settings.ID_LAYOUT_OUTPUT_DIR
    ]
    
    for output_dir in output_dirs:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    logger.info("All directories initialized")
    
    # Test master pipeline initialization
    try:
        from .services.master_pipeline_service import MasterPipelineService
        pipeline_service = MasterPipelineService()
        logger.info("Master pipeline service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize master pipeline service: {str(e)}")
    
    logger.info("🚀 KYC Processing API with Master Pipeline is ready!")

# ============= SHUTDOWN EVENT =============
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down KYC Processing API")
    
    # Cleanup temporary files
    try:
        temp_dir = Path(tempfile.gettempdir()) / "kyc_temp"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Temporary files cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary files: {e}")
    
    logger.info("KYC Processing API shutdown complete")

# ============= ADDITIONAL UTILITY ENDPOINTS =============
@app.get("/pipeline-demo")
async def pipeline_demo():
    """Demo endpoint showing pipeline usage examples"""
    return {
        "pipeline_demo": {
            "title": "KYC Processing Pipeline Demo",
            "description": "Example usage of the master pipeline API",
            "examples": {
                "single_document_processing": {
                    "endpoint": "POST /api/v1/pipeline/process-document/",
                    "description": "Process a single document (KYC, NID, or TC)",
                    "example_request": {
                        "method": "POST",
                        "headers": {"Content-Type": "multipart/form-data"},
                        "body": "file=@document.jpg&enable_validation=false"
                    },
                    "example_response": {
                        "filename": "document.jpg",
                        "classification": {"class": "KYC", "confidence": 0.95},
                        "pipeline_results": {
                            "KYC": {
                                "final_result": {
                                    "signature_verification": {
                                        "agent_signature": True,
                                        "customer_signature": True,
                                        "store_stamp": True
                                    },
                                    "extracted_fields": {
                                        "customer_name": "John Doe",
                                        "national_id": "123456789"
                                    }
                                }
                            }
                        }
                    }
                },
                "kyc_nid_pair_processing": {
                    "endpoint": "POST /api/v1/pipeline/process-kyc-nid-pair/",
                    "description": "Process KYC and NID documents together with validation",
                    "example_request": {
                        "method": "POST", 
                        "headers": {"Content-Type": "application/json"},
                        "body": {
                            "kyc_file": {
                                "filename": "kyc.jpg",
                                "content": "base64_encoded_content",
                                "file_type": "image/jpeg"
                            },
                            "nid_file": {
                                "filename": "nid.jpg", 
                                "content": "base64_encoded_content",
                                "file_type": "image/jpeg"
                            }
                        }
                    },
                    "example_response": {
                        "cross_validation": {
                            "validation_performed": True,
                            "overall_validation": True,
                            "field_validations": {
                                "Name": {
                                    "kyc_value": "John Doe",
                                    "nid_value": "John Doe", 
                                    "is_match": True,
                                    "similarity_score": 1.0
                                }
                            }
                        }
                    }
                }
            },
            "integration_tips": [
                "Use classification-only mode for quick document type detection",
                "Enable parallel processing for multiple documents",
                "Set appropriate validation thresholds for your use case",
                "Monitor processing times and adjust timeouts accordingly"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=4007,
        reload=True,
        log_level="info"
    )