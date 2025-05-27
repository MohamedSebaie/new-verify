from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from typing import List, Optional, Dict, Any
import logging
import asyncio
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from ..services.master_pipeline_service import MasterPipelineService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
pipeline_service = MasterPipelineService()
file_handler = FileHandler()

@router.post("/process-document/")
async def process_single_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    enable_validation: bool = Query(False, description="Enable cross-validation if multiple document types detected"),
    classification_only: bool = Query(False, description="Only classify document without full processing")
):
    """
    Master pipeline endpoint for processing single documents.
    Automatically classifies and routes through appropriate sub-pipeline.
    
    Supported document types:
    - KYC: Signature verification → Field detection → OCR
    - TC: Clarity assessment
    - NID: Layout detection → Element segmentation → OCR
    """
    try:
        # Handle file upload
        if file:
            file_path = await file_handler.process_uploaded_file(
                file, 
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_file' not in body:
                raise HTTPException(status_code=400, detail="No base64_file provided")
                
            file_path = await file_handler.process_base64_file(
                body['base64_file'],
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
            
            # Extract parameters from body
            if 'enable_validation' in body:
                enable_validation = body['enable_validation']
            if 'classification_only' in body:
                classification_only = body['classification_only']
        else:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Classification only mode
        if classification_only:
            logger.info("Processing in classification-only mode")
            classification_result = await pipeline_service.classification_service.process_file(file_path)
            return {
                "mode": "classification_only",
                "filename": file_path.name,
                "classification": classification_result,
                "message": f"Document classified as {classification_result.get('class', 'unknown')}"
            }
        
        # Full pipeline processing
        result = await pipeline_service.process_master_pipeline(file_path, enable_validation)
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-documents/")
async def process_multiple_documents(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    enable_validation: bool = Query(True, description="Enable cross-validation for 3-document sets (KYC+TC+NID)"),
    parallel_processing: bool = Query(True, description="Process documents in parallel for better performance")
):
    """
    Master pipeline endpoint for processing multiple documents.
    
    Special Validation Logic for 3-Document Sets:
    - 3 images OR 3-page PDF containing KYC, TC, NID
    - If all 3 present: Run all pipelines + validation
    - If TC missing: Still validate KYC with NID (TC optional)
    - If KYC or NID missing: Process available documents + warning (no validation)
    - Less than 3 documents: Process normally, no 3-document validation
    
    Features:
    - Automatic document classification and routing
    - Cross-validation when both KYC and NID are present
    - Comprehensive result aggregation with warnings
    """
    try:
        file_paths = []
        is_single_pdf = False
        
        # Handle file uploads
        if files:
            if len(files) == 1 and files[0].filename.lower().endswith('.pdf'):
                # Single PDF - check if it has 3 pages
                pdf_file = files[0]
                temp_pdf_path = await file_handler.process_uploaded_file(
                    pdf_file,
                    {'.pdf'}
                )
                
                # Check PDF page count and extract pages if it's 3 pages
                pdf_pages = await _extract_pdf_pages_if_three(temp_pdf_path)
                if pdf_pages:
                    file_paths = pdf_pages
                    is_single_pdf = True
                    logger.info(f"Processing 3-page PDF as 3-document set: {pdf_file.filename}")
                else:
                    # Regular PDF processing
                    file_paths = [temp_pdf_path]
            else:
                # Multiple files
                for file in files:
                    file_path = await file_handler.process_uploaded_file(
                        file,
                        {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
                    )
                    file_paths.append(file_path)
                    
                logger.info(f"Processing {len(files)} separate files")
                    
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_files' not in body or not body['base64_files']:
                raise HTTPException(status_code=400, detail="No base64_files provided")
            
            # Extract parameters from body
            if 'enable_validation' in body:
                enable_validation = body['enable_validation']
            if 'parallel_processing' in body:
                parallel_processing = body['parallel_processing']
            
            # Check if single PDF with 3 pages
            if len(body['base64_files']) == 1 and body['base64_files'][0]['filename'].lower().endswith('.pdf'):
                # Single PDF - check if it has 3 pages
                temp_pdf_path = await file_handler.process_base64_file(
                    body['base64_files'][0],
                    {'.pdf'}
                )
                
                pdf_pages = await _extract_pdf_pages_if_three(temp_pdf_path)
                if pdf_pages:
                    file_paths = pdf_pages
                    is_single_pdf = True
                    logger.info(f"Processing 3-page PDF as 3-document set: {body['base64_files'][0]['filename']}")
                else:
                    file_paths = [temp_pdf_path]
            else:
                # Multiple base64 files
                for b64file in body['base64_files']:
                    file_path = await file_handler.process_base64_file(
                        b64file,
                        {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
                    )
                    file_paths.append(file_path)
                    
                logger.info(f"Processing {len(body['base64_files'])} base64 files")
        else:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        # Process documents with 3-document validation logic
        result = await pipeline_service.process_multi_document_pipeline(
            file_paths, enable_validation
        )
        
        # Add metadata about processing type
        result["processing_metadata"] = {
            "total_files_received": len(file_paths),
            "is_three_document_set": len(file_paths) == 3,
            "is_single_pdf_split": is_single_pdf,
            "validation_enabled": enable_validation,
            "parallel_processing": parallel_processing
        }
        
        # Log validation warnings if any
        if "validation_warnings" in result and result["validation_warnings"]:
            for warning in result["validation_warnings"]:
                logger.warning(f"Validation warning: {warning}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing multiple documents: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


async def _extract_pdf_pages_if_three(pdf_path: Path) -> Optional[List[Path]]:
    """
    Extract individual pages from PDF if it has exactly 3 pages
    Returns list of page image paths or None if not 3 pages
    """
    try:
        import fitz  # PyMuPDF
        
        pdf_document = fitz.open(str(pdf_path))
        page_count = len(pdf_document)
        
        if page_count != 3:
            logger.info(f"PDF has {page_count} pages, not 3. Processing as single document.")
            pdf_document.close()
            return None
        
        # Extract 3 pages as separate images
        page_paths = []
        temp_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        temp_dir.mkdir(exist_ok=True)
        
        for page_num in range(3):
            page = pdf_document.load_page(page_num)
            
            # Convert page to image (higher DPI for better quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Save as image
            page_image_path = temp_dir / f"page_{page_num + 1}.png"
            pix.save(str(page_image_path))
            page_paths.append(page_image_path)
            
            logger.info(f"Extracted page {page_num + 1} from PDF to {page_image_path}")
        
        pdf_document.close()
        logger.info(f"Successfully extracted 3 pages from PDF for individual processing")
        return page_paths
        
    except Exception as e:
        logger.error(f"Error extracting PDF pages: {str(e)}")
        return None


@router.post("/process-three-document-set/")
async def process_three_document_set(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    enable_validation: bool = Query(True, description="Enable KYC-NID validation (TC is optional)")
):
    """
    Specialized endpoint for processing 3-document sets (KYC + TC + NID).
    
    Validation Logic:
    - Requires exactly 3 images OR 1 PDF with 3 pages
    - Documents should be KYC, TC, and NID
    - Validation between KYC and NID (TC is optional for validation)
    - If TC missing: Still validates KYC with NID + warning
    - If KYC or NID missing: Process available + warning (no validation)
    
    Use this endpoint when you specifically have a 3-document set for KYC processing.
    """
    try:
        file_paths = []
        
        # Handle file uploads
        if files:
            if len(files) == 1 and files[0].filename.lower().endswith('.pdf'):
                # Single 3-page PDF
                pdf_file = files[0]
                temp_pdf_path = await file_handler.process_uploaded_file(pdf_file, {'.pdf'})
                
                pdf_pages = await _extract_pdf_pages_if_three(temp_pdf_path)
                if not pdf_pages:
                    raise HTTPException(
                        status_code=400, 
                        detail="PDF must have exactly 3 pages for 3-document set processing"
                    )
                file_paths = pdf_pages
                
            elif len(files) == 3:
                # 3 separate images
                for file in files:
                    file_path = await file_handler.process_uploaded_file(
                        file, {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
                    )
                    file_paths.append(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide exactly 3 image files OR 1 PDF with 3 pages"
                )
                
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            
            if 'enable_validation' in body:
                enable_validation = body['enable_validation']
            
            if 'base64_files' in body:
                if len(body['base64_files']) == 1 and body['base64_files'][0]['filename'].lower().endswith('.pdf'):
                    # Single 3-page PDF
                    temp_pdf_path = await file_handler.process_base64_file(
                        body['base64_files'][0], {'.pdf'}
                    )
                    
                    pdf_pages = await _extract_pdf_pages_if_three(temp_pdf_path)
                    if not pdf_pages:
                        raise HTTPException(
                            status_code=400,
                            detail="PDF must have exactly 3 pages for 3-document set processing"
                        )
                    file_paths = pdf_pages
                    
                elif len(body['base64_files']) == 3:
                    # 3 separate base64 files
                    for b64file in body['base64_files']:
                        file_path = await file_handler.process_base64_file(
                            b64file, {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
                        )
                        file_paths.append(file_path)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Must provide exactly 3 base64 files OR 1 PDF with 3 pages"
                    )
            else:
                raise HTTPException(status_code=400, detail="No base64_files provided")
        else:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Process 3-document set
        result = await pipeline_service.process_three_document_set(
            file_paths, enable_validation
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing 3-document set: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))
async def process_kyc_nid_pair(
    request: Request,
    kyc_file: Optional[UploadFile] = File(None),
    nid_file: Optional[UploadFile] = File(None)
):
    """
    Specialized endpoint for processing KYC and NID document pairs.
    Automatically performs cross-validation between the two documents.
    
    This endpoint is optimized for the common use case of validating
    KYC applications against National ID documents.
    """
    try:
        file_paths = []
        
        # Handle file uploads
        if kyc_file and nid_file:
            kyc_path = await file_handler.process_uploaded_file(
                kyc_file,
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
            nid_path = await file_handler.process_uploaded_file(
                nid_file,
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
            file_paths = [kyc_path, nid_path]
            
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'kyc_file' not in body or 'nid_file' not in body:
                raise HTTPException(
                    status_code=400, 
                    detail="Both kyc_file and nid_file must be provided"
                )
            
            kyc_path = await file_handler.process_base64_file(
                body['kyc_file'],
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
            nid_path = await file_handler.process_base64_file(
                body['nid_file'],
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
            )
            file_paths = [kyc_path, nid_path]
        else:
            raise HTTPException(
                status_code=400, 
                detail="Both KYC and NID files must be provided"
            )
        
        # Process the document pair with validation enabled
        result = await pipeline_service.process_multi_document_pipeline(
            file_paths, enable_validation=True
        )
        
        # Add specialized response structure for KYC-NID pairs
        kyc_result = None
        nid_result = None
        
        for doc_key, doc_result in result.get("document_results", {}).items():
            doc_type = doc_result.get("processing_summary", {}).get("document_type")
            if doc_type == "KYC":
                kyc_result = doc_result
            elif doc_type == "NID":
                nid_result = doc_result
        
        # Enhanced response for KYC-NID pairs
        enhanced_result = {
            **result,
            "pair_processing": {
                "kyc_processed": kyc_result is not None,
                "nid_processed": nid_result is not None,
                "both_successful": (
                    kyc_result and nid_result and
                    kyc_result.get("processing_summary", {}).get("pipeline_success", False) and
                    nid_result.get("processing_summary", {}).get("pipeline_success", False)
                ),
                "validation_available": result.get("cross_validation", {}).get("validation_performed", False)
            }
        }
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Error processing KYC-NID pair: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-extracted-data/")
async def validate_extracted_data(request: Request):
    """
    Standalone validation endpoint for comparing extracted data from KYC and NID documents.
    Useful when you have already processed documents separately and want to validate them.
    """
    try:
        body = await request.json()
        
        if 'kyc_data' not in body or 'nid_data' not in body:
            raise HTTPException(
                status_code=400,
                detail="Both kyc_data and nid_data must be provided"
            )
        
        kyc_data = body['kyc_data']
        nid_data = body['nid_data']
        
        # Perform validation
        validation_result = pipeline_service.validation_service.validate_kyc_nid_data(
            kyc_data, nid_data
        )
        
        return {
            "validation_type": "standalone",
            "timestamp": pipeline_service.validation_service.time.time(),
            "validation_results": validation_result
        }
        
    except Exception as e:
        logger.error(f"Error in standalone validation: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-status/")
async def get_pipeline_status():
    """
    Get the current status of all pipeline services and their capabilities.
    """
    try:
        status = {
            "pipeline_services": {
                "classification": {
                    "available": True,
                    "supported_classes": ["KYC", "NID", "TC"],
                    "model_loaded": hasattr(pipeline_service.classification_service, 'model')
                },
                "verification": {
                    "available": True,
                    "required_elements": ["Store_Stamp", "Agent_Signature", "CS_Signature"],
                    "model_loaded": hasattr(pipeline_service.verification_service, 'model')
                },
                "detection": {
                    "available": True,
                    "model_loaded": hasattr(pipeline_service.detection_service, 'model'),
                    "ocr_integration": pipeline_service.detection_service.ocr_available
                },
                "clarity": {
                    "available": True,
                    "model_loaded": hasattr(pipeline_service.clarity_service, 'model')
                },
                "id_layout": {
                    "available": True,
                    "supported_regions": pipeline_service.id_layout_service.region_types,
                    "model_loaded": hasattr(pipeline_service.id_layout_service, 'model'),
                    "ocr_integration": pipeline_service.id_layout_service.ocr_available
                },
                "national_id": {
                    "available": True,
                    "supported_elements": pipeline_service.national_id_service.id_elements,
                    "model_loaded": hasattr(pipeline_service.national_id_service, 'model'),
                    "ocr_integration": pipeline_service.national_id_service.ocr_available
                },
                "ocr": {
                    "available": pipeline_service.ocr_available,
                    "provider": pipeline_service.ocr_service.provider if pipeline_service.ocr_available else None,
                    "text_cleaning": True
                },
                "validation": {
                    "available": True,
                    "supported_validations": ["name_match", "id_number_match", "address_match", "date_of_birth_match"]
                }
            },
            "supported_formats": {
                "images": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
                "documents": [".pdf"]
            },
            "pipeline_capabilities": {
                "single_document_processing": True,
                "multi_document_processing": True,
                "cross_validation": True,
                "parallel_processing": True,
                "classification_only_mode": True
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-document-types/")
async def get_supported_document_types():
    """
    Get detailed information about supported document types and their processing pipelines.
    """
    return {
        "document_types": {
            "KYC": {
                "full_name": "Know Your Customer",
                "pipeline_stages": [
                    {
                        "stage": "verification",
                        "description": "Check for required signatures and stamps",
                        "required_elements": ["Agent_Signature", "CS_Signature", "Store_Stamp"]
                    },
                    {
                        "stage": "detection",
                        "description": "Detect form fields and layout regions"
                    },
                    {
                        "stage": "ocr",
                        "description": "Extract text from detected fields"
                    }
                ],
                "output_fields": [
                    "signature_verification",
                    "extracted_fields"
                ]
            },
            "NID": {
                "full_name": "National ID",
                "pipeline_stages": [
                    {
                        "stage": "id_layout",
                        "description": "Detect ID card components (front, back, signature, stamp)"
                    },
                    {
                        "stage": "nid_detection",
                        "description": "Segment individual ID fields and elements"
                    },
                    {
                        "stage": "ocr",
                        "description": "Extract text from segmented fields"
                    }
                ],
                "output_fields": [
                    "elements_detected",
                    "extracted_data"
                ]
            },
            "TC": {
                "full_name": "Trade Certificate",
                "pipeline_stages": [
                    {
                        "stage": "clarity",
                        "description": "Assess document image quality and clarity"
                    }
                ],
                "output_fields": [
                    "is_clear",
                    "confidence",
                    "message"
                ]
            }
        },
        "validation_capabilities": {
            "kyc_nid_validation": {
                "description": "Cross-validate KYC form data against National ID information",
                "validated_fields": [
                    "customer_name vs name",
                    "national_id vs id_number", 
                    "customer_address vs address",
                    "date_of_birth"
                ],
                "similarity_threshold": 0.85
            }
        }
    }


@router.get("/health")
async def pipeline_health():
    """
    Health check endpoint for the master pipeline service.
    """
    try:
        health_status = {
            "status": "healthy",
            "service": "master_pipeline",
            "timestamp": pipeline_service.time.time(),
            "components": {}
        }
        
        # Check each service component
        services_to_check = [
            ("classification", pipeline_service.classification_service),
            ("verification", pipeline_service.verification_service),
            ("detection", pipeline_service.detection_service),
            ("clarity", pipeline_service.clarity_service),
            ("id_layout", pipeline_service.id_layout_service),
            ("national_id", pipeline_service.national_id_service)
        ]
        
        for service_name, service in services_to_check:
            try:
                # Basic health check - verify service has required attributes
                has_model = hasattr(service, 'model')
                health_status["components"][service_name] = {
                    "status": "healthy" if has_model else "warning",
                    "model_loaded": has_model
                }
            except Exception as e:
                health_status["components"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check OCR service
        health_status["components"]["ocr"] = {
            "status": "healthy" if pipeline_service.ocr_available else "unavailable",
            "available": pipeline_service.ocr_available
        }
        
        # Overall health determination
        unhealthy_services = [
            name for name, status in health_status["components"].items()
            if status["status"] == "unhealthy"
        ]
        
        if unhealthy_services:
            health_status["status"] = "degraded"
            health_status["unhealthy_services"] = unhealthy_services
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "master_pipeline",
            "error": str(e),
            "timestamp": time.time()
        }