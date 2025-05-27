from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query # type: ignore
from typing import List, Optional, Dict
import logging
import numpy as np
import base64
import io
from pathlib import Path
from PIL import Image
from datetime import datetime
from ..models.detection import DetectionResult, BatchDetectionResult, Base64File
from ..services.detection_service import DetectionService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = DetectionService()
file_handler = FileHandler()

@router.post("/detect-document/", response_model=DetectionResult)
async def detect_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected objects"),
    ocr_prompt: Optional[str] = Query(None, description="Custom OCR prompt")
):
    """Detect objects in a document with optional OCR."""
    try:
        # Initialize class_prompts to empty dict by default
        class_prompts = {}
        
        # Handle file upload
        if file:
            file_path = await file_handler.process_uploaded_file(
                file, 
                service.supported_images | service.supported_docs
            )
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_file' not in body:
                raise HTTPException(status_code=400, detail="No base64_file provided")
                
            file_path = await file_handler.process_base64_file(
                body['base64_file'],
                service.supported_images | service.supported_docs
            )
            
            # Extract OCR parameters from body if not provided in query
            if 'enable_ocr' in body:
                enable_ocr = body['enable_ocr']
            if 'ocr_prompt' in body:
                ocr_prompt = body['ocr_prompt']
            
            # Extract class-specific prompts
            class_prompts = body.get('class_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Process document with OCR if enabled
        if enable_ocr:
            result = await service.process_file_with_ocr(
                file_path, 
                enable_ocr, 
                ocr_prompt, 
                class_prompts,
                clean_text=True  # Always clean text for better results
            )
        else:
            result = await service.process_file(file_path)
            
        return result
        
    except Exception as e:
        logger.error(f"Error detecting objects in document: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-document-simple/")
async def detect_document_simple(
    request: Request,
    file: Optional[UploadFile] = File(None),
    clean_text: bool = Query(True, description="Clean and simplify extracted text"),
    ocr_prompt: Optional[str] = Query(None, description="Custom OCR prompt")
):
    """Detect objects and return only clean OCR results in simplified format."""
    try:
        # Initialize class_prompts to empty dict by default
        class_prompts = {}
        
        # Handle file upload
        if file:
            file_path = await file_handler.process_uploaded_file(
                file, 
                service.supported_images | service.supported_docs
            )
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_file' not in body:
                raise HTTPException(status_code=400, detail="No base64_file provided")
                
            file_path = await file_handler.process_base64_file(
                body['base64_file'],
                service.supported_images | service.supported_docs
            )
            
            # Extract parameters
            if 'clean_text' in body:
                clean_text = body['clean_text']
            if 'ocr_prompt' in body:
                ocr_prompt = body['ocr_prompt']
            class_prompts = body.get('class_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Always process with OCR enabled and simplified response
        result = await service.process_file_with_ocr(
            file_path, 
            enable_ocr=True, 
            ocr_prompt=ocr_prompt, 
            class_prompts=class_prompts,
            clean_text=clean_text,
            simplified_response=True
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in simplified detection OCR: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-documents/", response_model=BatchDetectionResult)
async def detect_documents(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected objects"),
    ocr_prompt: Optional[str] = Query(None, description="Default OCR prompt for all files")
):
    """Detect objects in multiple documents with optional OCR."""
    try:
        file_paths = []
        # Initialize class_prompts to empty dict by default
        class_prompts = {}
        
        # Handle file uploads
        if files:
            for file in files:
                file_path = await file_handler.process_uploaded_file(
                    file,
                    service.supported_images | service.supported_docs
                )
                file_paths.append(file_path)
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_files' not in body or not body['base64_files']:
                raise HTTPException(status_code=400, detail="No base64_files provided")
                
            for b64file in body['base64_files']:
                file_path = await file_handler.process_base64_file(
                    b64file,
                    service.supported_images | service.supported_docs
                )
                file_paths.append(file_path)
            
            # Extract OCR parameters from body
            if 'enable_ocr' in body:
                enable_ocr = body['enable_ocr']
            if 'ocr_prompt' in body:
                ocr_prompt = body['ocr_prompt']
            class_prompts = body.get('class_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Process all documents with OCR if enabled
        if enable_ocr:
            results = await service.process_batch_with_ocr(
                file_paths, 
                enable_ocr, 
                ocr_prompt, 
                class_prompts,
                clean_text=True
            )
        else:
            results = await service.process_batch(file_paths)
        
        # Calculate statistics
        total_files = len(results)
        total_detections = 0
        successful_ocr = 0
        
        for result in results:
            if isinstance(result, dict):
                # Count detections
                if result.get("file_type") == "pdf":
                    for page_result in result.get("results", []):
                        total_detections += page_result.get("num_detections", 0)
                else:
                    for page_result in result.get("results", []):
                        total_detections += page_result.get("num_detections", 0)
                
                # Count successful OCR
                if enable_ocr and result.get("ocr_summary"):
                    successful_ocr += result.get("ocr_summary", {}).get("successful_ocr", 0)
        
        summary = {
            "total_files": total_files,
            "total_detections": total_detections,
            "ocr_enabled": enable_ocr
        }
        
        if enable_ocr:
            summary["successful_ocr"] = successful_ocr
        
        return BatchDetectionResult(
            results=results,
            total_files=total_files,
            total_detections=total_detections
        )
        
    except Exception as e:
        logger.error(f"Error detecting objects in documents: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    """Get list of supported file formats."""
    return {
        "supported_images": list(service.supported_images),
        "supported_documents": list(service.supported_docs)
    }

@router.get("/health")
async def detection_health() -> Dict[str, str]:
    """Check health of detection service."""
    return {
        "status": "healthy",
        "service": "detection",
        "model": service.model_path,
        "ocr_available": service.ocr_available
    }