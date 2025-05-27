# Fixed app/routers/national_id.py - Corrected OCR parameter handling

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query # type: ignore
from typing import List, Optional, Dict, Any
import logging
import os
from pathlib import Path
from ..models.national_id import NationalIDResult, BatchNationalIDResult, IDElementDetection
from ..services.national_id_service import NationalIDService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = NationalIDService()
file_handler = FileHandler()

@router.post("/process-id/", response_model=NationalIDResult)
async def process_national_id(
    request: Request,
    file: Optional[UploadFile] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected elements"),
    ocr_prompt: Optional[str] = Query(None, description="Custom OCR prompt")
):
    """Process a National ID card and detect its elements with optional OCR."""
    try:
        # Initialize element_prompts to empty dict by default
        element_prompts = {}
        
        # Handle file upload
        if file:
            file_path = await file_handler.process_uploaded_file(
                file, 
                service.supported_images | service.supported_docs
            )
            # For file uploads, element_prompts remains empty dict unless passed in form data
            # Note: form-data doesn't easily support complex JSON, so we rely on query params
            
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
            
            # Extract element-specific prompts
            element_prompts = body.get('element_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Process document with OCR if enabled
        if enable_ocr:
            result = await service.process_file_with_ocr(
                file_path, 
                enable_ocr, 
                ocr_prompt, 
                element_prompts
            )
        else:
            result = await service.process_file(file_path)
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing National ID: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-ids/", response_model=BatchNationalIDResult)
async def process_national_ids(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected elements"),
    ocr_prompt: Optional[str] = Query(None, description="Default OCR prompt for all files")
):
    """Process multiple National ID cards with optional OCR."""
    try:
        file_paths = []
        # Initialize element_prompts to empty dict by default
        element_prompts = {}
        
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
            element_prompts = body.get('element_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Process all documents with OCR if enabled
        if enable_ocr:
            results = await service.process_batch_with_ocr(
                file_paths, 
                enable_ocr, 
                ocr_prompt, 
                element_prompts
            )
        else:
            results = await service.process_batch(file_paths)
        
        # Calculate summary
        total = len(results)
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = total - successful
        
        if enable_ocr:
            total_elements = sum(
                r.get("ocr_summary", {}).get("total_elements", 0) 
                for r in results
            )
            successful_ocr = sum(
                r.get("ocr_summary", {}).get("successful_ocr", 0) 
                for r in results
            )
            
            summary = {
                "total_processed": total,
                "successful": successful,
                "failed": failed,
                "total_elements_detected": total_elements,
                "ocr_enabled": True,
                "successful_ocr": successful_ocr
            }
        else:
            total_elements = sum(len(r.get("elements", [])) for r in results if isinstance(r, dict))
            summary = {
                "total_processed": total,
                "successful": successful,
                "failed": failed,
                "total_elements_detected": total_elements,
                "ocr_enabled": False
            }
        
        return BatchNationalIDResult(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error processing National IDs: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/element-types")
async def get_element_types() -> Dict[str, List[str]]:
    """Get list of detectable ID elements."""
    return {
        "id_elements": service.id_elements,
        "front_elements": service.front_elements,
        "back_elements": service.back_elements
    }

@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    """Get list of supported file formats."""
    return {
        "supported_images": list(service.supported_images),
        "supported_documents": list(service.supported_docs)
    }

@router.get("/health")
async def national_id_health() -> Dict[str, Any]:
    """Check health of National ID detection service."""
    return {
        "status": "healthy",
        "service": "national_id",
        "model": service.model_path,
        "supported_elements": service.id_elements
    }