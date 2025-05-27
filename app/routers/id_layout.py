from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query # type: ignore
from typing import List, Optional, Dict, Any
import logging
import time
from pathlib import Path
from ..models.id_layout import LayoutDetectionResult, BatchLayoutResult, LayoutRegion
from ..services.id_layout_service import IDLayoutService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = IDLayoutService()
file_handler = FileHandler()

@router.post("/detect-layout/", response_model=LayoutDetectionResult)
async def detect_id_layout(
    request: Request,
    file: Optional[UploadFile] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected regions"),
    ocr_prompt: Optional[str] = Query(None, description="Custom OCR prompt"),
):
    """Detect the layout of a National ID card with optional OCR."""
    try:
        # Initialize region_prompts to empty dict by default
        region_prompts = {}
        
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
            
            # Extract region-specific prompts
            region_prompts = body.get('region_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Process document with OCR if enabled
        if enable_ocr:
            result = await service.process_file_with_ocr(
                file_path, 
                enable_ocr, 
                ocr_prompt, 
                region_prompts,
                clean_text=True  # Always clean text for better results
            )
        else:
            result = await service.process_file(file_path)
            
        return result
        
    except Exception as e:
        logger.error(f"Error detecting ID layout: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-layout-simple/")
async def detect_id_layout_simple(
    request: Request,
    file: Optional[UploadFile] = File(None),
    clean_text: bool = Query(True, description="Clean and simplify extracted text"),
    ocr_prompt: Optional[str] = Query(None, description="Custom OCR prompt")
):
    """Detect ID layout and return only clean OCR results in simplified format."""
    try:
        # Initialize region_prompts to empty dict by default
        region_prompts = {}
        
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
            region_prompts = body.get('region_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Always process with OCR enabled and simplified response
        result = await service.process_file_with_ocr(
            file_path, 
            enable_ocr=True, 
            ocr_prompt=ocr_prompt, 
            region_prompts=region_prompts,
            clean_text=clean_text,
            simplified_response=True
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in simplified OCR processing: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-layouts/", response_model=BatchLayoutResult)
async def detect_id_layouts(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    enable_ocr: bool = Query(False, description="Enable OCR on detected regions"),
    ocr_prompt: Optional[str] = Query(None, description="Default OCR prompt for all files")
):
    """Detect layouts of multiple National ID cards with optional OCR."""
    try:
        file_paths = []
        # Initialize region_prompts to empty dict by default
        region_prompts = {}
        
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
            region_prompts = body.get('region_prompts', {})
        else:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Process all documents with OCR if enabled
        if enable_ocr:
            results = await service.process_batch_with_ocr(
                file_paths, 
                enable_ocr, 
                ocr_prompt, 
                region_prompts,
                clean_text=True
            )
        else:
            results = await service.process_batch(file_paths)
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.get("status") == "success")
        
        if enable_ocr:
            total_ocr = sum(
                r.get("ocr_summary", {}).get("total_regions", 0) 
                for r in results
            )
            successful_ocr = sum(
                r.get("ocr_summary", {}).get("successful_ocr", 0) 
                for r in results
            )
            
            summary = {
                "total_processed": total,
                "successful": successful,
                "failed": total - successful,
                "ocr_enabled": True,
                "total_ocr_regions": total_ocr,
                "successful_ocr": successful_ocr
            }
        else:
            summary = {
                "total_processed": total,
                "successful": successful,
                "failed": total - successful,
                "ocr_enabled": False
            }
        
        return BatchLayoutResult(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error detecting ID layouts: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/layout-types")
async def get_layout_types() -> Dict[str, List[str]]:
    """Get list of supported ID layout types."""
    return {
        "layout_types": service.layout_types,
        "region_types": service.region_types
    }

@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    """Get list of supported file formats."""
    return {
        "supported_images": list(service.supported_images),
        "supported_documents": list(service.supported_docs)
    }

@router.get("/health")
async def layout_detection_health() -> Dict[str, Any]:
    """Check health of ID layout detection service."""
    return {
        "status": "healthy",
        "service": "id_layout",
        "model": service.model_path,
        "layout_types": service.layout_types,
        "ocr_available": service.ocr_available
    }