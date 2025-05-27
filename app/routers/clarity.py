from fastapi import APIRouter, File, UploadFile, HTTPException, Request # type: ignore
from typing import List, Optional, Dict, Any
import logging
from ..models import ClarityResult, BatchClarityResult, PDFClarityResult
from ..services.clarity_service import ClarityService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = ClarityService()
file_handler = FileHandler()

@router.post("/check-clarity/", response_model=Dict[str, Any])
async def check_document_clarity(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """Check clarity of a document."""
    try:
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
        else:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Process document
        result = await service.process_file(file_path)
        return result
        
    except Exception as e:
        logger.error(f"Error checking clarity: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-clarity-batch/", response_model=BatchClarityResult)
async def check_documents_clarity(
    request: Request,
    files: Optional[List[UploadFile]] = File(None)
):
    """Check clarity of multiple documents."""
    try:
        file_paths = []
        
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
        else:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Process all documents
        results = await service.process_batch(file_paths)
        
        # Calculate summary
        total = len(results)
        clear = sum(1 for r in results if 'is_clear' in r and r['is_clear'])
        unclear = sum(1 for r in results if 'is_clear' in r and not r['is_clear'])
        errors = sum(1 for r in results if 'error' in r)
        
        return BatchClarityResult(
            results=results,
            summary={
                "total_processed": total,
                "clear_documents": clear,
                "unclear_documents": unclear,
                "errors": errors
            }
        )
        
    except Exception as e:
        logger.error(f"Error checking clarity for batch: {str(e)}")
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
async def clarity_health() -> Dict[str, str]:
    """Check health of clarity service."""
    return {
        "status": "healthy",
        "service": "clarity"
    }