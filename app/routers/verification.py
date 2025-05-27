from fastapi import APIRouter, File, UploadFile, HTTPException, Request # type: ignore
from typing import List, Optional, Dict, Any
import logging
from ..models import VerificationResult, BatchVerificationResult, Base64File
from ..services.verification_service import VerificationService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = VerificationService()
file_handler = FileHandler()

@router.post("/process-document/", response_model=VerificationResult)
async def process_document(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """Process a document for verification."""
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
        logger.error(f"Error processing document: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-documents/", response_model=BatchVerificationResult)
async def process_documents(
    request: Request,
    files: Optional[List[UploadFile]] = File(None)
):
    """Process multiple documents for verification."""
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
        successful = sum(1 for r in results if r["status"] == "success")
        rejected = sum(1 for r in results if r["status"] == "rejected")
        errors = sum(1 for r in results if r["status"] == "error")
        
        return BatchVerificationResult(
            total_files=total,
            successful_files=successful,
            rejected_files=rejected,
            failed_files=errors,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
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
async def verification_health() -> Dict[str, str]:
    """Check health of verification service."""
    return {
        "status": "healthy",
        "service": "verification"
    }