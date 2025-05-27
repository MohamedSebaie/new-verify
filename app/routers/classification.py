from fastapi import APIRouter, File, UploadFile, HTTPException, Request # type: ignore
from typing import List, Optional, Dict, Any
import logging
from ..models import ClassificationResult, BatchClassificationResult, PDFClassificationResult
from ..services.classification_service import ClassificationService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
service = ClassificationService()
file_handler = FileHandler()

@router.post("/classify/", response_model=Dict[str, Any])
async def classify_document(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """Classify a document as KYC, NID, or TC."""
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
        logger.error(f"Error classifying document: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-batch/", response_model=BatchClassificationResult)
async def classify_documents_batch(
    request: Request,
    files: Optional[List[UploadFile]] = File(None)
):
    """Classify multiple documents."""
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
        
        # Calculate class distribution
        total = len(results)
        class_counts = {cls: 0 for cls in service.id2label.values()}
        errors = 0
        
        for r in results:
            if 'error' in r:
                errors += 1
            elif 'class' in r:
                cls = r['class']
                class_counts[cls] += 1
            elif 'overall_result' in r and 'class' in r['overall_result']:
                cls = r['overall_result']['class']
                class_counts[cls] += 1
                
        return BatchClassificationResult(
            results=results,
            summary={
                "total_processed": total,
                "class_distribution": class_counts,
                "errors": errors
            }
        )
        
    except Exception as e:
        logger.error(f"Error classifying batch: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-classes")
async def get_available_classes() -> Dict[str, Any]:
    """Get list of available document classes."""
    return {
        "classes": list(service.id2label.values()),
        "class_mapping": service.id2label
    }

@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    """Get list of supported file formats."""
    return {
        "supported_images": list(service.supported_images),
        "supported_documents": list(service.supported_docs)
    }

@router.get("/health")
async def classification_health() -> Dict[str, Any]:
    """Check health of classification service."""
    return {
        "status": "healthy",
        "service": "classification",
        "available_classes": list(service.id2label.values())
    }