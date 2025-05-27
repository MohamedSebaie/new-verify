from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query # type: ignore
from typing import List, Optional, Dict, Any
import logging
import time
from pathlib import Path
from ..models.ocr import (
    OCRRequest, SingleImageOCRResult, BatchOCRResult, 
    OCRConfig, RegionSpecificPrompts
)
from ..services.ocr_service import OCRService
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)
router = APIRouter()
file_handler = FileHandler()

# Initialize OCR services for both providers
ocr_services = {}
try:
    # Try to initialize OCI
    ocr_services["OCI"] = OCRService(provider="OCI")
    logger.info("OCI OCR service initialized")
except Exception as e:
    logger.warning(f"OCI OCR service not available: {str(e)}")

try:
    # Try to initialize VLLM
    ocr_services["VLLM"] = OCRService(provider="VLLM")
    logger.info("VLLM OCR service initialized")
except Exception as e:
    logger.warning(f"VLLM OCR service not available: {str(e)}")

# Check if any OCR service is available
ocr_available = len(ocr_services) > 0

def get_ocr_service(provider: Optional[str] = None) -> OCRService:
    """Get OCR service by provider or return default available service"""
    if not ocr_available:
        raise HTTPException(
            status_code=503, 
            detail="No OCR services are available. Please check OCI or VLLM configuration."
        )
    
    if provider and provider in ocr_services:
        if not ocr_services[provider].ocr_available:
            raise HTTPException(
                status_code=503,
                detail=f"{provider} OCR service is not available"
            )
        return ocr_services[provider]
    
    # Return first available service
    for service_name, service in ocr_services.items():
        if service.ocr_available:
            return service
    
    raise HTTPException(
        status_code=503,
        detail="No OCR services are currently available"
    )

@router.post("/process-image/", response_model=SingleImageOCRResult)
async def process_image_ocr(
    request: Request,
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Query(None, description="Custom OCR prompt"),
    provider: Optional[str] = Query(None, description="OCR provider: OCI or VLLM"),
    clean_text: bool = Query(True, description="Clean and simplify extracted text")
):
    """Perform OCR on a single image with specified provider."""
    ocr_service = get_ocr_service(provider)
    start_time = time.time()
    
    try:
        # Handle file upload
        if file:
            file_path = await file_handler.process_uploaded_file(
                file, 
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
            )
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_file' not in body:
                raise HTTPException(status_code=400, detail="No base64_file provided")
                
            file_path = await file_handler.process_base64_file(
                body['base64_file'],
                {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
            )
            
            # Extract parameters from request body
            if prompt is None and 'prompt' in body:
                prompt = body['prompt']
            if 'provider' in body:
                provider = body['provider']
                ocr_service = get_ocr_service(provider)  # Re-get service with specified provider
            if 'clean_text' in body:
                clean_text = body['clean_text']
        else:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Perform OCR
        ocr_result = await ocr_service.perform_ocr(file_path, prompt, clean_text)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SingleImageOCRResult(
            filename=file_path.name,
            ocr_result=ocr_result,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing OCR: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-images/", response_model=BatchOCRResult)
async def process_images_ocr(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    default_prompt: Optional[str] = Query(None, description="Default prompt for all images"),
    provider: Optional[str] = Query(None, description="OCR provider: OCI or VLLM"),
    clean_text: bool = Query(True, description="Clean and simplify extracted text")
):
    """Perform OCR on multiple images with specified provider."""
    ocr_service = get_ocr_service(provider)
    start_time = time.time()
    
    try:
        file_paths = []
        prompts = []
        
        # Handle file uploads
        if files:
            for file in files:
                file_path = await file_handler.process_uploaded_file(
                    file,
                    {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
                )
                file_paths.append(file_path)
                prompts.append(default_prompt)
        # Handle base64 input
        elif request.headers.get('content-type') == 'application/json':
            body = await request.json()
            if 'base64_files' not in body or not body['base64_files']:
                raise HTTPException(status_code=400, detail="No base64_files provided")
            
            # Extract parameters from body
            if default_prompt is None and 'default_prompt' in body:
                default_prompt = body['default_prompt']
            if 'provider' in body:
                provider = body['provider']
                ocr_service = get_ocr_service(provider)  # Re-get service with specified provider
            if 'clean_text' in body:
                clean_text = body['clean_text']
                
            for i, b64file in enumerate(body['base64_files']):
                file_path = await file_handler.process_base64_file(
                    b64file,
                    {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
                )
                file_paths.append(file_path)
                
                # Use individual prompt if provided, otherwise default
                if isinstance(b64file, dict) and 'prompt' in b64file:
                    prompts.append(b64file['prompt'])
                else:
                    prompts.append(default_prompt)
        else:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Perform batch OCR
        ocr_results = await ocr_service.perform_batch_ocr(file_paths, prompts, clean_text)
        
        # Create individual results
        individual_results = []
        successful = 0
        failed = 0
        
        for i, (file_path, ocr_result) in enumerate(zip(file_paths, ocr_results)):
            if ocr_result['success']:
                successful += 1
            else:
                failed += 1
                
            individual_results.append(
                SingleImageOCRResult(
                    filename=file_path.name,
                    ocr_result=ocr_result
                )
            )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchOCRResult(
            results=individual_results,
            summary={
                "total_processed": len(file_paths),
                "successful": successful,
                "failed": failed,
                "provider": ocr_service.provider
            },
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing batch OCR: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-regions/")
async def process_regions_ocr(
    request: Request,
    provider: Optional[str] = Query(None, description="OCR provider: OCI or VLLM"),
    clean_text: bool = Query(True, description="Clean and simplify extracted text")
):
    """Perform OCR on pre-cropped regions from layout detection with specified provider."""
    ocr_service = get_ocr_service(provider)
    
    try:
        # Expect JSON input with regions and their crop paths
        body = await request.json()
        
        if 'regions' not in body:
            raise HTTPException(status_code=400, detail="No regions provided")
        
        regions = body['regions']
        region_prompts = body.get('region_prompts', {})
        
        # Extract provider from body if specified
        if 'provider' in body:
            provider = body['provider']
            ocr_service = get_ocr_service(provider)  # Re-get service with specified provider
        
        if 'clean_text' in body:
            clean_text = body['clean_text']
        
        # Perform OCR on regions
        enhanced_regions = await ocr_service.perform_ocr_on_regions(
            regions, 
            region_prompts,
            clean_text
        )
        
        # Calculate summary
        total_regions = len(enhanced_regions)
        successful_ocr = sum(1 for r in enhanced_regions if r.get('ocr_result', {}).get('success', False))
        
        return {
            "regions": enhanced_regions,
            "summary": {
                "total_regions": total_regions,
                "successful_ocr": successful_ocr,
                "failed_ocr": total_regions - successful_ocr,
                "provider": ocr_service.provider
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing regions OCR: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/providers")
async def get_available_providers() -> Dict[str, Any]:
    """Get list of available OCR providers and their status."""
    providers_info = {}
    
    for provider_name, service in ocr_services.items():
        providers_info[provider_name] = {
            "available": service.ocr_available,
            "info": service.get_provider_info() if service.ocr_available else None
        }
    
    return {
        "available_providers": list(ocr_services.keys()),
        "providers_detail": providers_info,
        "default_provider": next((name for name, service in ocr_services.items() if service.ocr_available), None)
    }

@router.get("/config")
async def get_ocr_config(provider: Optional[str] = Query(None, description="OCR provider: OCI or VLLM")):
    """Get current OCR configuration for specified provider."""
    if not ocr_available:
        return {"enabled": False, "error": "No OCR services available"}
    
    try:
        ocr_service = get_ocr_service(provider)
        return {
            "enabled": True,
            **ocr_service.get_provider_info()
        }
    except HTTPException:
        return {"enabled": False, "error": f"Provider {provider} not available"}

@router.post("/test-connection")
async def test_ocr_connection(provider: Optional[str] = Query(None, description="OCR provider to test: OCI or VLLM")):
    """Test OCR service connection for specified provider."""
    try:
        ocr_service = get_ocr_service(provider)
        
        if ocr_service.ocr_available:
            return {
                "status": "connected",
                "provider": ocr_service.provider,
                "info": ocr_service.get_provider_info()
            }
        else:
            return {
                "status": "disconnected",
                "provider": ocr_service.provider,
                "error": f"{ocr_service.provider} service not available"
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

@router.get("/prompts/examples")
async def get_example_prompts() -> Dict[str, str]:
    """Get example prompts for different region types and providers."""
    return {
        # General prompts
        "general_oci": "You are an expert Arabic OCR system. Extract all text from this image. Return only the text without any explanation, description, or translation. For numbers, be precise with Arabic numerals.",
        "general_vllm": "You are an expert OCR system. Extract all text from this image accurately. Return only the extracted text without any explanations or descriptions.",
        
        # Region-specific prompts
        "photo_zone": "Extract any text visible in this photo area of an ID card:",
        "mrz_zone": "Extract the machine-readable zone (MRZ) text from this ID card. Provide each line separately:",
        "signature_zone": "Extract any text or name from this signature area:",
        "personal_info_zone": "Extract all personal information text from this area of the ID card:",
        "document_number_zone": "Extract the document/ID number from this area:",
        "name": "Extract the full name from this area of the ID card:",
        "date_of_birth": "Extract the date of birth from this area:",
        "address": "Extract the complete address from this area:",
        
        # Arabic prompts
        "arabic_general": "أنت خبير OCR عربي. استخرج جميع النصوص من هذه الصورة بدقة:",
        "arabic_name": "استخرج الاسم الكامل من هذه المنطقة في بطاقة الهوية:",
        "arabic_address": "استخرج العنوان الكامل من هذه المنطقة:",
        
        # VLLM-specific prompts
        "vllm_arabic": "Extract all Arabic text from this image. Be precise with Arabic numerals and diacritics. Return only the text.",
        "vllm_english": "Extract all English text from this image. Return only the text without explanations."
    }

@router.get("/health")
async def ocr_health() -> Dict[str, Any]:
    """Check health of all OCR services."""
    health_info = {
        "overall_status": "healthy" if ocr_available else "unavailable",
        "services": {}
    }
    
    for provider_name, service in ocr_services.items():
        health_info["services"][provider_name] = {
            "status": "healthy" if service.ocr_available else "unavailable",
            "provider": provider_name
        }
        
        if service.ocr_available:
            if provider_name == "OCI":
                health_info["services"][provider_name].update({
                    "endpoint": getattr(service, 'endpoint', 'N/A'),
                    "features": ["Arabic OCR", "Multi-language support", "High accuracy"]
                })
            elif provider_name == "VLLM":
                health_info["services"][provider_name].update({
                    "model": getattr(service, 'model', 'N/A'),
                    "base_url": getattr(service, 'client', {}).base_url if hasattr(service, 'client') else 'N/A',
                    "features": ["Vision Language Model", "Multi-modal", "Flexible prompting"]
                })
    
    return health_info