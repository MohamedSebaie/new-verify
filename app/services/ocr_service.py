import oci
from oci.signer import Signer # type: ignore
import json
import requests
from string import Template
import base64
import re
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import asyncio
import aiofiles # type: ignore
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI # type: ignore
from ..core.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    """Service for OCR processing using Oracle Cloud Infrastructure or VLLM with Arabic focus"""
    
    def __init__(self, provider: Optional[str] = None):
        """Initialize OCR service with specified provider (OCI or VLLM)"""
        self.provider = provider or settings.OCR_PROVIDER
        self.ocr_available = False
        
        # Initialize based on provider
        if self.provider == "OCI":
            self._init_oci()
        elif self.provider == "VLLM":
            self._init_vllm()
        else:
            logger.error(f"Unknown OCR provider: {self.provider}")
            raise ValueError(f"Unsupported OCR provider: {self.provider}")
        
        # Thread executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"OCR service initialized successfully with {self.provider} provider")
    
    def _init_oci(self):
        """Initialize OCI OCR service"""
        try:
            # Initialize OCI configuration
            self.config = oci.config.from_file(settings.OCI_CONFIG_FILE)
            
            # Set up authentication
            self.auth = Signer(
                tenancy=self.config['tenancy'],
                user=self.config['user'],
                fingerprint=self.config['fingerprint'],
                private_key_file_location=self.config['key_file']
            )
            
            # API endpoint
            self.endpoint = settings.OCI_OCR_ENDPOINT
            
            # Default headers
            self.headers = {"Content-Type": "application/json"}
            
            # OCI-specific settings
            self.default_prompt = settings.OCI_DEFAULT_OCR_PROMPT
            self.max_tokens = settings.OCI_OCR_MAX_TOKENS
            self.temperature = settings.OCI_OCR_TEMPERATURE
            self.top_p = settings.OCI_OCR_TOP_P
            
            self.ocr_available = True
            logger.info("OCI OCR service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OCI OCR service: {str(e)}")
            self.ocr_available = False
            raise RuntimeError(f"Failed to initialize OCI OCR service: {str(e)}")
    
    def _init_vllm(self):
        """Initialize VLLM OCR service"""
        try:
            # Initialize OpenAI client for VLLM
            self.client = OpenAI(
                api_key=settings.VLLM_API_KEY,
                base_url=settings.VLLM_BASE_URL
            )
            
            # VLLM-specific settings
            self.model = settings.VLLM_MODEL
            self.available_models = settings.VLLM_AVAILABLE_MODELS
            self.default_prompt = settings.VLLM_DEFAULT_OCR_PROMPT
            self.max_tokens = settings.VLLM_OCR_MAX_TOKENS
            self.temperature = settings.VLLM_OCR_TEMPERATURE
            
            # Test connection
            try:
                # Simple test to verify VLLM connection
                test_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    temperature=0.1
                )
                self.ocr_available = True
                logger.info("VLLM OCR service initialized and tested successfully")
                
            except Exception as e:
                logger.warning(f"VLLM connection test failed: {str(e)}")
                self.ocr_available = False
                
        except Exception as e:
            logger.error(f"Error initializing VLLM OCR service: {str(e)}")
            self.ocr_available = False
            raise RuntimeError(f"Failed to initialize VLLM OCR service: {str(e)}")
    
    def clean_extracted_text(self, raw_text: str) -> str:
        """Extract and clean text from OCR response with Arabic number correction"""
        if not raw_text:
            return ""
        
        # First, normalize the text
        text = raw_text.strip()
        
        # Remove common OCR response patterns and explanations
        patterns_to_remove = [
            r"the text extracted.*?is[:\s]*",
            r"extracted text.*?is[:\s]*",
            r"the image shows[:\s]*",
            r"this represents[:\s]*",
            r"this translates to[:\s]*",
            r"in english[:\s]*",
            r"the text is[:\s]*",
            r"text[:\s]*:",
            r"result[:\s]*:",
            r"output[:\s]*:",
            r"arabic text[:\s]*:",
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Extract text between ** markers first (often contains the actual text)
        bold_pattern = r'\*\*(.*?)\*\*'
        bold_matches = re.findall(bold_pattern, text)
        
        if bold_matches:
            text = bold_matches[0].strip()
        
        # Remove all markdown formatting
        text = re.sub(r'[*_`#]', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common prefixes and suffixes
        text = re.sub(r'^[:\-\.\s]+|[:\-\.\s]+$', '', text)
        
        # Handle Arabic numbers and convert . to ٠ (Arabic zero)
        number_patterns = [
            r'[\u0660-\u0669\.\/\-\d]+',  # Arabic numbers with dots, slashes, dashes
            r'\d+[\.\d\/\-]*\d*',         # Western numbers with dots, slashes, dashes
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if re.search(r'[\d\u0660-\u0669]', match):
                    corrected = match.replace('.', '٠')
                    text = text.replace(match, corrected)
        
        # If the text still seems verbose, try to extract just the meaningful part
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not any(skip in line.lower() for skip in [
                'text', 'image', 'shows', 'represents', 'translates', 'english'
            ]):
                text = line
                break
        
        # Final cleaning
        text = re.sub(r'^[\s\-:\.]+|[\s\-:\.]+$', '', text)
        
        return text
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise RuntimeError(f"Failed to encode image: {str(e)}")
    
    async def encode_image_async(self, image_path: Path) -> str:
        """Async version of image encoding"""
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise RuntimeError(f"Failed to encode image: {str(e)}")
    
    def _perform_oci_ocr_sync(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """Synchronous OCI OCR processing"""
        payload = {
            "model": "odsc-llm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": Template("""<|image_1|>\n $prompt""").substitute(prompt=prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        try:
            response = requests.post(
                self.endpoint, 
                json=payload, 
                auth=self.auth, 
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "confidence": 1.0,  # OCI doesn't provide confidence scores
                "model_response": result
            }
        except Exception as e:
            logger.error(f"OCI OCR API request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": ""
            }
    
    def _perform_vllm_ocr_sync(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """Synchronous VLLM OCR processing"""
        try:
            # Prepare messages for VLLM
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that can understand images and extract text accurately."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
            
            # Make API call to VLLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            extracted_text = response.choices[0].message.content
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "confidence": 1.0,  # VLLM doesn't provide confidence scores
                "model_response": {
                    "model": self.model,
                    "response": extracted_text
                }
            }
            
        except Exception as e:
            logger.error(f"VLLM OCR API request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": ""
            }
    
    async def perform_ocr(self, image_path: Path, prompt: Optional[str] = None, clean_text: bool = True) -> Dict[str, Any]:
        """Perform OCR on a single image with optional text cleaning"""
        if not self.ocr_available:
            return {
                "success": False,
                "error": f"{self.provider} OCR service not available",
                "extracted_text": "",
                "image_path": str(image_path),
                "provider": self.provider
            }
        
        if prompt is None:
            prompt = self.default_prompt
        
        try:
            # Encode image asynchronously
            image_base64 = await self.encode_image_async(image_path)
            
            # Perform OCR based on provider
            loop = asyncio.get_event_loop()
            if self.provider == "OCI":
                result = await loop.run_in_executor(
                    self.executor, 
                    self._perform_oci_ocr_sync, 
                    image_base64, 
                    prompt
                )
            elif self.provider == "VLLM":
                result = await loop.run_in_executor(
                    self.executor, 
                    self._perform_vllm_ocr_sync, 
                    image_base64, 
                    prompt
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Clean the extracted text if requested and successful
            if clean_text and result.get("success") and result.get("extracted_text"):
                original_text = result["extracted_text"]
                cleaned_text = self.clean_extracted_text(original_text)
                result["extracted_text"] = cleaned_text
                result["original_text"] = original_text  # Keep original for debugging
            
            # Add metadata
            result["image_path"] = str(image_path)
            result["prompt_used"] = prompt
            result["provider"] = self.provider
            
            return result
        except Exception as e:
            logger.error(f"Error performing OCR on {image_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "image_path": str(image_path),
                "prompt_used": prompt,
                "provider": self.provider
            }
    
    async def perform_batch_ocr(
        self, 
        image_paths: List[Path], 
        prompts: Optional[List[str]] = None,
        clean_text: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform OCR on multiple images with optional text cleaning"""
        if prompts is None:
            prompts = [self.default_prompt] * len(image_paths)
        elif len(prompts) != len(image_paths):
            prompts = (prompts * ((len(image_paths) // len(prompts)) + 1))[:len(image_paths)]
        
        # Process all images concurrently
        tasks = [
            self.perform_ocr(image_path, prompt, clean_text) 
            for image_path, prompt in zip(image_paths, prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "extracted_text": "",
                    "image_path": str(image_paths[i]),
                    "prompt_used": prompts[i],
                    "provider": self.provider
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def perform_ocr_on_regions(
        self, 
        regions: List[Dict[str, Any]], 
        region_specific_prompts: Optional[Dict[str, str]] = None,
        clean_text: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform OCR on cropped regions with general Arabic OCR"""
        if not regions:
            return []
        
        # Use general Arabic OCR prompt for all regions
        general_prompt = "You are an expert Arabic OCR system. Extract all text visible in this image. Return only the text without any explanation or description. Be precise with Arabic numerals and text."
        
        # Prepare prompts for each region
        region_prompts = []
        for region in regions:
            region_type = region.get("region_type") or region.get("element_type", "unknown")
            
            # Use region-specific prompt if provided, otherwise use general Arabic prompt
            if region_specific_prompts and region_type in region_specific_prompts:
                prompt = region_specific_prompts[region_type]
            else:
                prompt = general_prompt
            
            region_prompts.append(prompt)
        
        # Get image paths from regions
        image_paths = []
        valid_regions = []
        for i, region in enumerate(regions):
            crop_path = region.get("crop_path")
            if crop_path and Path(crop_path).exists():
                image_paths.append(Path(crop_path))
                valid_regions.append(region)
        
        if not image_paths:
            logger.warning("No valid crop paths found in regions")
            return regions  # Return original regions without OCR
        
        # Perform batch OCR with text cleaning
        ocr_results = await self.perform_batch_ocr(
            image_paths, 
            region_prompts[:len(image_paths)], 
            clean_text=clean_text
        )
        
        # Combine OCR results with region information
        enhanced_results = []
        result_index = 0
        
        for region in regions:
            crop_path = region.get("crop_path")
            if crop_path and Path(crop_path).exists() and result_index < len(ocr_results):
                ocr_result = ocr_results[result_index]
                enhanced_result = {
                    **region,
                    "ocr_result": ocr_result
                }
                enhanced_results.append(enhanced_result)
                result_index += 1
            else:
                enhanced_result = {
                    **region,
                    "ocr_result": {
                        "success": False,
                        "error": "No valid crop path or OCR result",
                        "extracted_text": "",
                        "provider": self.provider
                    }
                }
                enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current OCR provider"""
        info = {
            "provider": self.provider,
            "available": self.ocr_available,
            "default_prompt": self.default_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if self.provider == "VLLM":
            info.update({
                "model": self.model,
                "available_models": self.available_models,
                "base_url": settings.VLLM_BASE_URL
            })
        elif self.provider == "OCI":
            info.update({
                "endpoint": self.endpoint,
                "top_p": self.top_p
            })
        
        return info
    
    def switch_provider(self, new_provider: str):
        """Switch to a different OCR provider"""
        if new_provider not in ["OCI", "VLLM"]:
            raise ValueError(f"Unsupported provider: {new_provider}. Use 'OCI' or 'VLLM'")
        
        if new_provider != self.provider:
            logger.info(f"Switching OCR provider from {self.provider} to {new_provider}")
            self.provider = new_provider
            
            # Re-initialize with new provider
            if new_provider == "OCI":
                self._init_oci()
            elif new_provider == "VLLM":
                self._init_vllm()
    
    def __del__(self):
        """Cleanup thread executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)