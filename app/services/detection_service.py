import os
import cv2 # type: ignore
import numpy as np
import fitz  # type: ignore # PyMuPDF
import logging
import tempfile
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
import time
from ultralytics import YOLO # type: ignore
from ..core.config import settings

logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self):
        try:
            # Load the YOLO model
            self.model_path = settings.DETECTION_MODEL_PATH
            self.model = YOLO(self.model_path)
            logger.info(f"Detection model loaded successfully from {self.model_path}")
            
            # Supported file types
            self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
            self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)
            
            # Setup output directories
            self.output_dir = Path(settings.DETECTION_OUTPUT_DIR)
            self.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize OCR service if available
            try:
                from .ocr_service import OCRService
                self.ocr_service = OCRService()
                self.ocr_available = True
                logger.info("OCR service initialized for detection")
            except Exception as e:
                logger.warning(f"OCR service not available for detection: {str(e)}")
                self.ocr_service = None
                self.ocr_available = False
            
            logger.info("Detection service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detection service: {str(e)}")
            raise RuntimeError(f"Failed to initialize detection service: {str(e)}")
            
    def create_file_dirs(self, filename: str) -> tuple:
        """Create directory structure for output files"""
        # Remove file extension and create safe directory name
        base_name = Path(filename).stem
        safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
        
        # Create timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{safe_name}_{timestamp}"
        
        # Create directory structure
        file_dir = self.output_dir / dir_name
        crops_dir = file_dir / "crops"
        images_dir = file_dir / "images"
        
        file_dir.mkdir(exist_ok=True, parents=True)
        crops_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        return file_dir, crops_dir, images_dir
        
    async def process_image(self, img: np.ndarray, filename: str, page_num: int = None) -> Dict[str, Any]:
        """Process a single image and return detection results"""
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(filename).stem
        
        # Create directory structure
        file_dir, crops_dir, images_dir = self.create_file_dirs(filename)
        
        # Add page number to filename if processing PDF pages
        if page_num is not None:
            base_filename = f"{original_filename}_page{page_num}_{timestamp}"
        else:
            base_filename = f"{original_filename}_{timestamp}"
        
        # Run detection
        results = self.model(img)
        result = results[0]
        
        # Save annotated image
        annotated_img = result.plot()
        annotated_img_path = images_dir / f"annotated_{base_filename}.jpg"
        cv2.imwrite(str(annotated_img_path), annotated_img)
        
        # Process detections
        detections = []
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0].item())
                cls_name = result.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                
                # Crop and save detection
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                crop_path = None
                if crop.size > 0:
                    crop_filename = f"{base_filename}_{cls_name}_{i}_{conf:.2f}.jpg"
                    crop_path = crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                
                # Add to results with crop path
                detections.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "crop_path": str(crop_path) if crop_path else None
                })
        
        # Return results for this image
        return {
            "folder": str(file_dir),
            "annotated_image_path": str(annotated_img_path),
            "num_detections": len(detections),
            "detections": detections
        }
        
    async def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process each page of a PDF file"""
        results = []
        filename = file_path.name
        
        try:
            pdf_document = fitz.open(str(file_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Higher DPI for better quality images
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to OpenCV format
                img_data = pix.samples
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                # Convert RGBA to BGR if needed
                if pix.n == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 1:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Process this page as an image
                page_result = await self.process_image(img, filename, page_num)
                page_result["page_number"] = page_num + 1
                results.append(page_result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise RuntimeError(f"Error processing PDF: {str(e)}")
        
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) and return detection results"""
        try:
            file_path = Path(file_path)
            file_type = file_path.suffix.lower()
            
            # Get base folder for this file
            file_dir, _, _ = self.create_file_dirs(file_path.name)
            
            if file_type in self.supported_docs:
                # Process PDF
                results = await self.process_pdf(file_path)
                
                # Combine results from all pages
                return {
                    "filename": file_path.name,
                    "file_type": "pdf",
                    "output_folder": str(file_dir),
                    "pages_processed": len(results),
                    "results": results
                }
                
            elif file_type in self.supported_images:
                # Process image
                # Read image with OpenCV
                img = cv2.imread(str(file_path))
                
                # If OpenCV fails, try PIL
                if img is None:
                    pil_img = Image.open(file_path)
                    img = np.array(pil_img.convert('RGB'))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if img is None:
                    raise RuntimeError("Could not decode image data")
                
                result = await self.process_image(img, file_path.name)
                
                return {
                    "filename": file_path.name,
                    "file_type": "image",
                    "output_folder": str(file_dir),
                    "pages_processed": 1,
                    "results": [result]
                }
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"File processing failed: {str(e)}")
    
    def _create_simplified_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create simplified response with clean OCR text only"""
        simplified_detections = []
        
        # Extract all detections from all pages/results
        all_detections = []
        if result.get("file_type") == "pdf":
            for page_result in result.get("results", []):
                all_detections.extend(page_result.get("detections", []))
        else:
            for page_result in result.get("results", []):
                all_detections.extend(page_result.get("detections", []))
        
        for detection in all_detections:
            ocr_result = detection.get("ocr_result", {})
            
            simplified_detection = {
                "class_name": detection.get("class_name"),
                "bbox": detection.get("bbox"),
                "confidence": detection.get("confidence")
            }
            
            if ocr_result.get("success", False):
                simplified_detection["text"] = ocr_result.get("extracted_text", "")
            else:
                simplified_detection["text"] = ""
            
            simplified_detections.append(simplified_detection)
        
        return {
            "filename": result.get("filename"),
            "extracted_data": simplified_detections,
            "processing_time_ms": result.get("processing_time_ms")
        }
    
    async def process_file_with_ocr(
        self, 
        file_path: Path, 
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        class_prompts: Optional[Dict[str, str]] = None,
        clean_text: bool = True,
        simplified_response: bool = False
    ) -> Dict[str, Any]:
        """Process a file with optional OCR on detected crops"""
        start_time = time.time()
        
        # First perform standard detection
        detection_result = await self.process_file(file_path)
        
        if enable_ocr and self.ocr_available:
            logger.info(f"OCR enabled for detection. OCR available: {self.ocr_available}")
            
            try:
                # Extract all detections from all pages
                all_detections = []
                if detection_result.get("file_type") == "pdf":
                    for page_result in detection_result.get("results", []):
                        all_detections.extend(page_result.get("detections", []))
                else:
                    for page_result in detection_result.get("results", []):
                        all_detections.extend(page_result.get("detections", []))
                
                # Filter detections that have crop paths
                detections_with_crops = [d for d in all_detections if d.get("crop_path")]
                
                if detections_with_crops:
                    logger.info(f"Processing {len(detections_with_crops)} detections with OCR")
                    
                    # Convert detections to regions format for OCR
                    regions_for_ocr = []
                    for detection in detections_with_crops:
                        region = {
                            "region_type": detection.get("class_name"),
                            "element_type": detection.get("class_name"),
                            "confidence": detection.get("confidence"),
                            "bbox": detection.get("bbox"),
                            "crop_path": detection.get("crop_path")
                        }
                        regions_for_ocr.append(region)
                    
                    # Perform OCR on detections
                    enhanced_detections = await self.ocr_service.perform_ocr_on_regions(
                        regions_for_ocr, 
                        class_prompts,
                        clean_text=clean_text
                    )
                    
                    # Update detections with OCR results
                    ocr_detection_map = {}
                    for enhanced in enhanced_detections:
                        crop_path = enhanced.get("crop_path")
                        if crop_path:
                            ocr_detection_map[crop_path] = enhanced.get("ocr_result")
                    
                    # Update original detection results with OCR
                    if detection_result.get("file_type") == "pdf":
                        for page_result in detection_result.get("results", []):
                            for detection in page_result.get("detections", []):
                                crop_path = detection.get("crop_path")
                                if crop_path and crop_path in ocr_detection_map:
                                    detection["ocr_result"] = ocr_detection_map[crop_path]
                    else:
                        for page_result in detection_result.get("results", []):
                            for detection in page_result.get("detections", []):
                                crop_path = detection.get("crop_path")
                                if crop_path and crop_path in ocr_detection_map:
                                    detection["ocr_result"] = ocr_detection_map[crop_path]
                    
                    # Add OCR summary
                    total_detections = len(detections_with_crops)
                    successful_ocr = sum(
                        1 for crop_path, ocr_result in ocr_detection_map.items()
                        if ocr_result.get("success", False)
                    )
                    
                    detection_result["ocr_enabled"] = True
                    detection_result["ocr_summary"] = {
                        "total_detections": total_detections,
                        "successful_ocr": successful_ocr,
                        "failed_ocr": total_detections - successful_ocr,
                        "ocr_available": self.ocr_available
                    }
                    
                    logger.info(f"OCR completed: {successful_ocr}/{total_detections} detections processed successfully")
                else:
                    detection_result["ocr_enabled"] = False
                    detection_result["ocr_summary"] = {
                        "message": "No detections with crops found for OCR"
                    }
                    logger.info("No detections with crops found for OCR processing")
                    
            except Exception as e:
                logger.error(f"Error performing OCR on detections: {str(e)}")
                detection_result["ocr_enabled"] = False
                detection_result["ocr_summary"] = {
                    "error": str(e),
                    "message": "OCR processing failed"
                }
        else:
            detection_result["ocr_enabled"] = False
            if enable_ocr and not self.ocr_available:
                detection_result["ocr_summary"] = {
                    "message": "OCR service not available",
                    "ocr_available": False
                }
                logger.warning("OCR requested but service not available")
        
        # Update processing time
        processing_time_ms = (time.time() - start_time) * 1000
        detection_result["processing_time_ms"] = processing_time_ms
        
        # Return simplified response if requested
        if simplified_response and enable_ocr:
            return self._create_simplified_response(detection_result)
        
        return detection_result
        
    async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files and return results"""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "filename": Path(file_path).name,
                    "status": "error",
                    "message": str(e),
                    "error": str(e)
                })
                
        return results
    
    async def process_batch_with_ocr(
        self, 
        file_paths: List[Path],
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        class_prompts: Optional[Dict[str, str]] = None,
        clean_text: bool = True,
        simplified_response: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple files with optional OCR"""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file_with_ocr(
                    file_path, 
                    enable_ocr, 
                    ocr_prompt, 
                    class_prompts,
                    clean_text,
                    simplified_response
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path} with OCR: {str(e)}")
                results.append({
                    "filename": Path(file_path).name,
                    "status": "error",
                    "message": f"Error processing file: {str(e)}",
                    "results": [],
                    "ocr_enabled": False
                })
                
        return results