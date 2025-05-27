import os
import cv2 # type: ignore
import numpy as np
import math
import fitz # type: ignore
import logging
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
import time
from ultralytics import YOLO # type: ignore
from ..core.config import settings

logger = logging.getLogger(__name__)

class IDLayoutService:
    """Service for National ID layout detection and region extraction with clean OCR"""
    
    def __init__(self):
        try:
            # Load the YOLO model for layout detection
            self.model_path = settings.ID_LAYOUT_MODEL_PATH
            self.model = YOLO(self.model_path)
            logger.info(f"ID Layout detection model loaded successfully from {self.model_path}")
            
            # Supported file types
            self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
            self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)
            
            # Setup output directories
            self.output_dir = Path(settings.ID_LAYOUT_OUTPUT_DIR)
            self.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Define layout types
            self.layout_types = [
                "standard_id", 
                "digital_id", 
                "old_format", 
                "foreign_id", 
                "temporary_id",
                "unknown"
            ]
            
            # Define region types
            self.region_types = [
                "photo_zone", 
                "mrz_zone", 
                "signature_zone", 
                "personal_info_zone", 
                "document_number_zone",
                "fingerprint_zone",
                "barcode_zone",
                "hologram_zone",
                "header_zone"
            ]
            
            # Color mapping for visualization
            self.colors = {
                "photo_zone": (0, 0, 255),
                "mrz_zone": (0, 255, 0),
                "signature_zone": (255, 0, 0),
                "personal_info_zone": (255, 165, 0),
                "document_number_zone": (128, 0, 128),
                "fingerprint_zone": (255, 192, 203),
                "barcode_zone": (165, 42, 42),
                "hologram_zone": (0, 255, 255),
                "header_zone": (128, 128, 128)
            }
            
            # Initialize OCR service if available
            try:
                from .ocr_service import OCRService
                self.ocr_service = OCRService()
                self.ocr_available = True
                logger.info("OCR service initialized for ID Layout detection")
            except Exception as e:
                logger.warning(f"OCR service not available for ID Layout: {str(e)}")
                self.ocr_service = None
                self.ocr_available = False
            
            logger.info("ID Layout detection service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ID Layout detection service: {str(e)}")
            raise RuntimeError(f"Failed to initialize ID Layout detection service: {str(e)}")
    
    def create_file_dirs(self, filename: str) -> tuple:
        """Create directory structure for output files"""
        # Remove file extension and create safe directory name
        base_name = Path(filename).stem
        safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
        
        # Create timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"layout_{safe_name}_{timestamp}"
        
        # Create directory structure
        file_dir = self.output_dir / dir_name
        regions_dir = file_dir / "regions"
        images_dir = file_dir / "images"
        
        file_dir.mkdir(exist_ok=True, parents=True)
        regions_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        return file_dir, regions_dir, images_dir
    
    def detect_orientation(self, img: np.ndarray) -> str:
        """Detect if the ID is in portrait or landscape orientation"""
        height, width = img.shape[:2]
        return "portrait" if height > width else "landscape"
    
    def detect_layout_type(self, regions: List[Dict]) -> str:
        """Determine the layout type based on detected regions"""
        # This is a simplified logic for demonstration
        # In a production system, you would have more sophisticated rules
        
        region_types = [r["region_type"] for r in regions]
        
        if "mrz_zone" in region_types and "photo_zone" in region_types:
            return "digital_id"
        elif "barcode_zone" in region_types:
            return "standard_id"
        elif "fingerprint_zone" in region_types:
            return "foreign_id"
        elif len(region_types) <= 2:
            return "temporary_id"
        elif "header_zone" in region_types:
            return "old_format"
        else:
            return "unknown"
    
    def assess_region_quality(self, img_region: np.ndarray, region_type: str) -> float:
        """Assess the quality of a region based on its type"""
        # This is a simplified quality assessment
        # Real implementations would use more advanced metrics
        
        # Check if image is too small
        h, w = img_region.shape[:2]
        if h < 20 or w < 20:
            return 0.3
            
        # Convert to grayscale for analysis
        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_region
            
        # Calculate basic image quality metrics
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Normalize to 0-1 range
        blur_score = min(1.0, blur / 1000)
        contrast_score = min(1.0, std_val / 80)
        
        # Different types may need different quality metrics
        if region_type == "photo_zone":
            # Photos need good contrast and sharpness
            return (blur_score * 0.7 + contrast_score * 0.3)
        elif region_type == "mrz_zone":
            # MRZ needs good contrast
            return contrast_score
        elif region_type == "signature_zone":
            # Signatures need good sharpness
            return blur_score
        else:
            # Default quality metric
            return (blur_score * 0.5 + contrast_score * 0.5)
    
    async def process_image(self, img: np.ndarray, filename: str, page_num: int = None) -> Dict[str, Any]:
        """Process a single image and detect ID layout regions"""
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(filename).stem
        
        # Create directory structure
        file_dir, regions_dir, images_dir = self.create_file_dirs(filename)
        
        # Add page number to filename if processing PDF pages
        if page_num is not None:
            base_filename = f"{original_filename}_page{page_num}_{timestamp}"
        else:
            base_filename = f"{original_filename}_{timestamp}"
        
        # Detect orientation
        orientation = self.detect_orientation(img)
        
        # Create a copy for annotation
        annotated_img = img.copy()
        
        # Run layout detection model
        results = self.model(img, conf=0.3)  # Lower threshold for layout regions
        result = results[0]
        
        # Process detections
        regions = []
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0].item())
                cls_name = result.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                
                # Make sure coordinates are within image bounds
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # If valid region
                if x2 > x1 and y2 > y1:
                    # Crop the region
                    region_img = img[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Assess quality
                    quality_score = self.assess_region_quality(region_img, cls_name)
                    
                    # Save the cropped region
                    region_filename = f"{base_filename}_{cls_name}_{i}_{conf:.2f}.jpg"
                    region_path = regions_dir / region_filename
                    cv2.imwrite(str(region_path), region_img)
                    
                    # Create region data
                    region = {
                        "region_type": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "crop_path": str(region_path),
                        "quality_score": quality_score
                    }
                    
                    # Draw on annotated image
                    color = self.colors.get(cls_name, (0, 255, 0))
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add labels with quality score
                    label = f"{cls_name} ({conf:.2f}) Q:{quality_score:.2f}"
                    cv2.putText(annotated_img, label, (int(x1), int(y1-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    regions.append(region)
        
        # Determine layout type
        layout_type = self.detect_layout_type(regions)
        
        # Add layout type to the annotated image
        cv2.putText(annotated_img, f"Layout: {layout_type}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_img, f"Orientation: {orientation}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save annotated image
        annotated_img_path = images_dir / f"annotated_{base_filename}.jpg"
        cv2.imwrite(str(annotated_img_path), annotated_img)
        
        # Return results
        return {
            "layout_type": layout_type,
            "orientation": orientation,
            "regions": regions,
            "output_folder": str(file_dir),
            "annotated_image_path": str(annotated_img_path)
        }
    
    async def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF file and detect ID layout on all pages"""
        filename = file_path.name
        start_time = time.time()
        
        try:
            pdf_document = fitz.open(str(file_path))
            
            if len(pdf_document) == 0:
                return {
                    "filename": filename,
                    "status": "error",
                    "message": "PDF file contains no pages",
                    "regions": []
                }
            
            all_regions = []
            all_pages_results = []
            
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
                
                # Process this page
                page_result = await self.process_image(img, filename, page_num)
                page_result["page_number"] = page_num + 1
                all_pages_results.append(page_result)
                all_regions.extend(page_result["regions"])
            
            # For PDF, use the layout type of the first page with detected regions
            layout_type = "unknown"
            orientation = "unknown"
            for page_result in all_pages_results:
                if page_result.get("layout_type") != "unknown":
                    layout_type = page_result.get("layout_type")
                    orientation = page_result.get("orientation")
                    break
            
            # Get processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "filename": filename,
                "status": "success",
                "message": f"Processed {len(pdf_document)} pages",
                "layout_type": layout_type,
                "orientation": orientation,
                "regions": all_regions,
                "pages": all_pages_results,
                "processing_time_ms": processing_time_ms
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {
                "filename": filename,
                "status": "error",
                "message": f"Error processing PDF: {str(e)}",
                "regions": []
            }
    
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) and return ID layout detection results"""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            file_type = file_path.suffix.lower()
            filename = file_path.name
            
            if file_type in self.supported_docs:
                # Process PDF
                result = await self.process_pdf(file_path)
                
            elif file_type in self.supported_images:
                # Process image
                img = cv2.imread(str(file_path))
                
                # If OpenCV fails, try PIL
                if img is None:
                    pil_img = Image.open(file_path)
                    img = np.array(pil_img.convert('RGB'))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if img is None:
                    return {
                        "filename": filename,
                        "status": "error",
                        "message": "Could not decode image data",
                        "regions": []
                    }
                
                # Process the image
                image_result = await self.process_image(img, filename)
                
                # Add processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                result = {
                    "filename": filename,
                    "status": "success",
                    "message": "Image processed successfully",
                    "layout_type": image_result["layout_type"],
                    "orientation": image_result["orientation"],
                    "regions": image_result["regions"],
                    "output_folder": image_result["output_folder"],
                    "annotated_image_path": image_result["annotated_image_path"],
                    "processing_time_ms": processing_time_ms
                }
            else:
                return {
                    "filename": filename,
                    "status": "error",
                    "message": f"Unsupported file type: {file_type}",
                    "regions": []
                }
                
            return result
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "status": "error",
                "message": f"Error processing file: {str(e)}",
                "regions": []
            }
    
    def _create_simplified_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create simplified response with clean OCR text only"""
        simplified_regions = []
        
        for region in result.get("regions", []):
            ocr_result = region.get("ocr_result", {})
            
            simplified_region = {
                "region_type": region.get("region_type"),
                "bbox": region.get("bbox"),
                "confidence": region.get("confidence")
            }
            
            if ocr_result.get("success", False):
                simplified_region["text"] = ocr_result.get("extracted_text", "")
            else:
                simplified_region["text"] = ""
            
            simplified_regions.append(simplified_region)
        
        return {
            "filename": result.get("filename"),
            "extracted_data": simplified_regions,
            "processing_time_ms": result.get("processing_time_ms")
        }
    
    async def process_file_with_ocr(
        self, 
        file_path: Path, 
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        region_prompts: Optional[Dict[str, str]] = None,
        clean_text: bool = True,
        simplified_response: bool = False
    ) -> Dict[str, Any]:
        """Process a file with optional OCR and text cleaning"""
        start_time = time.time()
        
        # First perform standard layout detection
        layout_result = await self.process_file(file_path)
        
        if enable_ocr and layout_result.get("status") == "success":
            logger.info(f"OCR enabled for layout detection. OCR available: {self.ocr_available}")
            
            if self.ocr_available:
                try:
                    # Extract regions for OCR
                    regions = layout_result.get("regions", [])
                    
                    if regions:
                        logger.info(f"Processing {len(regions)} regions with OCR")
                        # Perform OCR on regions with text cleaning
                        enhanced_regions = await self.ocr_service.perform_ocr_on_regions(
                            regions, 
                            region_prompts,
                            clean_text=clean_text
                        )
                        
                        # Update layout result with OCR
                        layout_result["regions"] = enhanced_regions
                        layout_result["ocr_enabled"] = True
                        
                        # Add OCR summary
                        total_regions = len(enhanced_regions)
                        successful_ocr = sum(
                            1 for r in enhanced_regions 
                            if r.get('ocr_result', {}).get('success', False)
                        )
                        
                        layout_result["ocr_summary"] = {
                            "total_regions": total_regions,
                            "successful_ocr": successful_ocr,
                            "failed_ocr": total_regions - successful_ocr,
                            "ocr_available": self.ocr_available
                        }
                        
                        logger.info(f"OCR completed: {successful_ocr}/{total_regions} regions processed successfully")
                    else:
                        layout_result["ocr_enabled"] = False
                        layout_result["ocr_summary"] = {
                            "message": "No regions detected for OCR"
                        }
                        logger.info("No regions detected for OCR processing")
                        
                except Exception as e:
                    logger.error(f"Error performing OCR on layout regions: {str(e)}")
                    layout_result["ocr_enabled"] = False
                    layout_result["ocr_summary"] = {
                        "error": str(e),
                        "message": "OCR processing failed"
                    }
            else:
                layout_result["ocr_enabled"] = False
                layout_result["ocr_summary"] = {
                    "message": "OCR service not available",
                    "ocr_available": False
                }
                logger.warning("OCR requested but service not available")
        else:
            layout_result["ocr_enabled"] = False
            if enable_ocr:
                layout_result["ocr_summary"] = {
                    "message": "Layout detection failed, OCR skipped"
                }
        
        # Update processing time
        processing_time_ms = (time.time() - start_time) * 1000
        layout_result["processing_time_ms"] = processing_time_ms
        
        # Return simplified response if requested
        if simplified_response and enable_ocr:
            return self._create_simplified_response(layout_result)
        
        return layout_result
    
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
                    "message": f"Error processing file: {str(e)}",
                    "regions": []
                })
                
        return results
    
    async def process_batch_with_ocr(
        self, 
        file_paths: List[Path],
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        region_prompts: Optional[Dict[str, str]] = None,
        clean_text: bool = True,
        simplified_response: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple files with optional OCR and text cleaning"""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file_with_ocr(
                    file_path, 
                    enable_ocr, 
                    ocr_prompt, 
                    region_prompts,
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
                    "regions": [],
                    "ocr_enabled": False
                })
                
        return results