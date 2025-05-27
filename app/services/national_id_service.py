import os
import cv2 # type: ignore
import numpy as np
import fitz  # type: ignore # PyMuPDF
import logging
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
import time
from ultralytics import YOLO  # type: ignore
from ..core.config import settings
from ..models.ocr import RegionWithOCR, NationalIDWithOCRResult
from .ocr_service import OCRService

logger = logging.getLogger(__name__)

class NationalIDService:
    """Service for National ID card processing and element detection"""
    
    def __init__(self):
        try:
            # Load the YOLO model
            self.model_path = settings.NATIONAL_ID_MODEL_PATH
            self.model = YOLO(self.model_path)
            logger.info(f"National ID detection model loaded successfully from {self.model_path}")
            
            # Supported file types
            self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
            self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)
            
            # Setup output directories
            self.output_dir = Path(settings.NATIONAL_ID_OUTPUT_DIR)
            self.output_dir.mkdir(exist_ok=True, parents=True)
            
            # Define ID elements
            self.id_elements = [
                "ID_Front", "ID_Back", "Agent_Signature", "Store_Stamp", 
                "Photo", "Name", "Date_of_Birth", "ID_Number", "Signature", "Address"
            ]
            
            # Elements typically found on front and back
            self.front_elements = ["ID_Front", "Photo", "Name", "Date_of_Birth", "ID_Number"]
            self.back_elements = ["ID_Back", "Signature", "Address", "Agent_Signature", "Store_Stamp"]
            
            # Color mapping for visualization
            self.colors = {
                "ID_Front": (0, 0, 255),       # Red
                "ID_Back": (0, 255, 0),        # Green
                "Agent_Signature": (255, 0, 0), # Blue
                "Store_Stamp": (255, 165, 0),   # Orange
                "Photo": (128, 0, 128),        # Purple
                "Name": (255, 192, 203),       # Pink
                "Date_of_Birth": (0, 255, 255), # Yellow
                "ID_Number": (128, 128, 128),  # Gray
                "Signature": (165, 42, 42),    # Brown
                "Address": (0, 128, 128)       # Teal
            }
            # Initialize OCR service
            try:
                self.ocr_service = OCRService()
                self.ocr_available = True
                logger.info("OCR service initialized for National ID processing")
            except Exception as e:
                logger.warning(f"OCR service not available for National ID: {str(e)}")
                self.ocr_service = None
                self.ocr_available = False
            
            logger.info("National ID detection service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing National ID detection service: {str(e)}")
            raise RuntimeError(f"Failed to initialize National ID detection service: {str(e)}")
            
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
    
    def determine_id_type(self, elements: List[Dict]) -> str:
        """Determine if the ID is front, back, or both"""
        element_types = {e["element_type"] for e in elements}
        
        has_front = "ID_Front" in element_types or any(e in element_types for e in self.front_elements)
        has_back = "ID_Back" in element_types or any(e in element_types for e in self.back_elements)
        
        if has_front and has_back:
            return "both"
        elif has_front:
            return "front"
        elif has_back:
            return "back"
        else:
            return "unknown"
        
    async def process_image(self, img: np.ndarray, filename: str, page_num: int = None) -> Dict[str, Any]:
        """Process a single image and detect National ID elements"""
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
        results = self.model(img, conf=0.4)  # Higher confidence threshold for ID elements
        result = results[0]
        
        # Create a copy for annotation
        annotated_img = img.copy()
        
        # Process detections
        elements = []
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0].item())
                cls_name = result.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                
                # Add to results
                element = {
                    "element_type": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                }
                
                # Draw rectangle on annotated image
                color = self.colors.get(cls_name, (0, 255, 0))  # Default to green if not in color map
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_img, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Crop and save detection
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    crop_filename = f"{base_filename}_{cls_name}_{i}_{conf:.2f}.jpg"
                    crop_path = crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    element["crop_path"] = str(crop_path)
                
                elements.append(element)
        
        # Save annotated image
        annotated_img_path = images_dir / f"annotated_{base_filename}.jpg"
        cv2.imwrite(str(annotated_img_path), annotated_img)
        
        # Determine ID type
        id_type = self.determine_id_type(elements)
        
        # Return results
        return {
            "elements": elements,
            "id_type": id_type,
            "output_folder": str(file_dir),
            "annotated_image_path": str(annotated_img_path)
        }
        
    async def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF file and detect National ID elements on all pages"""
        filename = file_path.name
        start_time = time.time()
        
        try:
            pdf_document = fitz.open(str(file_path))
            
            if len(pdf_document) == 0:
                return {
                    "filename": filename,
                    "status": "error",
                    "message": "PDF file contains no pages",
                    "elements": []
                }
            
            all_elements = []
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
                all_elements.extend(page_result["elements"])
            
            # Determine overall ID type
            id_type = self.determine_id_type(all_elements)
            
            # Get processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "filename": filename,
                "status": "success",
                "message": f"Processed {len(pdf_document)} pages",
                "id_type": id_type,
                "elements": all_elements,
                "pages": all_pages_results,
                "processing_time_ms": processing_time_ms
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {
                "filename": filename,
                "status": "error",
                "message": f"Error processing PDF: {str(e)}",
                "elements": []
            }
        
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) and return National ID detection results"""
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
                        "elements": []
                    }
                
                # Process the image
                image_result = await self.process_image(img, filename)
                
                # Add processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                result = {
                    "filename": filename,
                    "status": "success",
                    "message": "Image processed successfully",
                    "id_type": image_result["id_type"],
                    "elements": image_result["elements"],
                    "output_folder": image_result["output_folder"],
                    "annotated_image_path": image_result["annotated_image_path"],
                    "processing_time_ms": processing_time_ms
                }
            else:
                return {
                    "filename": filename,
                    "status": "error",
                    "message": f"Unsupported file type: {file_type}",
                    "elements": []
                }
                
            return result
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "status": "error",
                "message": f"Error processing file: {str(e)}",
                "elements": []
            }
        
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
                    "elements": []
                })
                
        return results
    
    async def process_file_with_ocr(
        self, 
        file_path: Path,
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        element_prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Process a National ID file with optional OCR on detected elements"""
        start_time = time.time()
        
        # First perform standard National ID processing
        id_result = await self.process_file(file_path)
        
        if enable_ocr and self.ocr_available and id_result.get("status") == "success":
            try:
                # Extract elements for OCR (rename 'elements' to match region format)
                elements = id_result.get("elements", [])
                
                # Convert elements to region format for OCR
                regions_for_ocr = []
                for element in elements:
                    region = {
                        "region_type": element.get("element_type"),
                        "element_type": element.get("element_type"),
                        "confidence": element.get("confidence"),
                        "bbox": element.get("bbox"),
                        "crop_path": element.get("crop_path")
                    }
                    regions_for_ocr.append(region)
                
                if regions_for_ocr:
                    # Perform OCR on elements
                    enhanced_elements = await self.ocr_service.perform_ocr_on_regions(
                        regions_for_ocr, 
                        element_prompts
                    )
                    
                    # Update ID result with OCR
                    id_result["elements"] = enhanced_elements
                    id_result["ocr_enabled"] = True
                    
                    # Add OCR summary
                    total_elements = len(enhanced_elements)
                    successful_ocr = sum(
                        1 for e in enhanced_elements 
                        if e.get('ocr_result', {}).get('success', False)
                    )
                    
                    id_result["ocr_summary"] = {
                        "total_elements": total_elements,
                        "successful_ocr": successful_ocr,
                        "failed_ocr": total_elements - successful_ocr,
                        "ocr_available": self.ocr_available
                    }
                else:
                    id_result["ocr_enabled"] = False
                    id_result["ocr_summary"] = {
                        "message": "No elements detected for OCR"
                    }
                    
            except Exception as e:
                logger.error(f"Error performing OCR on ID elements: {str(e)}")
                id_result["ocr_enabled"] = False
                id_result["ocr_summary"] = {
                    "error": str(e),
                    "message": "OCR processing failed"
                }
        else:
            id_result["ocr_enabled"] = False
            if not self.ocr_available:
                id_result["ocr_summary"] = {
                    "message": "OCR service not available"
                }
        
        # Update processing time
        processing_time_ms = (time.time() - start_time) * 1000
        id_result["processing_time_ms"] = processing_time_ms
        
        return id_result
    
    async def process_batch_with_ocr(
        self, 
        file_paths: List[Path],
        enable_ocr: bool = False,
        ocr_prompt: Optional[str] = None,
        element_prompts: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple National ID files with optional OCR"""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file_with_ocr(
                    file_path, 
                    enable_ocr, 
                    ocr_prompt, 
                    element_prompts
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path} with OCR: {str(e)}")
                results.append({
                    "filename": Path(file_path).name,
                    "status": "error",
                    "message": f"Error processing file: {str(e)}",
                    "elements": [],
                    "ocr_enabled": False
                })
                
        return results