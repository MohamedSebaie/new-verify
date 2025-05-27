from ultralytics import YOLO # type: ignore
from PIL import Image
import pdf2image # type: ignore
from pathlib import Path
import logging
from typing import List, Dict, Any
from ..core.config import settings
from ..models import VerificationResult, Detection

logger = logging.getLogger(__name__)

class VerificationService:
    def __init__(self):
        # Load YOLO model
        try:
            self.model = YOLO(settings.YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise RuntimeError("Failed to initialize verification service")
            
        # Supported file types
        self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
        self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)
        
        # Class mapping for required elements
        self.required_elements = {"Store_Stamp", "Agent_Signature", "CS_Signature"}
        
        # Custom confidence thresholds
        self.upper_part_ratio = 0.4  # Upper 40% of the image
        self.upper_conf_threshold = 0.7  # Higher threshold for upper region
        self.default_conf_threshold = 0.6  # Default threshold for other regions

    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image and return detections with region-specific thresholds."""
        try:
            # Calculate upper part height
            height = image.height
            upper_part_height = int(height * self.upper_part_ratio)
            
            # Run YOLO detection
            results = self.model.predict(source=image, conf=self.default_conf_threshold)
            
            # Extract detections with special handling for upper part
            detections = []
            for r in results:
                for box in r.boxes:
                    class_name = self.model.names[int(box.cls)]
                    confidence = float(box.conf.item())
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    
                    # Skip detections in upper part that don't meet the higher threshold
                    y1 = bbox[1]  # Top y-coordinate of the detection box
                    if y1 < upper_part_height and confidence < self.upper_conf_threshold:
                        continue
                    
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox
                    ))
            
            # Check if all required elements are present
            detected_classes = {d.class_name for d in detections}
            all_elements_present = self.required_elements.issubset(detected_classes)
            
            return {
                "status": "success" if all_elements_present else "rejected",
                "message": "All required elements found" if all_elements_present else "Missing required elements",
                "detections": detections
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) and return verification result."""
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()
        
        try:
            if file_type in self.supported_images:
                image = Image.open(file_path).convert('RGB')
                result = await self.process_image(image)
                result["file_type"] = "image"
                return result
            elif file_type in self.supported_docs:
                # Convert PDF to images
                images = pdf2image.convert_from_path(str(file_path))
                
                if not images:
                    return {
                        "status": "error",
                        "message": "Could not extract images from PDF",
                        "detections": [],
                        "file_type": "pdf"
                    }
                    
                # Process first page only
                result = await self.process_image(images[0])
                result["total_pages"] = len(images)
                result["processed_page"] = 1
                result["file_type"] = "pdf"
                return result
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"File processing failed: {str(e)}")

    async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files and return results."""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file(file_path)
                result["filename"] = file_path.name
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "filename": file_path.name,
                    "status": "error",
                    "message": str(e),
                    "detections": [],
                    "file_type": file_path.suffix.lower()
                })
                
        return results