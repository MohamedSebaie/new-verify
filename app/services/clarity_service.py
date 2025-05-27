import torch # type: ignore
from transformers import ViTImageProcessor
from PIL import Image
import pdf2image # type: ignore
from pathlib import Path
import logging
from typing import List, Dict, Any
from ..core.config import settings
from ..models import ClarityResult, PDFClarityResult

logger = logging.getLogger(__name__)

class ClarityService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        try:
            self.model = torch.load(settings.VIT_MODEL_PATH, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = ViTImageProcessor.from_pretrained(settings.VIT_PROCESSOR_NAME)
            logger.info("Clarity model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading clarity model: {str(e)}")
            raise RuntimeError("Failed to initialize clarity service")

        self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
        self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)

    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image and assess clarity."""
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)

            pred_class = predictions.argmax().item()
            confidence = predictions[0][pred_class].item()

            return {
                "is_clear": bool(pred_class),
                "confidence": float(confidence),
                "message": f"Document is {'clear' if pred_class else 'unclear'} with {confidence:.2%} confidence"
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")

    async def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF file and assess clarity for all pages."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(str(file_path))
            page_results = []

            # Process each page
            for i, image in enumerate(images, 1):
                result = await self.process_image(image)
                result["page_number"] = i
                page_results.append(result)

            # Calculate overall clarity
            total_pages = len(images)
            clear_pages = sum(1 for r in page_results if r["is_clear"])
            avg_confidence = sum(r["confidence"] for r in page_results) / total_pages if total_pages > 0 else 0
            is_clear = clear_pages > total_pages / 2

            return {
                "overall_result": {
                    "is_clear": is_clear,
                    "confidence": float(avg_confidence),
                    "message": f"Document is {'clear' if is_clear else 'unclear'} with {avg_confidence:.2%} average confidence"
                },
                "page_results": page_results,
                "total_pages": total_pages,
                "clear_pages": clear_pages,
                "unclear_pages": total_pages - clear_pages,
                "file_type": "pdf"
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) and assess clarity."""
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()
        
        try:
            if file_type in self.supported_images:
                image = Image.open(file_path).convert('RGB')
                result = await self.process_image(image)
                result["file_type"] = "image"
                return result
            elif file_type in self.supported_docs:
                return await self.process_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"File processing failed: {str(e)}")

    async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files and return clarity assessment results."""
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
                    "error": str(e),
                    "file_type": file_path.suffix.lower()
                })
                
        return results