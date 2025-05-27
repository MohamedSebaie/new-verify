import torch # type: ignore
from transformers import ViTImageProcessor
from PIL import Image
import pdf2image # type: ignore
from pathlib import Path
import logging
from typing import List, Dict, Any
from ..core.config import settings
from ..models import ClassificationResult, PDFClassificationResult

logger = logging.getLogger(__name__)

class ClassificationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        try:
            self.model = torch.load(settings.DOCUMENT_CLASS_MODEL_PATH, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = ViTImageProcessor.from_pretrained(settings.VIT_PROCESSOR_NAME)
            
            # Define class mappings
            self.id2label = {0: "KYC", 1: "NID", 2: "TC"}
            self.label2id = {"KYC": 0, "NID": 1, "TC": 2}

            # Use model's config if available
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                
            logger.info(f"Classification model loaded with classes: {list(self.id2label.values())}")
        except Exception as e:
            logger.error(f"Error loading classification model: {str(e)}")
            raise RuntimeError("Failed to initialize classification service")

        self.supported_images = set(settings.SUPPORTED_IMAGE_TYPES)
        self.supported_docs = set(settings.SUPPORTED_DOC_TYPES)

    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image for document classification."""
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)

            pred_class_idx = predictions.argmax().item()
            confidence = predictions[0][pred_class_idx].item()
            pred_class_name = self.id2label[pred_class_idx]

            # Get all class probabilities
            all_probs = {self.id2label[i]: float(prob) for i, prob in enumerate(predictions[0].tolist())}

            return {
                'class': pred_class_name,
                'class_index': pred_class_idx,
                'confidence': float(confidence),
                'probabilities': all_probs,
                'message': f"Document classified as {pred_class_name} with {confidence:.2%} confidence"
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Image processing failed: {str(e)}")

    async def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF file for document classification."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(str(file_path))
            results = []

            # Process each page
            for i, image in enumerate(images, 1):
                result = await self.process_image(image)
                result['page_number'] = i
                results.append(result)

            # Calculate majority class
            class_votes = {cls: 0 for cls in self.id2label.values()}
            for r in results:
                class_votes[r['class']] += 1
            
            majority_class = max(class_votes.items(), key=lambda x: x[1])[0]

            # Calculate average confidence
            majority_confidences = [r['confidence'] for r in results if r['class'] == majority_class]
            avg_confidence = sum(majority_confidences) / len(majority_confidences) if majority_confidences else 0

            # Calculate overall probabilities
            overall_probs = {}
            for cls in self.id2label.values():
                cls_probs = [r['probabilities'][cls] for r in results]
                overall_probs[cls] = sum(cls_probs) / len(cls_probs) if cls_probs else 0

            return {
                'overall_result': {
                    'class': majority_class,
                    'confidence': float(avg_confidence),
                    'probabilities': overall_probs,
                    'message': f"Document classified as {majority_class} with {avg_confidence:.2%} average confidence"
                },
                'page_results': results,
                'file_type': 'pdf',
                'total_pages': len(images)
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a file (image or PDF) for document classification."""
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()
        
        try:
            if file_type in self.supported_images:
                image = Image.open(file_path).convert('RGB')
                result = await self.process_image(image)
                result['file_type'] = 'image'
                return result
            elif file_type in self.supported_docs:
                return await self.process_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"File processing failed: {str(e)}")

    async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files for document classification."""
        results = []
        for file_path in file_paths:
            try:
                result = await self.process_file(file_path)
                result['filename'] = file_path.name
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    'filename': file_path.name,
                    'error': str(e),
                    'file_type': file_path.suffix.lower()
                })
                
        return results