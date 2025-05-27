import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from fastapi import HTTPException
from ..services.classification_service import ClassificationService
from ..services.verification_service import VerificationService
from ..services.detection_service import DetectionService
from ..services.clarity_service import ClarityService
from ..services.id_layout_service import IDLayoutService
from ..services.national_id_service import NationalIDService
from ..services.ocr_service import OCRService

logger = logging.getLogger(__name__)

class ValidationService:
    """Service for cross-validating KYC and NID extracted data"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return text.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        if not text1 or not text2:
            return 0.0
        
        norm1 = ValidationService.normalize_text(text1)
        norm2 = ValidationService.normalize_text(text2)
        
        if norm1 == norm2:
            return 1.0
        
        # Simple character-based similarity
        if len(norm1) == 0 or len(norm2) == 0:
            return 0.0
            
        common_chars = sum(1 for c1, c2 in zip(norm1, norm2) if c1 == c2)
        max_len = max(len(norm1), len(norm2))
        
        return common_chars / max_len
    
    @staticmethod
    def is_match(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts match based on similarity threshold"""
        similarity = ValidationService.calculate_similarity(text1, text2)
        return similarity >= threshold
    
    @staticmethod
    def validate_kyc_nid_data(kyc_data: Dict[str, Any], nid_data: Dict[str, Any], tc_available: bool = False) -> Dict[str, Any]:
        """Validate KYC data against NID data with TC context"""
        validation_results = {
            "validation_performed": True,
            "overall_validation": False,
            "field_validations": {},
            "discrepancies": [],
            "similarity_scores": {},
            "tc_context": {
                "tc_available": tc_available,
                "tc_note": "TC document is optional for KYC-NID validation" if tc_available else "TC document not available"
            }
        }
        
        # Extract KYC fields (from detection OCR results)
        kyc_fields = {}
        if "extracted_fields" in kyc_data:
            kyc_fields = kyc_data["extracted_fields"]
        
        # Extract NID fields
        nid_front_fields = nid_data.get("extracted_data", {}).get("front_fields", {})
        nid_back_fields = nid_data.get("extracted_data", {}).get("back_fields", {})
        
        # Field mapping for validation
        field_mappings = [
            ("customer_name", "name", "Name"),
            ("national_id", "id_number", "National ID"),
            ("customer_address", "address", "Address"),
            ("date_of_birth", "date_of_birth", "Date of Birth")
        ]
        
        matched_fields = 0
        total_fields = 0
        
        for kyc_field, nid_field, display_name in field_mappings:
            kyc_value = kyc_fields.get(kyc_field, "")
            nid_value = nid_front_fields.get(nid_field) or nid_back_fields.get(nid_field, "")
            
            if kyc_value or nid_value:
                total_fields += 1
                similarity = ValidationService.calculate_similarity(kyc_value, nid_value)
                is_valid = ValidationService.is_match(kyc_value, nid_value)
                
                validation_results["field_validations"][display_name] = {
                    "kyc_value": kyc_value,
                    "nid_value": nid_value,
                    "is_match": is_valid,
                    "similarity_score": similarity
                }
                
                validation_results["similarity_scores"][display_name] = similarity
                
                if is_valid:
                    matched_fields += 1
                else:
                    validation_results["discrepancies"].append(display_name)
        
        # Overall validation (at least 70% of fields should match)
        if total_fields > 0:
            match_percentage = matched_fields / total_fields
            validation_results["overall_validation"] = match_percentage >= 0.7
            validation_results["match_percentage"] = match_percentage
        
        # Add contextual information
        if tc_available:
            validation_results["validation_note"] = f"KYC-NID validation completed successfully. TC document was available but not required for validation."
        else:
            validation_results["validation_note"] = f"KYC-NID validation completed. TC document not provided (TC is optional for validation)."
        
        return validation_results


class MasterPipelineService:
    """Master service that orchestrates the complete document processing pipeline"""
    
    def __init__(self):
        # Initialize all services
        self.classification_service = ClassificationService()
        self.verification_service = VerificationService()
        self.detection_service = DetectionService()
        self.clarity_service = ClarityService()
        self.id_layout_service = IDLayoutService()
        self.national_id_service = NationalIDService()
        self.validation_service = ValidationService()
        
        # Initialize OCR service if available
        try:
            self.ocr_service = OCRService()
            self.ocr_available = True
        except Exception as e:
            logger.warning(f"OCR service not available: {str(e)}")
            self.ocr_service = None
            self.ocr_available = False
        
        logger.info("Master Pipeline Service initialized successfully")
    
    async def process_kyc_pipeline(self, file_path: Path) -> Dict[str, Any]:
        """Process KYC document through complete pipeline"""
        logger.info(f"Processing KYC pipeline for: {file_path}")
        pipeline_result = {
            "document_type": "KYC",
            "pipeline_stages": {},
            "final_result": {}
        }
        
        try:
            # Stage 1: Verification (Check signatures and stamps)
            logger.info("KYC Stage 1: Signature Verification")
            verification_result = await self.verification_service.process_file(file_path)
            pipeline_result["pipeline_stages"]["verification"] = verification_result
            
            # Check if all required signatures exist
            detections = verification_result.get("detections", [])
            detected_classes = {d.class_name for d in detections}
            required_elements = {"Store_Stamp", "Agent_Signature", "CS_Signature"}
            
            signature_verification = {
                "agent_signature": "Agent_Signature" in detected_classes,
                "customer_signature": "CS_Signature" in detected_classes,
                "store_stamp": "Store_Stamp" in detected_classes,
                "all_signatures_verified": required_elements.issubset(detected_classes)
            }
            
            # Stage 2: Detection (Layout detection for KYC fields)
            logger.info("KYC Stage 2: Field Detection")
            detection_result = await self.detection_service.process_file(file_path)
            pipeline_result["pipeline_stages"]["field_detection"] = detection_result
            
            # Stage 3: OCR (Extract text from detected fields)
            extracted_fields = {}
            if self.ocr_available:
                logger.info("KYC Stage 3: OCR Text Extraction")
                ocr_result = await self.detection_service.process_file_with_ocr(
                    file_path,
                    enable_ocr=True,
                    clean_text=True
                )
                pipeline_result["pipeline_stages"]["ocr"] = ocr_result
                
                # Extract field texts from OCR results
                extracted_fields = self._extract_kyc_fields_from_ocr(ocr_result)
            else:
                logger.warning("OCR not available for KYC text extraction")
            
            # Compile final KYC result
            pipeline_result["final_result"] = {
                "status": "success",
                "signature_verification": signature_verification,
                "extracted_fields": extracted_fields,
                "processing_summary": {
                    "signatures_found": len(detected_classes.intersection(required_elements)),
                    "total_signatures_required": len(required_elements),
                    "fields_extracted": len(extracted_fields),
                    "ocr_available": self.ocr_available
                }
            }
            
            logger.info("KYC pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error in KYC pipeline: {str(e)}")
            pipeline_result["final_result"] = {
                "status": "error",
                "error": str(e),
                "signature_verification": {},
                "extracted_fields": {}
            }
            return pipeline_result
    
    async def process_tc_pipeline(self, file_path: Path) -> Dict[str, Any]:
        """Process TC (Trade Certificate) document through clarity check"""
        logger.info(f"Processing TC pipeline for: {file_path}")
        pipeline_result = {
            "document_type": "TC",
            "pipeline_stages": {},
            "final_result": {}
        }
        
        try:
            # Single Stage: Clarity Assessment
            logger.info("TC Stage 1: Clarity Assessment")
            clarity_result = await self.clarity_service.process_file(file_path)
            pipeline_result["pipeline_stages"]["clarity"] = clarity_result
            
            # Compile final TC result
            pipeline_result["final_result"] = {
                "status": "success",
                "is_clear": clarity_result.get("is_clear", False),
                "confidence": clarity_result.get("confidence", 0.0),
                "message": clarity_result.get("message", ""),
                "file_type": clarity_result.get("file_type", "unknown")
            }
            
            logger.info("TC pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error in TC pipeline: {str(e)}")
            pipeline_result["final_result"] = {
                "status": "error",
                "error": str(e),
                "is_clear": False,
                "confidence": 0.0
            }
            return pipeline_result
    
    async def process_nid_pipeline(self, file_path: Path) -> Dict[str, Any]:
        """Process NID (National ID) document through complete pipeline"""
        logger.info(f"Processing NID pipeline for: {file_path}")
        pipeline_result = {
            "document_type": "NID",
            "pipeline_stages": {},
            "final_result": {}
        }
        
        try:
            # Stage 1: ID Layout Detection (Check for ID front, back, signature, stamp)
            logger.info("NID Stage 1: ID Layout Detection")
            layout_result = await self.id_layout_service.process_file(file_path)
            pipeline_result["pipeline_stages"]["id_layout"] = layout_result
            
            # Check which elements are detected
            regions = layout_result.get("regions", [])
            detected_regions = {r.get("region_type") for r in regions}
            
            elements_detected = {
                "id_front": any("front" in r.lower() for r in detected_regions),
                "id_back": any("back" in r.lower() for r in detected_regions),
                "signature": any("signature" in r.lower() for r in detected_regions),
                "stamp": any("stamp" in r.lower() for r in detected_regions)
            }
            
            # Stage 2: NID Element Detection (Detailed segmentation)
            logger.info("NID Stage 2: NID Element Detection")
            nid_result = await self.national_id_service.process_file(file_path)
            pipeline_result["pipeline_stages"]["nid_detection"] = nid_result
            
            # Stage 3: OCR (Extract text from segmented fields)
            extracted_data = {"front_fields": {}, "back_fields": {}}
            if self.ocr_available:
                logger.info("NID Stage 3: OCR Text Extraction")
                ocr_result = await self.national_id_service.process_file_with_ocr(
                    file_path,
                    enable_ocr=True,
                    clean_text=True
                )
                pipeline_result["pipeline_stages"]["ocr"] = ocr_result
                
                # Extract field texts from OCR results
                extracted_data = self._extract_nid_fields_from_ocr(ocr_result)
            else:
                logger.warning("OCR not available for NID text extraction")
            
            # Compile final NID result
            pipeline_result["final_result"] = {
                "status": "success",
                "elements_detected": elements_detected,
                "extracted_data": extracted_data,
                "processing_summary": {
                    "layout_regions_found": len(regions),
                    "nid_elements_found": len(nid_result.get("elements", [])),
                    "front_fields_extracted": len(extracted_data["front_fields"]),
                    "back_fields_extracted": len(extracted_data["back_fields"]),
                    "ocr_available": self.ocr_available
                }
            }
            
            logger.info("NID pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error in NID pipeline: {str(e)}")
            pipeline_result["final_result"] = {
                "status": "error",
                "error": str(e),
                "elements_detected": {},
                "extracted_data": {"front_fields": {}, "back_fields": {}}
            }
            return pipeline_result
    
    def _extract_kyc_fields_from_ocr(self, ocr_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract KYC fields from OCR results"""
        extracted_fields = {}
        
        # Navigate through OCR results structure
        results = ocr_result.get("results", [])
        for result in results:
            detections = result.get("detections", [])
            for detection in detections:
                class_name = detection.get("class_name", "")
                ocr_data = detection.get("ocr_result", {})
                
                if ocr_data.get("success", False):
                    extracted_text = ocr_data.get("extracted_text", "")
                    if extracted_text:
                        # Map detection class to field name
                        field_mapping = {
                            "customer_name": "customer_name",
                            "national_id": "national_id",
                            "address": "customer_address",
                            "phone": "phone_number",
                            "date": "date_of_birth"
                        }
                        
                        field_name = field_mapping.get(class_name.lower(), class_name)
                        extracted_fields[field_name] = extracted_text
        
        return extracted_fields
    
    def _extract_nid_fields_from_ocr(self, ocr_result: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Extract NID fields from OCR results"""
        extracted_data = {"front_fields": {}, "back_fields": {}}
        
        elements = ocr_result.get("elements", [])
        for element in elements:
            element_type = element.get("element_type", "")
            ocr_data = element.get("ocr_result", {})
            
            if ocr_data.get("success", False):
                extracted_text = ocr_data.get("extracted_text", "")
                if extracted_text:
                    # Classify field as front or back
                    front_fields = {"Name", "Date_of_Birth", "ID_Number", "Photo"}
                    back_fields = {"Address", "Signature"}
                    
                    if element_type in front_fields:
                        field_name = element_type.lower()
                        extracted_data["front_fields"][field_name] = extracted_text
                    elif element_type in back_fields:
                        field_name = element_type.lower()
                        extracted_data["back_fields"][field_name] = extracted_text
                    else:
                        # Default to front if unclear
                        extracted_data["front_fields"][element_type.lower()] = extracted_text
        
        return extracted_data
    
    async def process_master_pipeline(
        self, 
        file_path: Path, 
        enable_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Master pipeline that processes any document type through appropriate sub-pipeline
        """
        start_time = time.time()
        logger.info(f"Starting master pipeline for: {file_path}")
        
        master_result = {
            "filename": file_path.name,
            "processing_timestamp": time.time(),
            "classification": {},
            "pipeline_results": {},
            "validation_results": {},
            "processing_summary": {}
        }
        
        try:
            # Step 1: Document Classification
            logger.info("Master Pipeline Step 1: Document Classification")
            classification_result = await self.classification_service.process_file(file_path)
            master_result["classification"] = classification_result
            
            document_type = classification_result.get("class", "unknown").upper()
            confidence = classification_result.get("confidence", 0.0)
            
            logger.info(f"Document classified as: {document_type} (confidence: {confidence:.2%})")
            
            # Step 2: Route to appropriate pipeline
            if document_type == "KYC":
                pipeline_result = await self.process_kyc_pipeline(file_path)
            elif document_type == "TC":
                pipeline_result = await self.process_tc_pipeline(file_path)
            elif document_type == "NID":
                pipeline_result = await self.process_nid_pipeline(file_path)
            else:
                raise ValueError(f"Unknown document type: {document_type}")
            
            master_result["pipeline_results"][document_type] = pipeline_result
            
            # Step 3: Cross-validation (if applicable)
            if enable_validation and len(master_result["pipeline_results"]) > 1:
                logger.info("Master Pipeline Step 3: Cross-validation")
                # This would be used when processing multiple documents together
                # For now, single document processing doesn't trigger validation
                master_result["validation_results"] = {
                    "validation_performed": False,
                    "reason": "Single document processed - validation requires both KYC and NID"
                }
            else:
                master_result["validation_results"] = {
                    "validation_performed": False,
                    "reason": "Validation disabled or insufficient documents"
                }
            
            # Processing Summary
            processing_time = time.time() - start_time
            master_result["processing_summary"] = {
                "total_processing_time_ms": processing_time * 1000,
                "document_type": document_type,
                "classification_confidence": confidence,
                "pipeline_success": pipeline_result["final_result"].get("status") == "success",
                "ocr_enabled": self.ocr_available,
                "validation_performed": master_result["validation_results"]["validation_performed"]
            }
            
            logger.info(f"Master pipeline completed successfully in {processing_time:.2f}s")
            return master_result
            
        except Exception as e:
            logger.error(f"Error in master pipeline: {str(e)}")
            master_result["error"] = str(e)
            master_result["processing_summary"] = {
                "total_processing_time_ms": (time.time() - start_time) * 1000,
                "pipeline_success": False,
                "error": str(e)
            }
            return master_result
    
    async def process_three_document_set(
        self, 
        file_paths: List[Path], 
        enable_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Process 3-document set (KYC, TC, NID) with validation logic
        """
        start_time = time.time()
        logger.info(f"Starting 3-document set processing for {len(file_paths)} documents")
        
        multi_result = {
            "total_documents": len(file_paths),
            "processing_timestamp": time.time(),
            "document_results": {},
            "cross_validation": {},
            "validation_warnings": [],
            "processing_summary": {}
        }
        
        try:
            # Step 1: Classify all documents first
            document_types = {}
            for file_path in file_paths:
                classification_result = await self.classification_service.process_file(file_path)
                doc_type = classification_result.get("class", "unknown").upper()
                document_types[file_path] = doc_type
                logger.info(f"Document {file_path.name} classified as: {doc_type}")
            
            # Step 2: Check what documents we have
            available_types = set(document_types.values())
            has_kyc = "KYC" in available_types
            has_nid = "NID" in available_types  
            has_tc = "TC" in available_types
            
            logger.info(f"Available document types: {available_types}")
            
            # Step 3: Process each document through its pipeline
            for file_path in file_paths:
                doc_type = document_types[file_path]
                result = await self.process_master_pipeline(file_path, enable_validation=False)
                multi_result["document_results"][f"{doc_type}_{file_path.name}"] = result
            
            # Step 4: Validation logic for 3-document set
            if enable_validation and len(file_paths) == 3:
                # Check for missing documents
                missing_docs = []
                if not has_kyc:
                    missing_docs.append("KYC")
                if not has_nid:
                    missing_docs.append("NID")
                if not has_tc:
                    missing_docs.append("TC")
                
                # Add warnings for missing documents
                if missing_docs:
                    warning_msg = f"Missing documents in 3-document set: {', '.join(missing_docs)}"
                    multi_result["validation_warnings"].append(warning_msg)
                    logger.warning(warning_msg)
                
                # Validation decision logic
                if has_kyc and has_nid:
                    # Can validate KYC with NID (TC is optional)
                    kyc_result = None
                    nid_result = None
                    
                    # Find KYC and NID results
                    for doc_key, doc_result in multi_result["document_results"].items():
                        if doc_key.startswith("KYC_"):
                            kyc_pipeline = doc_result.get("pipeline_results", {}).get("KYC", {})
                            kyc_result = kyc_pipeline.get("final_result")
                        elif doc_key.startswith("NID_"):
                            nid_pipeline = doc_result.get("pipeline_results", {}).get("NID", {})
                            nid_result = nid_pipeline.get("final_result")
                    
                    if kyc_result and nid_result:
                        logger.info("Performing KYC-NID cross-validation (TC optional)")
                        validation_result = self.validation_service.validate_kyc_nid_data(
                            kyc_result, nid_result, tc_available=has_tc
                        )
                        validation_result["tc_available"] = has_tc
                        if not has_tc:
                            validation_result["validation_notes"] = "Validation performed without TC document (TC is optional for KYC-NID validation)"
                        
                        multi_result["cross_validation"] = validation_result
                    else:
                        multi_result["cross_validation"] = {
                            "validation_performed": False,
                            "reason": "Failed to extract data from KYC or NID documents"
                        }
                
                elif has_kyc and not has_nid:
                    # Cannot validate without NID
                    multi_result["cross_validation"] = {
                        "validation_performed": False,
                        "reason": "NID document missing - cannot validate KYC without NID",
                        "available_documents": ["KYC"] + (["TC"] if has_tc else [])
                    }
                    multi_result["validation_warnings"].append("Validation skipped: NID document required for KYC validation")
                
                elif has_nid and not has_kyc:
                    # Cannot validate without KYC
                    multi_result["cross_validation"] = {
                        "validation_performed": False,
                        "reason": "KYC document missing - cannot validate NID without KYC",
                        "available_documents": ["NID"] + (["TC"] if has_tc else [])
                    }
                    multi_result["validation_warnings"].append("Validation skipped: KYC document required for NID validation")
                
                else:
                    # Neither KYC nor NID available
                    multi_result["cross_validation"] = {
                        "validation_performed": False,
                        "reason": "Both KYC and NID documents missing - validation not possible",
                        "available_documents": ["TC"] if has_tc else []
                    }
                    multi_result["validation_warnings"].append("Validation skipped: Both KYC and NID documents required")
            
            else:
                # Not a 3-document set or validation disabled
                multi_result["cross_validation"] = {
                    "validation_performed": False,
                    "reason": f"Validation only available for 3-document sets (received {len(file_paths)} documents)"
                }
            
            # Processing Summary
            processing_time = time.time() - start_time
            successful_docs = sum(1 for r in multi_result["document_results"].values() 
                                if r.get("processing_summary", {}).get("pipeline_success", False))
            
            multi_result["processing_summary"] = {
                "total_processing_time_ms": processing_time * 1000,
                "successful_documents": successful_docs,
                "failed_documents": len(file_paths) - successful_docs,
                "validation_performed": multi_result["cross_validation"].get("validation_performed", False),
                "validation_warnings_count": len(multi_result["validation_warnings"]),
                "document_types_detected": list(available_types),
                "is_three_document_set": len(file_paths) == 3,
                "overall_success": successful_docs == len(file_paths)
            }
            
            logger.info(f"3-document set processing completed in {processing_time:.2f}s")
            return multi_result
            
        except Exception as e:
            logger.error(f"Error in 3-document set processing: {str(e)}")
            multi_result["error"] = str(e)
            multi_result["processing_summary"] = {
                "total_processing_time_ms": (time.time() - start_time) * 1000,
                "overall_success": False,
                "error": str(e)
            }
            return multi_result

    async def process_multi_document_pipeline(
        self, 
        file_paths: List[Path], 
        enable_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents with 3-document set validation logic
        """
        # Check if this is a 3-document set for validation
        if len(file_paths) == 3 and enable_validation:
            return await self.process_three_document_set(file_paths, enable_validation)
        
        # Original multi-document processing for other cases
        start_time = time.time()
        logger.info(f"Starting multi-document pipeline for {len(file_paths)} documents")
        
        multi_result = {
            "total_documents": len(file_paths),
            "processing_timestamp": time.time(),
            "document_results": {},
            "cross_validation": {},
            "processing_summary": {}
        }
        
        try:
            # Process each document individually
            for file_path in file_paths:
                result = await self.process_master_pipeline(file_path, enable_validation=False)
                doc_type = result.get("processing_summary", {}).get("document_type", "unknown")
                multi_result["document_results"][f"{doc_type}_{file_path.name}"] = result
            
            # Only perform validation if exactly 2 documents (KYC + NID)
            if enable_validation and len(file_paths) == 2:
                kyc_result = None
                nid_result = None
                
                # Find KYC and NID results
                for doc_key, doc_result in multi_result["document_results"].items():
                    doc_type = doc_result.get("processing_summary", {}).get("document_type")
                    if doc_type == "KYC" and "KYC" in doc_result.get("pipeline_results", {}):
                        kyc_result = doc_result["pipeline_results"]["KYC"]["final_result"]
                    elif doc_type == "NID" and "NID" in doc_result.get("pipeline_results", {}):
                        nid_result = doc_result["pipeline_results"]["NID"]["final_result"]
                
                # Perform validation if both documents are present
                if kyc_result and nid_result:
                    logger.info("Performing cross-validation between KYC and NID")
                    validation_result = self.validation_service.validate_kyc_nid_data(
                        kyc_result, nid_result, tc_available=False
                    )
                    multi_result["cross_validation"] = validation_result
                else:
                    multi_result["cross_validation"] = {
                        "validation_performed": False,
                        "reason": "Both KYC and NID documents required for validation"
                    }
            else:
                multi_result["cross_validation"] = {
                    "validation_performed": False,
                    "reason": f"Validation available for 2-document (KYC+NID) or 3-document sets only (received {len(file_paths)} documents)"
                }
            
            # Processing Summary
            processing_time = time.time() - start_time
            successful_docs = sum(1 for r in multi_result["document_results"].values() 
                                if r.get("processing_summary", {}).get("pipeline_success", False))
            
            multi_result["processing_summary"] = {
                "total_processing_time_ms": processing_time * 1000,
                "successful_documents": successful_docs,
                "failed_documents": len(file_paths) - successful_docs,
                "validation_performed": multi_result["cross_validation"].get("validation_performed", False),
                "overall_success": successful_docs == len(file_paths)
            }
            
            logger.info(f"Multi-document pipeline completed in {processing_time:.2f}s")
            return multi_result
            
        except Exception as e:
            logger.error(f"Error in multi-document pipeline: {str(e)}")
            multi_result["error"] = str(e)
            multi_result["processing_summary"] = {
                "total_processing_time_ms": (time.time() - start_time) * 1000,
                "overall_success": False,
                "error": str(e)
            }
            return multi_result