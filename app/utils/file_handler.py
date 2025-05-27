import base64
import tempfile
from pathlib import Path
from typing import Optional, Set
from fastapi import UploadFile, HTTPException # type: ignore
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: Set[str]) -> bool:
        """Validate file extension."""
        return Path(filename).suffix.lower() in allowed_extensions

    @staticmethod
    def validate_file_size(file_size: int, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate file size."""
        return file_size <= max_size

    @staticmethod
    async def process_uploaded_file(upload_file: UploadFile, allowed_extensions: Set[str]) -> Path:
        """Process and validate uploaded file."""
        # Validate file type
        if not FileHandler.validate_file_type(upload_file.filename, allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {allowed_extensions}"
            )
        
        # Read file content
        content = await upload_file.read()
        
        # Validate file size
        if not FileHandler.validate_file_size(len(content)):
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {10 * 1024 * 1024} bytes"
            )
        
        # Save to temp file
        temp_dir = Path(tempfile.gettempdir()) / "kyc_temp"
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / upload_file.filename
        with open(file_path, "wb") as f:
            f.write(content)
            
        return file_path

    @staticmethod
    async def process_base64_file(base64_data: dict, allowed_extensions: Set[str]) -> Path:
        """Process base64 encoded file."""
        # Validate required fields
        required_fields = ['filename', 'content', 'file_type']
        for field in required_fields:
            if field not in base64_data:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field: {field}"
                )
        
        # Validate file type
        if not FileHandler.validate_file_type(base64_data['filename'], allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {allowed_extensions}"
            )
        
        try:
            # Clean base64 content
            content = base64_data['content']
            if ',' in content:
                content = content.split(',')[1]
                
            # Decode base64
            file_bytes = base64.b64decode(content)
            
            # Validate size
            if not FileHandler.validate_file_size(len(file_bytes)):
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum limit of {10 * 1024 * 1024} bytes"
                )
                
            # Save to temp file
            temp_dir = Path(tempfile.gettempdir()) / "kyc_temp"
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / base64_data['filename']
            with open(file_path, "wb") as f:
                f.write(file_bytes)
                
            return file_path
        except Exception as e:
            logger.error(f"Error processing base64 file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")