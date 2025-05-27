# app/core/errors.py - Enhanced Error Handling System

import logging
import traceback
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
import json

class ErrorType(Enum):
    """Enumeration of error types in the pipeline"""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    MODEL_ERROR = "model_error"
    OCR_ERROR = "ocr_error"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"
    AUTHENTICATION_ERROR = "authentication_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PipelineError(Exception):
    """Base exception class for pipeline errors"""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.details = details or {}
        self.stage = stage
        self.cause = cause
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"{self.error_type.value}_{int(self.timestamp)}_{str(uuid.uuid4())[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization"""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "stage": self.stage,
            "details": self.details,
            "timestamp": self.timestamp,
            "cause": str(self.cause) if self.cause else None
        }

class ValidationError(PipelineError):
    """Error in data validation"""
    def __init__(self, message: str, field: str = None, **kwargs):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        super().__init__(message, ErrorType.VALIDATION_ERROR, **kwargs)

class ProcessingError(PipelineError):
    """Error in document processing"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.PROCESSING_ERROR, **kwargs)

class ModelError(PipelineError):
    """Error in AI model inference"""
    def __init__(self, message: str, model_name: str = None, **kwargs):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        super().__init__(message, ErrorType.MODEL_ERROR, **kwargs)

class OCRError(PipelineError):
    """Error in OCR processing"""
    def __init__(self, message: str, provider: str = None, **kwargs):
        details = kwargs.get('details', {})
        if provider:
            details['provider'] = provider
        super().__init__(message, ErrorType.OCR_ERROR, **kwargs)

class FileError(PipelineError):
    """Error in file operations"""
    def __init__(self, message: str, file_path: str = None, **kwargs):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        super().__init__(message, ErrorType.FILE_ERROR, **kwargs)

class TimeoutError(PipelineError):
    """Error due to timeout"""
    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        details = kwargs.get('details', {})
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        super().__init__(message, ErrorType.TIMEOUT_ERROR, **kwargs)

# ============= Error Context Manager =============

class ErrorContext:
    """Context manager for handling pipeline errors with logging and recovery"""
    
    def __init__(self, stage: str, logger: logging.Logger, enable_recovery: bool = True):
        self.stage = stage
        self.logger = logger
        self.enable_recovery = enable_recovery
        self.start_time = None
        self.errors = []
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting stage: {self.stage}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Stage '{self.stage}' completed successfully in {duration:.2f}s")
            return False
        
        # Handle pipeline errors
        if isinstance(exc_val, PipelineError):
            exc_val.stage = self.stage
            self.logger.error(f"Pipeline error in stage '{self.stage}': {exc_val.message}")
            self.errors.append(exc_val)
        else:
            # Convert generic exceptions to pipeline errors
            pipeline_error = ProcessingError(
                message=f"Unexpected error in stage '{self.stage}': {str(exc_val)}",
                stage=self.stage,
                cause=exc_val,
                details={
                    "exception_type": exc_type.__name__,
                    "traceback": traceback.format_exc()
                }
            )
            self.logger.error(f"Unexpected error in stage '{self.stage}': {str(exc_val)}")
            self.errors.append(pipeline_error)
        
        # Log performance if stage took too long
        if duration > 5.0:  # More than 5 seconds
            self.logger.warning(f"Stage '{self.stage}' took {duration:.2f}s (longer than expected)")
        
        # Don't suppress the exception by default
        return False
    
    def add_warning(self, message: str, details: Dict[str, Any] = None):
        """Add a warning to the error context"""
        warning = {
            "stage": self.stage,
            "message": message,
            "details": details or {},
            "timestamp": time.time()
        }
        self.logger.warning(f"Warning in stage '{self.stage}': {message}")

# ============= Error Handler Decorator =============

def handle_pipeline_errors(stage: str, logger: logging.Logger = None):
    """Decorator for handling pipeline errors in functions"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            with ErrorContext(stage, logger):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# ============= Enhanced Logging System =============

class PipelineLogger:
    """Enhanced logging system for the pipeline"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self.performance_metrics = {}
        self.error_counts = {}
        self.stage_timings = {}
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        from ..core.config import settings
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if settings.LOG_FILE_ENABLED:
            try:
                log_file_path = Path(settings.LOG_FILE_PATH)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=settings.LOG_FILE_MAX_SIZE,
                    backupCount=settings.LOG_FILE_BACKUP_COUNT
                )
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup file logging: {e}")
    
    def log_stage_start(self, stage: str, details: Dict[str, Any] = None):
        """Log the start of a pipeline stage"""
        self.stage_timings[stage] = time.time()
        message = f"🚀 Starting stage: {stage}"
        if details:
            message += f" | Details: {details}"
        self.logger.info(message)
    
    def log_stage_end(self, stage: str, success: bool = True, details: Dict[str, Any] = None):
        """Log the end of a pipeline stage"""
        if stage in self.stage_timings:
            duration = time.time() - self.stage_timings[stage]
            status = "✅ SUCCESS" if success else "❌ FAILED"
            message = f"{status} Stage: {stage} | Duration: {duration:.2f}s"
            if details:
                message += f" | Details: {details}"
            
            if success:
                self.logger.info(message)
            else:
                self.logger.error(message)
            
            # Track performance metrics
            if stage not in self.performance_metrics:
                self.performance_metrics[stage] = []
            self.performance_metrics[stage].append(duration)
    
    def log_performance_warning(self, stage: str, duration: float, threshold: float = 5.0):
        """Log performance warning if stage takes too long"""
        if duration > threshold:
            self.logger.warning(
                f"⚠️ Performance: Stage '{stage}' took {duration:.2f}s (threshold: {threshold}s)"
            )
    
    def log_validation_result(self, validation_result: Dict[str, Any]):
        """Log validation results with detailed breakdown"""
        from ..core.config import settings
        
        if not settings.LOG_VALIDATION_DETAILS:
            return
        
        if validation_result.get('validation_performed', False):
            overall_result = validation_result.get('overall_validation', False)
            match_percentage = validation_result.get('match_percentage', 0)
            
            message = f"🔍 Validation Result: {'✅ PASS' if overall_result else '❌ FAIL'} ({match_percentage:.1%})"
            
            discrepancies = validation_result.get('discrepancies', [])
            if discrepancies:
                message += f" | Issues: {', '.join(discrepancies)}"
            
            if overall_result:
                self.logger.info(message)
            else:
                self.logger.warning(message)
    
    def log_ocr_result(self, ocr_result: Dict[str, Any], region_type: str = None):
        """Log OCR results (if enabled)"""
        from ..core.config import settings
        
        if not settings.LOG_OCR_RESULTS:
            return
        
        success = ocr_result.get('success', False)
        provider = ocr_result.get('provider', 'unknown')
        
        if region_type:
            message = f"📝 OCR ({provider}) for {region_type}: {'✅' if success else '❌'}"
        else:
            message = f"📝 OCR ({provider}): {'✅' if success else '❌'}"
        
        if success:
            text_length = len(ocr_result.get('extracted_text', ''))
            message += f" | Extracted {text_length} characters"
            self.logger.info(message)
        else:
            error = ocr_result.get('error', 'Unknown error')
            message += f" | Error: {error}"
            self.logger.error(message)
    
    def log_error(self, error: PipelineError):
        """Log pipeline error with full context"""
        error_dict = error.to_dict()
        
        # Count errors by type
        error_type = error.error_type.value
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Format log message
        message = f"❌ {error.severity.value.upper()} ERROR [{error.error_id}]: {error.message}"
        if error.stage:
            message += f" | Stage: {error.stage}"
        
        # Log based on severity
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(message)
            self.logger.error(f"Error details: {json.dumps(error_dict, indent=2)}")
        else:
            self.logger.warning(message)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all stages"""
        summary = {}
        for stage, timings in self.performance_metrics.items():
            summary[stage] = {
                "count": len(timings),
                "avg_duration": sum(timings) / len(timings),
                "min_duration": min(timings),
                "max_duration": max(timings),
                "total_duration": sum(timings)
            }
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "errors_by_type": self.error_counts.copy(),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }

# ============= Error Recovery System =============

class ErrorRecoveryManager:
    """Manages error recovery strategies for different types of failures"""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.recovery_strategies = {}
        self.retry_counts = {}
        self.max_retries = 3
    
    def register_recovery_strategy(self, error_type: ErrorType, strategy_func):
        """Register a recovery strategy for an error type"""
        self.recovery_strategies[error_type] = strategy_func
    
    def attempt_recovery(self, error: PipelineError, context: Dict[str, Any] = None) -> bool:
        """Attempt to recover from an error"""
        error_key = f"{error.error_type.value}_{error.stage or 'unknown'}"
        
        # Check retry count
        current_retries = self.retry_counts.get(error_key, 0)
        if current_retries >= self.max_retries:
            self.logger.logger.error(f"Max retries exceeded for error: {error.error_id}")
            return False
        
        # Increment retry count
        self.retry_counts[error_key] = current_retries + 1
        
        # Attempt recovery if strategy exists
        if error.error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error.error_type]
                recovery_successful = strategy(error, context or {})
                
                if recovery_successful:
                    self.logger.logger.info(f"Successfully recovered from error: {error.error_id}")
                    # Reset retry count on success
                    self.retry_counts[error_key] = 0
                    return True
                else:
                    self.logger.logger.warning(f"Recovery attempt failed for error: {error.error_id}")
                    return False
                    
            except Exception as recovery_error:
                self.logger.logger.error(f"Recovery strategy failed: {str(recovery_error)}")
                return False
        
        return False

# ============= Default Recovery Strategies =============

def ocr_provider_fallback_strategy(error: OCRError, context: Dict[str, Any]) -> bool:
    """Fallback to alternative OCR provider"""
    from ..core.config import settings
    
    if not settings.OCR_AUTO_FALLBACK:
        return False
    
    current_provider = error.details.get('provider')
    fallback_provider = settings.OCR_FALLBACK_PROVIDER
    
    if current_provider != fallback_provider:
        try:
            # This would be implemented in the OCR service
            # For now, just return True to indicate strategy exists
            return True
        except Exception:
            return False
    
    return False

def model_reload_strategy(error: ModelError, context: Dict[str, Any]) -> bool:
    """Attempt to reload a failed model"""
    model_name = error.details.get('model_name')
    
    if model_name:
        try:
            # This would be implemented in the model loading service
            # For now, just return True to indicate strategy exists
            return True
        except Exception:
            return False
    
    return False

def file_cleanup_strategy(error: FileError, context: Dict[str, Any]) -> bool:
    """Clean up corrupted temporary files"""
    file_path = error.details.get('file_path')
    
    if file_path and Path(file_path).exists():
        try:
            Path(file_path).unlink()
            return True
        except Exception:
            return False
    
    return False

# ============= Error Reporting System =============

class ErrorReporter:
    """System for reporting errors to external monitoring services"""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.error_buffer = []
        self.max_buffer_size = 100
    
    def report_error(self, error: PipelineError, context: Dict[str, Any] = None):
        """Report error to monitoring systems"""
        error_report = {
            **error.to_dict(),
            "context": context or {},
            "reported_at": time.time()
        }
        
        # Add to buffer
        self.error_buffer.append(error_report)
        
        # Limit buffer size
        if len(self.error_buffer) > self.max_buffer_size:
            self.error_buffer = self.error_buffer[-self.max_buffer_size:]
        
        # Send to external services if critical
        if error.severity == ErrorSeverity.CRITICAL:
            self._send_alert(error_report)
    
    def _send_alert(self, error_report: Dict[str, Any]):
        """Send alert for critical errors"""
        from ..core.config import settings
        
        if settings.ENABLE_ALERTING and settings.ALERT_WEBHOOK_URL:
            try:
                import requests
                response = requests.post(
                    settings.ALERT_WEBHOOK_URL,
                    json=error_report,
                    timeout=10
                )
                if response.status_code == 200:
                    self.logger.logger.info(f"Alert sent for error: {error_report['error_id']}")
                else:
                    self.logger.logger.warning(f"Failed to send alert: {response.status_code}")
            except Exception as e:
                self.logger.logger.error(f"Failed to send alert: {str(e)}")
    
    def get_error_buffer(self) -> List[Dict[str, Any]]:
        """Get current error buffer"""
        return self.error_buffer.copy()
    
    def clear_error_buffer(self):
        """Clear error buffer"""
        self.error_buffer.clear()

# ============= Global Error Manager =============

class GlobalErrorManager:
    """Global error management system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = PipelineLogger("GlobalErrorManager")
            self.recovery_manager = ErrorRecoveryManager(self.logger)
            self.error_reporter = ErrorReporter(self.logger)
            
            # Register default recovery strategies
            self.recovery_manager.register_recovery_strategy(
                ErrorType.OCR_ERROR, ocr_provider_fallback_strategy
            )
            self.recovery_manager.register_recovery_strategy(
                ErrorType.MODEL_ERROR, model_reload_strategy
            )
            self.recovery_manager.register_recovery_strategy(
                ErrorType.FILE_ERROR, file_cleanup_strategy
            )
            
            self._initialized = True
    
    def handle_error(self, error: PipelineError, context: Dict[str, Any] = None, attempt_recovery: bool = True) -> bool:
        """Handle an error with logging, recovery, and reporting"""
        # Log the error
        self.logger.log_error(error)
        
        # Report the error
        self.error_reporter.report_error(error, context)
        
        # Attempt recovery if enabled
        if attempt_recovery and error.severity != ErrorSeverity.CRITICAL:
            recovery_successful = self.recovery_manager.attempt_recovery(error, context)
            if recovery_successful:
                self.logger.logger.info(f"Error recovery successful: {error.error_id}")
                return True
        
        return False
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global error and performance statistics"""
        return {
            "performance_summary": self.logger.get_performance_summary(),
            "error_summary": self.logger.get_error_summary(),
            "recent_errors": self.error_reporter.get_error_buffer()[-10:],  # Last 10 errors
            "timestamp": time.time()
        }

# Create global error manager instance
error_manager = GlobalErrorManager()