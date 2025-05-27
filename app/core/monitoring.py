# app/core/monitoring.py - Monitoring and Metrics System

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ============= Metrics Definitions =============

# Prometheus metrics
PIPELINE_REQUESTS_TOTAL = Counter(
    'pipeline_requests_total',
    'Total number of pipeline requests',
    ['document_type', 'endpoint', 'status']
)

PIPELINE_PROCESSING_TIME = Histogram(
    'pipeline_processing_seconds',
    'Time spent processing documents',
    ['document_type', 'stage'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

PIPELINE_ACTIVE_REQUESTS = Gauge(
    'pipeline_active_requests',
    'Number of currently active pipeline requests'
)

PIPELINE_ERRORS_TOTAL = Counter(
    'pipeline_errors_total',
    'Total number of pipeline errors',
    ['error_type', 'stage', 'severity']
)

OCR_REQUESTS_TOTAL = Counter(
    'ocr_requests_total',
    'Total number of OCR requests',
    ['provider', 'status']
)

OCR_PROCESSING_TIME = Histogram(
    'ocr_processing_seconds',
    'Time spent on OCR processing',
    ['provider'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

VALIDATION_RESULTS = Counter(
    'validation_results_total',
    'Validation results',
    ['result', 'field_type']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time',
    ['model_name', 'model_type'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# ============= Metrics Data Classes =============

@dataclass
class ProcessingMetrics:
    """Container for processing metrics"""
    document_type: str
    total_time: float
    stage_times: Dict[str, float]
    success: bool
    error_type: Optional[str] = None
    validation_result: Optional[bool] = None
    ocr_enabled: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SystemMetrics:
    """Container for system metrics"""
    memory_usage_mb: float
    cpu_usage_percent: float
    active_requests: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

# ============= Metrics Collector =============

class MetricsCollector:
    """Collect and manage application metrics"""
    
    def __init__(self):
        self.processing_metrics = deque(maxlen=10000)  # Keep last 10k metrics
        self.system_metrics = deque(maxlen=1000)       # Keep last 1k system metrics
        self.error_counts = defaultdict(int)
        self.validation_stats = defaultdict(int)
        self.performance_stats = defaultdict(list)
        self._lock = threading.Lock()
        self.monitoring_active = True
        
        # Start system monitoring
        self._start_system_monitoring()
    
    def record_processing_metrics(self, metrics: ProcessingMetrics):
        """Record processing metrics"""
        with self._lock:
            self.processing_metrics.append(metrics)
            
            # Update Prometheus metrics
            status = "success" if metrics.success else "error"
            PIPELINE_REQUESTS_TOTAL.labels(
                document_type=metrics.document_type,
                endpoint="pipeline",
                status=status
            ).inc()
            
            PIPELINE_PROCESSING_TIME.labels(
                document_type=metrics.document_type,
                stage="total"
            ).observe(metrics.total_time)
            
            # Record stage times
            for stage, time_taken in metrics.stage_times.items():
                PIPELINE_PROCESSING_TIME.labels(
                    document_type=metrics.document_type,
                    stage=stage
                ).observe(time_taken)
            
            # Record errors
            if not metrics.success and metrics.error_type:
                self.error_counts[metrics.error_type] += 1
                PIPELINE_ERRORS_TOTAL.labels(
                    error_type=metrics.error_type,
                    stage="pipeline",
                    severity="high"
                ).inc()
            
            # Record validation results
            if metrics.validation_result is not None:
                result = "pass" if metrics.validation_result else "fail"
                self.validation_stats[result] += 1
                VALIDATION_RESULTS.labels(
                    result=result,
                    field_type="overall"
                ).inc()
    
    def record_ocr_metrics(self, provider: str, processing_time: float, success: bool):
        """Record OCR-specific metrics"""
        status = "success" if success else "error"
        OCR_REQUESTS_TOTAL.labels(provider=provider, status=status).inc()
        OCR_PROCESSING_TIME.labels(provider=provider).observe(processing_time)
    
    def record_model_metrics(self, model_name: str, model_type: str, inference_time: float):
        """Record model inference metrics"""
        MODEL_INFERENCE_TIME.labels(
            model_name=model_name,
            model_type=model_type
        ).observe(inference_time)
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics"""
        with self._lock:
            self.system_metrics.append(metrics)
            
            # Update Prometheus metrics
            SYSTEM_MEMORY_USAGE.set(metrics.memory_usage_mb * 1024 * 1024)  # Convert to bytes
            SYSTEM_CPU_USAGE.set(metrics.cpu_usage_percent)
            PIPELINE_ACTIVE_REQUESTS.set(metrics.active_requests)
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.processing_metrics
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics available for the specified time range"}
        
        # Calculate statistics
        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.success)
        error_rate = (total_requests - successful_requests) / total_requests * 100
        
        processing_times = [m.total_time for m in recent_metrics]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        document_type_counts = defaultdict(int)
        for m in recent_metrics:
            document_type_counts[m.document_type] += 1
        
        return {
            "time_range_hours": hours,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate_percent": round(error_rate, 2),
            "avg_processing_time_seconds": round(avg_processing_time, 2),
            "document_type_distribution": dict(document_type_counts),
            "top_errors": dict(list(self.error_counts.items())[:5]),
            "validation_stats": dict(self.validation_stats)
        }
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while self.monitoring_active:
                try:
                    # Get system metrics
                    memory_mb = psutil.virtual_memory().used / 1024 / 1024
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # Count active requests (simplified)
                    active_requests = len([
                        m for m in self.processing_metrics
                        if (datetime.utcnow() - m.timestamp).seconds < 60
                    ])
                    
                    metrics = SystemMetrics(
                        memory_usage_mb=memory_mb,
                        cpu_usage_percent=cpu_percent,
                        active_requests=active_requests
                    )
                    
                    self.record_system_metrics(metrics)
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                
                time.sleep(30)  # Monitor every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False

# Global metrics collector
metrics_collector = MetricsCollector()

# ============= Monitoring Context Manager =============

@asynccontextmanager
async def monitor_pipeline_request(document_type: str, endpoint: str = "pipeline"):
    """Context manager to monitor pipeline requests"""
    start_time = time.time()
    stage_times = {}
    stage_start = None
    success = True
    error_type = None
    validation_result = None
    
    # Increment active requests
    PIPELINE_ACTIVE_REQUESTS.inc()
    
    try:
        yield {
            'start_stage': lambda stage: stage_times.update({f"{stage}_start": time.time()}),
            'end_stage': lambda stage: stage_times.update({
                stage: time.time() - stage_times.get(f"{stage}_start", start_time)
            }),
            'set_validation_result': lambda result: locals().update(validation_result=result)
        }
        
    except Exception as e:
        success = False
        error_type = type(e).__name__
        logger.error(f"Pipeline monitoring error: {e}")
        raise
    
    finally:
        # Decrement active requests
        PIPELINE_ACTIVE_REQUESTS.dec()
        
        # Record metrics
        total_time = time.time() - start_time
        
        # Clean up stage times (remove start times)
        clean_stage_times = {
            k: v for k, v in stage_times.items()
            if not k.endswith('_start')
        }
        
        processing_metrics = ProcessingMetrics(
            document_type=document_type,
            total_time=total_time,
            stage_times=clean_stage_times,
            success=success,
            error_type=error_type,
            validation_result=validation_result
        )
        
        metrics_collector.record_processing_metrics(processing_metrics)

# ============= Health Monitoring =============

class HealthMonitor:
    """Monitor application health"""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_time = {}
        self.check_interval = 60  # seconds
        
    def register_health_check(self, name: str, check_func):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_healthy = False
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global health monitor
health_monitor = HealthMonitor()

# ============= Default Health Checks =============

def check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits"""
    try:
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90  # Less than 90%
    except:
        return False

def check_disk_usage() -> bool:
    """Check if disk usage is within acceptable limits"""
    try:
        disk_percent = psutil.disk_usage('/').percent
        return disk_percent < 90  # Less than 90%
    except:
        return False

async def check_ocr_service() -> bool:
    """Check if OCR service is available"""
    try:
        from ..services.ocr_service import OCRService
        ocr_service = OCRService()
        return ocr_service.ocr_available
    except:
        return False

async def check_models_loaded() -> bool:
    """Check if AI models are loaded"""
    try:
        from ..core.performance import model_manager
        return len(model_manager.loaded_models) > 0
    except:
        return False

# Register default health checks
health_monitor.register_health_check("memory_usage", check_memory_usage)
health_monitor.register_health_check("disk_usage", check_disk_usage)
health_monitor.register_health_check("ocr_service", check_ocr_service)
health_monitor.register_health_check("models_loaded", check_models_loaded)

# ============= Database Integration (Optional) =============

# app/database/models.py
"""
Database models for storing pipeline results and metrics
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from datetime import datetime

Base = declarative_base()

class ProcessingHistory(Base):
    """Store processing history"""
    __tablename__ = "processing_history"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    stage_times = Column(JSON, nullable=True)
    validation_result = Column(Boolean, nullable=True)
    ocr_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ValidationHistory(Base):
    """Store validation results"""
    __tablename__ = "validation_history"
    
    id = Column(Integer, primary_key=True)
    kyc_filename = Column(String, nullable=False)
    nid_filename = Column(String, nullable=False)
    overall_validation = Column(Boolean, nullable=False)
    match_percentage = Column(Float, nullable=False)
    field_validations = Column(JSON, nullable=True)
    discrepancies = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetricsHistory(Base):
    """Store system metrics history"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True)
    memory_usage_mb = Column(Float, nullable=False)
    cpu_usage_percent = Column(Float, nullable=False)
    active_requests = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class OCRMetrics(Base):
    """Store OCR metrics"""
    __tablename__ = "ocr_metrics"
    
    id = Column(Integer, primary_key=True)
    provider = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=True)
    text_length = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ============= Database Manager =============

class DatabaseManager:
    """Manage database operations"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("KYC_DATABASE_URL")
        self.engine = None
        self.SessionLocal = None
        
        if self.database_url:
            self.initialize_database()
    
    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(self.database_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.engine = None
            self.SessionLocal = None
    
    def is_available(self) -> bool:
        """Check if database is available"""
        return self.engine is not None and self.SessionLocal is not None
    
    async def store_processing_result(self, result: Dict[str, Any]):
        """Store processing result in database"""
        if not self.is_available():
            return
        
        try:
            db = self.SessionLocal()
            
            processing_record = ProcessingHistory(
                filename=result.get("filename", ""),
                document_type=result.get("document_type", ""),
                processing_time_ms=result.get("processing_time_ms", 0),
                success=result.get("success", False),
                error_message=result.get("error"),
                stage_times=result.get("stage_times", {}),
                validation_result=result.get("validation_result"),
                ocr_enabled=result.get("ocr_enabled", False)
            )
            
            db.add(processing_record)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {e}")
        finally:
            if 'db' in locals():
                db.close()
    
    async def store_validation_result(self, result: Dict[str, Any]):
        """Store validation result in database"""
        if not self.is_available():
            return
        
        try:
            db = self.SessionLocal()
            
            validation_record = ValidationHistory(
                kyc_filename=result.get("kyc_filename", ""),
                nid_filename=result.get("nid_filename", ""),
                overall_validation=result.get("overall_validation", False),
                match_percentage=result.get("match_percentage", 0.0),
                field_validations=result.get("field_validations", {}),
                discrepancies=result.get("discrepancies", [])
            )
            
            db.add(validation_record)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")
        finally:
            if 'db' in locals():
                db.close()
    
    async def get_processing_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get processing statistics from database"""
        if not self.is_available():
            return {}
        
        try:
            db = self.SessionLocal()
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Query processing history
            records = db.query(ProcessingHistory).filter(
                ProcessingHistory.created_at >= cutoff_date
            ).all()
            
            # Calculate statistics
            total_requests = len(records)
            successful_requests = sum(1 for r in records if r.success)
            
            processing_times = [r.processing_time_ms for r in records]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "error_rate": (total_requests - successful_requests) / total_requests * 100 if total_requests > 0 else 0,
                "avg_processing_time_ms": avg_processing_time,
                "time_range_days": days
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {}
        finally:
            if 'db' in locals():
                db.close()

# Global database manager
database_manager = DatabaseManager()

# ============= Metrics API Endpoints =============

# app/routers/metrics.py
from fastapi import APIRouter, Query
from typing import Optional
import json

metrics_router = APIRouter()

@metrics_router.get("/performance-summary")
async def get_performance_summary(hours: int = Query(24, ge=1, le=168)):
    """Get performance summary for the last N hours"""
    return metrics_collector.get_performance_summary(hours)

@metrics_router.get("/health-detailed")
async def get_detailed_health():
    """Get detailed health check results"""
    return await health_monitor.run_health_checks()

@metrics_router.get("/system-metrics")
async def get_system_metrics():
    """Get current system metrics"""
    try:
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        disk_percent = psutil.disk_usage('/').percent
        
        return {
            "memory_usage_mb": round(memory_mb, 2),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "cpu_usage_percent": cpu_percent,
            "disk_usage_percent": disk_percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@metrics_router.get("/database-stats")
async def get_database_stats(days: int = Query(7, ge=1, le=30)):
    """Get database statistics"""
    if not database_manager.is_available():
        return {"message": "Database not configured"}
    
    return await database_manager.get_processing_stats(days)

@metrics_router.get("/error-analysis")
async def get_error_analysis():
    """Get error analysis"""
    return {
        "error_counts": dict(metrics_collector.error_counts),
        "validation_stats": dict(metrics_collector.validation_stats),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============= Monitoring Configuration =============

def setup_monitoring(app, enable_prometheus: bool = True, prometheus_port: int = 8000):
    """Setup monitoring for the application"""
    
    # Start Prometheus metrics server
    if enable_prometheus:
        try:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    # Add metrics router to app
    app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics"])
    
    # Add startup event to initialize monitoring
    @app.on_event("startup")
    async def initialize_monitoring():
        logger.info("Monitoring system initialized")
    
    # Add shutdown event to cleanup monitoring
    @app.on_event("shutdown")
    async def cleanup_monitoring():
        metrics_collector.stop_monitoring()
        logger.info("Monitoring system stopped")

# ============= Alert Manager =============

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_rules = []
        self.notification_channels = []
    
    def add_alert_rule(self, name: str, condition_func, message: str, severity: str = "warning"):
        """Add an alert rule"""
        self.alert_rules.append({
            "name": name,
            "condition": condition_func,
            "message": message,
            "severity": severity
        })
    
    def add_notification_channel(self, channel_func):
        """Add a notification channel"""
        self.notification_channels.append(channel_func)
    
    async def check_alerts(self):
        """Check all alert rules and send notifications"""
        for rule in self.alert_rules:
            try:
                if rule["condition"]():
                    await self._send_alert(rule)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    async def _send_alert(self, rule: Dict[str, Any]):
        """Send alert through all notification channels"""
        alert_data = {
            "name": rule["name"],
            "message": rule["message"],
            "severity": rule["severity"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for channel in self.notification_channels:
            try:
                await channel(alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert through channel: {e}")

# Example alert rules
def high_error_rate_check() -> bool:
    """Check if error rate is too high"""
    summary = metrics_collector.get_performance_summary(hours=1)
    return summary.get("error_rate_percent", 0) > 10

def high_memory_usage_check() -> bool:
    """Check if memory usage is too high"""
    return psutil.virtual_memory().percent > 85

# Global alert manager
alert_manager = AlertManager()
alert_manager.add_alert_rule(
    "high_error_rate",
    high_error_rate_check,
    "Error rate exceeded 10% in the last hour",
    "critical"
)
alert_manager.add_alert_rule(
    "high_memory_usage",
    high_memory_usage_check,
    "Memory usage exceeded 85%",
    "warning"
)