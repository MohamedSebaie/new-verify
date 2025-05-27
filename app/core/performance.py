# app/core/performance.py - Performance Optimization Utilities

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from functools import wraps, lru_cache
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import hashlib
import pickle
import gc
from contextlib import contextmanager
import resource

logger = logging.getLogger(__name__)

# ============= Performance Monitoring =============

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.resource_usage = {}
        self.bottlenecks = []
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            
            # Remove from start_times
            del self.start_times[operation]
            
            return duration
        return 0.0
    
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
    
    def record_resource_usage(self, operation: str):
        """Record current resource usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            self.resource_usage[operation] = {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to record resource usage: {e}")
    
    def identify_bottlenecks(self, threshold_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for operation, durations in self.metrics.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                
                if avg_duration > threshold_seconds or max_duration > threshold_seconds * 2:
                    bottlenecks.append({
                        "operation": operation,
                        "avg_duration": avg_duration,
                        "max_duration": max_duration,
                        "count": len(durations),
                        "severity": "high" if avg_duration > threshold_seconds * 2 else "medium"
                    })
        
        return sorted(bottlenecks, key=lambda x: x["avg_duration"], reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, durations in self.metrics.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations)
                }
        
        return summary

# Global performance monitor
performance_monitor = PerformanceMonitor()

# ============= Caching System =============

class LRUCache:
    """Custom LRU Cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.access_order = []
        self._lock = threading.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.creation_times:
            return True
        
        age = time.time() - self.creation_times[key]
        return age > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_order:
            return
        
        lru_key = self.access_order[0]
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """Remove key from all data structures"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.creation_times:
            del self.creation_times[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        with self._lock:
            if key not in self.cache or self._is_expired(key):
                return None
            
            # Update access time and order
            self.access_times[key] = time.time()
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                self._remove_key(key)
            
            # Evict if at max capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            now = time.time()
            self.cache[key] = value
            self.access_times[key] = now
            self.creation_times[key] = now
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

# Global caches
classification_cache = LRUCache(max_size=500, ttl_seconds=3600)
ocr_cache = LRUCache(max_size=1000, ttl_seconds=7200)
result_cache = LRUCache(max_size=200, ttl_seconds=1800)

# ============= Caching Decorators =============

def cache_classification_result(cache_instance: LRUCache = classification_cache):
    """Decorator to cache classification results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key based on file content hash
            file_path = args[1] if len(args) > 1 else kwargs.get('file_path')
            if file_path:
                try:
                    file_hash = _get_file_hash(file_path)
                    cache_key = f"classification_{file_hash}"
                    
                    # Check cache
                    cached_result = cache_instance.get(cache_key)
                    if cached_result is not None:
                        logger.info(f"Classification cache hit for {file_path}")
                        return cached_result
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Cache result
                    cache_instance.put(cache_key, result)
                    logger.info(f"Classification result cached for {file_path}")
                    
                    return result
                except Exception as e:
                    logger.warning(f"Caching failed for classification: {e}")
                    return await func(*args, **kwargs)
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            file_path = args[1] if len(args) > 1 else kwargs.get('file_path')
            if file_path:
                try:
                    file_hash = _get_file_hash(file_path)
                    cache_key = f"classification_{file_hash}"
                    
                    cached_result = cache_instance.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                    
                    result = func(*args, **kwargs)
                    cache_instance.put(cache_key, result)
                    
                    return result
                except Exception as e:
                    logger.warning(f"Caching failed for classification: {e}")
                    return func(*args, **kwargs)
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def cache_ocr_result(cache_instance: LRUCache = ocr_cache):
    """Decorator to cache OCR results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key based on image content and prompt
            image_path = args[1] if len(args) > 1 else kwargs.get('image_path')
            prompt = args[2] if len(args) > 2 else kwargs.get('prompt', '')
            
            if image_path:
                try:
                    file_hash = _get_file_hash(image_path)
                    prompt_hash = hashlib.md5(str(prompt).encode()).hexdigest()[:8]
                    cache_key = f"ocr_{file_hash}_{prompt_hash}"
                    
                    cached_result = cache_instance.get(cache_key)
                    if cached_result is not None:
                        logger.info(f"OCR cache hit for {image_path}")
                        return cached_result
                    
                    result = await func(*args, **kwargs)
                    
                    # Only cache successful OCR results
                    if result.get('success', False):
                        cache_instance.put(cache_key, result)
                        logger.info(f"OCR result cached for {image_path}")
                    
                    return result
                except Exception as e:
                    logger.warning(f"OCR caching failed: {e}")
                    return await func(*args, **kwargs)
            
            return await func(*args, **kwargs)
        
        return async_wrapper
    return decorator

def _get_file_hash(file_path: Union[str, Path]) -> str:
    """Get hash of file content for cache key"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
    except Exception:
        # Fallback to file path and modification time
        path = Path(file_path)
        stat = path.stat()
        content = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()

# ============= Batch Processing Optimization =============

class BatchProcessor:
    """Optimized batch processing for documents"""
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 5):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
    
    async def process_batch_async(
        self, 
        items: List[Any], 
        processor_func: Callable,
        use_processes: bool = False
    ) -> List[Any]:
        """Process batch of items asynchronously"""
        if not items:
            return []
        
        # Split into chunks
        chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]
        
        # Process chunks
        tasks = []
        for chunk in chunks:
            if use_processes:
                task = asyncio.get_event_loop().run_in_executor(
                    self.process_executor, self._process_chunk, chunk, processor_func
                )
            else:
                task = asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, self._process_chunk, chunk, processor_func
                )
            tasks.append(task)
        
        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        flattened_results = []
        for chunk_result in results:
            if isinstance(chunk_result, Exception):
                logger.error(f"Batch processing error: {chunk_result}")
                # Add error placeholders
                flattened_results.extend([{"error": str(chunk_result)}] * self.chunk_size)
            else:
                flattened_results.extend(chunk_result)
        
        return flattened_results[:len(items)]  # Trim to original size
    
    def _process_chunk(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single chunk"""
        results = []
        for item in chunk:
            try:
                if asyncio.iscoroutinefunction(processor_func):
                    # Handle async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(processor_func(item))
                    loop.close()
                else:
                    result = processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item in chunk: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def cleanup(self):
        """Cleanup executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# Global batch processor
batch_processor = BatchProcessor()

# ============= Memory Management =============

class MemoryManager:
    """Manage memory usage and cleanup"""
    
    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold_mb = max_memory_mb * 0.8  # Cleanup at 80%
        self.monitoring_enabled = True
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {}
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is high"""
        memory_usage = self.get_memory_usage()
        current_mb = memory_usage.get("rss_mb", 0)
        
        return current_mb > self.cleanup_threshold_mb
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        logger.info("Forcing garbage collection")
        
        # Clear caches first
        classification_cache.clear()
        ocr_cache.clear()
        result_cache.clear()
        
        # Force GC
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Log memory usage after cleanup
        memory_usage = self.get_memory_usage()
        logger.info(f"Memory after cleanup: {memory_usage.get('rss_mb', 0):.1f} MB")
    
    @contextmanager
    def memory_limit_context(self, limit_mb: int):
        """Context manager to enforce memory limits"""
        try:
            # Set memory limit (Unix only)
            if hasattr(resource, 'RLIMIT_AS'):
                old_limit = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (limit_mb * 1024 * 1024, old_limit[1]))
            
            yield
            
        except Exception as e:
            logger.warning(f"Memory limit context error: {e}")
        finally:
            # Restore old limit
            if hasattr(resource, 'RLIMIT_AS'):
                try:
                    resource.setrlimit(resource.RLIMIT_AS, old_limit)
                except:
                    pass
    
    def monitor_memory_usage(self):
        """Monitor and log memory usage"""
        if not self.monitoring_enabled:
            return
        
        memory_usage = self.get_memory_usage()
        current_mb = memory_usage.get("rss_mb", 0)
        
        if current_mb > self.cleanup_threshold_mb:
            logger.warning(f"High memory usage: {current_mb:.1f} MB")
            self.force_garbage_collection()
        elif current_mb > self.max_memory_mb * 0.6:  # 60% threshold
            logger.info(f"Memory usage: {current_mb:.1f} MB")

# Global memory manager
memory_manager = MemoryManager()

# ============= Model Loading Optimization =============

class ModelManager:
    """Optimized model loading and management"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_access_times = {}
        self.max_loaded_models = 3
        self._lock = threading.Lock()
    
    def get_model(self, model_name: str, model_path: str, loader_func: Callable):
        """Get model with lazy loading and caching"""
        with self._lock:
            # Check if model is already loaded
            if model_name in self.loaded_models:
                self.model_access_times[model_name] = time.time()
                return self.loaded_models[model_name]
            
            # Evict least recently used model if needed
            if len(self.loaded_models) >= self.max_loaded_models:
                self._evict_lru_model()
            
            # Load model
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            try:
                model = loader_func(model_path)
                load_time = time.time() - start_time
                
                self.loaded_models[model_name] = model
                self.model_access_times[model_name] = time.time()
                
                logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def _evict_lru_model(self):
        """Evict least recently used model"""
        if not self.model_access_times:
            return
        
        # Find LRU model
        lru_model = min(self.model_access_times.items(), key=lambda x: x[1])[0]
        
        # Remove model
        if lru_model in self.loaded_models:
            del self.loaded_models[lru_model]
            logger.info(f"Evicted model from cache: {lru_model}")
        
        if lru_model in self.model_access_times:
            del self.model_access_times[lru_model]
    
    def preload_models(self, model_configs: List[Dict[str, Any]]):
        """Preload models on startup"""
        for config in model_configs:
            try:
                self.get_model(
                    config['name'],
                    config['path'],
                    config['loader']
                )
            except Exception as e:
                logger.error(f"Failed to preload model {config['name']}: {e}")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        with self._lock:
            self.loaded_models.clear()
            self.model_access_times.clear()
            logger.info("All models unloaded")

# Global model manager
model_manager = ModelManager()

# ============= Performance Optimization Decorators =============

def optimize_performance(monitor_memory: bool = True, enable_caching: bool = True):
    """Decorator to optimize function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Start monitoring
            with performance_monitor.timer(func_name):
                if monitor_memory:
                    performance_monitor.record_resource_usage(f"{func_name}_start")
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if monitor_memory:
                        performance_monitor.record_resource_usage(f"{func_name}_end")
                        memory_manager.monitor_memory_usage()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Performance optimization error in {func_name}: {e}")
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            with performance_monitor.timer(func_name):
                if monitor_memory:
                    performance_monitor.record_resource_usage(f"{func_name}_start")
                
                try:
                    result = func(*args, **kwargs)
                    
                    if monitor_memory:
                        performance_monitor.record_resource_usage(f"{func_name}_end")
                        memory_manager.monitor_memory_usage()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Performance optimization error in {func_name}: {e}")
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ============= Resource Pool =============

class ResourcePool:
    """Pool of reusable resources (like database connections, etc.)"""
    
    def __init__(self, factory_func: Callable, max_size: int = 10):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = Queue(maxsize=max_size)
        self.active_count = 0
        self._lock = threading.Lock()
    
    def get_resource(self):
        """Get resource from pool"""
        try:
            # Try to get from pool (non-blocking)
            resource = self.pool.get_nowait()
            return resource
        except:
            # Pool is empty, create new resource if under limit
            with self._lock:
                if self.active_count < self.max_size:
                    resource = self.factory_func()
                    self.active_count += 1
                    return resource
                else:
                    # Wait for available resource
                    return self.pool.get(timeout=30)
    
    def return_resource(self, resource):
        """Return resource to pool"""
        try:
            self.pool.put_nowait(resource)
        except:
            # Pool is full, resource will be garbage collected
            with self._lock:
                self.active_count -= 1
    
    @contextmanager
    def resource_context(self):
        """Context manager for resource usage"""
        resource = self.get_resource()
        try:
            yield resource
        finally:
            self.return_resource(resource)

# ============= Performance Testing Utilities =============

class PerformanceTester:
    """Utilities for performance testing"""
    
    @staticmethod
    async def benchmark_function(func: Callable, args_list: List[tuple], iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a function with different arguments"""
        results = {
            "function_name": func.__name__,
            "iterations": iterations,
            "test_cases": []
        }
        
        for i, args in enumerate(args_list):
            case_results = []
            
            for _ in range(iterations):
                start_time = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(*args)
                    else:
                        func(*args)
                    
                    duration = time.time() - start_time
                    case_results.append(duration)
                    
                except Exception as e:
                    logger.error(f"Benchmark error: {e}")
                    case_results.append(float('inf'))
            
            # Calculate statistics
            valid_results = [r for r in case_results if r != float('inf')]
            if valid_results:
                results["test_cases"].append({
                    "case_index": i,
                    "args": str(args),
                    "avg_duration": sum(valid_results) / len(valid_results),
                    "min_duration": min(valid_results),
                    "max_duration": max(valid_results),
                    "success_rate": len(valid_results) / len(case_results)
                })
        
        return results
    
    @staticmethod
    def load_test_pipeline(num_requests: int = 100, concurrent_requests: int = 10):
        """Simple load test for pipeline endpoints"""
        # This would be implemented with actual HTTP requests
        # For now, just a placeholder
        logger.info(f"Load testing with {num_requests} requests, {concurrent_requests} concurrent")
        
        # Results would include:
        # - Average response time
        # - Throughput (requests per second)
        # - Error rate
        # - Resource usage during test
        
        return {
            "total_requests": num_requests,
            "concurrent_requests": concurrent_requests,
            "avg_response_time": 0.0,
            "throughput": 0.0,
            "error_rate": 0.0
        }

# ============= Performance Configuration =============

def configure_performance_optimizations():
    """Configure global performance optimizations"""
    from ..core.config import settings
    
    # Configure batch processor
    global batch_processor
    batch_processor = BatchProcessor(
        max_workers=settings.MAX_CONCURRENT_DOCUMENTS,
        chunk_size=5
    )
    
    # Configure memory manager
    global memory_manager
    memory_manager = MemoryManager(
        max_memory_mb=settings.MAX_MEMORY_USAGE_MB
    )
    
    # Configure caches
    global classification_cache, ocr_cache, result_cache
    
    if settings.ENABLE_CLASSIFICATION_CACHING:
        classification_cache = LRUCache(
            max_size=settings.CACHE_MAX_SIZE // 2,
            ttl_seconds=settings.CACHE_TTL_SECONDS
        )
    
    if settings.ENABLE_OCR_CACHING:
        ocr_cache = LRUCache(
            max_size=settings.CACHE_MAX_SIZE,
            ttl_seconds=settings.CACHE_TTL_SECONDS * 2
        )
    
    if settings.ENABLE_RESULT_CACHING:
        result_cache = LRUCache(
            max_size=settings.CACHE_MAX_SIZE // 4,
            ttl_seconds=settings.CACHE_TTL_SECONDS // 2
        )
    
    logger.info("Performance optimizations configured")

# Initialize performance optimizations
configure_performance_optimizations()