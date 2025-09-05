"""Performance monitoring and metrics collection for the Deep Agent System."""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from deep_agent_system.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric value with timestamp."""
    
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric value to dictionary.
        
        Returns:
            Dictionary representation of the metric value
        """
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    
    count: int
    sum: float
    min: float
    max: float
    avg: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric summary to dictionary.
        
        Returns:
            Dictionary representation of the metric summary
        """
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "avg": self.avg,
        }


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_values_per_metric: int = 1000):
        """Initialize the metrics collector.
        
        Args:
            max_values_per_metric: Maximum number of values to store per metric
        """
        self.max_values_per_metric = max_values_per_metric
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_values_per_metric))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Name of the counter
            value: Value to increment by (default: 1)
            labels: Optional labels for the metric
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._counters[metric_key] += value
            
            # Also store as time series
            metric_value = MetricValue(
                value=self._counters[metric_key],
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
            )
            self._metrics[metric_key].append(metric_value)
        
        logger.debug(
            f"Counter incremented: {name}",
            extra={
                "metric_name": name,
                "metric_type": "counter",
                "value": value,
                "total": self._counters[metric_key],
                "labels": labels,
            }
        )
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Name of the gauge
            value: Current value of the gauge
            labels: Optional labels for the metric
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._gauges[metric_key] = value
            
            # Also store as time series
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
            )
            self._metrics[metric_key].append(metric_value)
        
        logger.debug(
            f"Gauge set: {name}",
            extra={
                "metric_name": name,
                "metric_type": "gauge",
                "value": value,
                "labels": labels,
            }
        )
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram metric.
        
        Args:
            name: Name of the histogram
            value: Value to record
            labels: Optional labels for the metric
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._histograms[metric_key].append(value)
            
            # Keep only recent values to prevent memory growth
            if len(self._histograms[metric_key]) > self.max_values_per_metric:
                self._histograms[metric_key] = self._histograms[metric_key][-self.max_values_per_metric:]
            
            # Also store as time series
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
            )
            self._metrics[metric_key].append(metric_value)
        
        logger.debug(
            f"Histogram recorded: {name}",
            extra={
                "metric_name": name,
                "metric_type": "histogram",
                "value": value,
                "labels": labels,
            }
        )
    
    def timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric.
        
        Args:
            name: Name of the timing metric
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        self.histogram(f"{name}_duration_seconds", duration, labels)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value.
        
        Args:
            name: Name of the counter
            labels: Optional labels for the metric
            
        Returns:
            Current counter value
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            return self._counters.get(metric_key, 0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value.
        
        Args:
            name: Name of the gauge
            labels: Optional labels for the metric
            
        Returns:
            Current gauge value
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            return self._gauges.get(metric_key, 0.0)
    
    def get_histogram_summary(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[MetricSummary]:
        """Get histogram summary statistics.
        
        Args:
            name: Name of the histogram
            labels: Optional labels for the metric
            
        Returns:
            Histogram summary or None if no data
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            values = self._histograms.get(metric_key, [])
            
            if not values:
                return None
            
            return MetricSummary(
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values),
            )
    
    def get_time_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[MetricValue]:
        """Get time series data for a metric.
        
        Args:
            name: Name of the metric
            labels: Optional labels for the metric
            since: Optional timestamp to filter from
            
        Returns:
            List of metric values
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            values = list(self._metrics.get(metric_key, []))
            
            if since:
                values = [v for v in values if v.timestamp >= since]
            
            return values
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_summary(name.split("|")[0], self._parse_labels(name))
                    for name in self._histograms.keys()
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
        
        logger.info("All metrics reset")
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate a unique key for a metric with labels.
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Returns:
            Unique metric key
        """
        if not labels:
            return name
        
        # Sort labels for consistent key generation
        label_str = "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _parse_labels(self, metric_key: str) -> Optional[Dict[str, str]]:
        """Parse labels from a metric key.
        
        Args:
            metric_key: Metric key containing labels
            
        Returns:
            Parsed labels or None
        """
        if "|" not in metric_key:
            return None
        
        _, label_str = metric_key.split("|", 1)
        labels = {}
        
        for pair in label_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                labels[key] = value
        
        return labels if labels else None


class Timer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize the timer.
        
        Args:
            metrics_collector: Metrics collector to record timing to
            metric_name: Name of the timing metric
            labels: Optional labels for the metric
        """
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and record the duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.timing(self.metric_name, duration, self.labels)


class PerformanceMonitor:
    """Monitors system and agent performance."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        collection_interval: float = 30.0,
    ):
        """Initialize the performance monitor.
        
        Args:
            metrics_collector: Metrics collector to use
            collection_interval: Interval between metric collections in seconds
        """
        self.metrics_collector = metrics_collector
        self.collection_interval = collection_interval
        self._running = False
        self._monitor_task = None
        self._system_metrics_enabled = True
        
        # Try to import psutil for system metrics
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not available, system metrics disabled")
            self._system_metrics_enabled = False
            self._psutil = None
    
    async def start(self) -> None:
        """Start the performance monitor."""
        if self._running:
            logger.warning("Performance monitor already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitor started")
    
    async def stop(self) -> None:
        """Stop the performance monitor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        if not self._system_metrics_enabled or not self._psutil:
            return
        
        try:
            # CPU metrics
            cpu_percent = self._psutil.cpu_percent(interval=None)
            self.metrics_collector.gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = self._psutil.virtual_memory()
            self.metrics_collector.gauge("system_memory_percent", memory.percent)
            self.metrics_collector.gauge("system_memory_available_mb", memory.available / 1024 / 1024)
            self.metrics_collector.gauge("system_memory_used_mb", memory.used / 1024 / 1024)
            
            # Process metrics
            process = self._psutil.Process()
            process_memory = process.memory_info()
            self.metrics_collector.gauge("process_memory_rss_mb", process_memory.rss / 1024 / 1024)
            self.metrics_collector.gauge("process_memory_vms_mb", process_memory.vms / 1024 / 1024)
            self.metrics_collector.gauge("process_cpu_percent", process.cpu_percent())
            
            # Thread count
            self.metrics_collector.gauge("process_thread_count", process.num_threads())
            
            logger.debug("System metrics collected")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class AgentMetrics:
    """Tracks metrics specific to agent operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, agent_id: str):
        """Initialize agent metrics.
        
        Args:
            metrics_collector: Metrics collector to use
            agent_id: ID of the agent
        """
        self.metrics_collector = metrics_collector
        self.agent_id = agent_id
        self.labels = {"agent_id": agent_id}
    
    def message_sent(self, recipient_id: str, message_type: str) -> None:
        """Record a message sent by the agent.
        
        Args:
            recipient_id: ID of the message recipient
            message_type: Type of the message
        """
        labels = {**self.labels, "recipient_id": recipient_id, "message_type": message_type}
        self.metrics_collector.counter("agent_messages_sent_total", 1, labels)
    
    def message_received(self, sender_id: str, message_type: str) -> None:
        """Record a message received by the agent.
        
        Args:
            sender_id: ID of the message sender
            message_type: Type of the message
        """
        labels = {**self.labels, "sender_id": sender_id, "message_type": message_type}
        self.metrics_collector.counter("agent_messages_received_total", 1, labels)
    
    def message_processed(self, processing_time: float, success: bool) -> None:
        """Record message processing metrics.
        
        Args:
            processing_time: Time taken to process the message in seconds
            success: Whether processing was successful
        """
        status = "success" if success else "error"
        labels = {**self.labels, "status": status}
        
        self.metrics_collector.counter("agent_messages_processed_total", 1, labels)
        self.metrics_collector.timing("agent_message_processing", processing_time, labels)
    
    def rag_query(self, query_type: str, duration: float, result_count: int) -> None:
        """Record RAG query metrics.
        
        Args:
            query_type: Type of RAG query (vector, graph, hybrid)
            duration: Query duration in seconds
            result_count: Number of results returned
        """
        labels = {**self.labels, "query_type": query_type}
        
        self.metrics_collector.counter("agent_rag_queries_total", 1, labels)
        self.metrics_collector.timing("agent_rag_query", duration, labels)
        self.metrics_collector.histogram("agent_rag_results_count", result_count, labels)
    
    def workflow_step(self, workflow_id: str, step_name: str, duration: float, success: bool) -> None:
        """Record workflow step metrics.
        
        Args:
            workflow_id: ID of the workflow
            step_name: Name of the workflow step
            duration: Step duration in seconds
            success: Whether the step was successful
        """
        status = "success" if success else "error"
        labels = {
            **self.labels,
            "workflow_id": workflow_id,
            "step_name": step_name,
            "status": status,
        }
        
        self.metrics_collector.counter("agent_workflow_steps_total", 1, labels)
        self.metrics_collector.timing("agent_workflow_step", duration, labels)
    
    def error_occurred(self, error_type: str, component: str) -> None:
        """Record an error occurrence.
        
        Args:
            error_type: Type of error that occurred
            component: Component where the error occurred
        """
        labels = {**self.labels, "error_type": error_type, "component": component}
        self.metrics_collector.counter("agent_errors_total", 1, labels)


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.
    
    Returns:
        Global metrics collector
    """
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    
    return _global_metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance.
    
    Returns:
        Global performance monitor
    """
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        metrics_collector = get_metrics_collector()
        _global_performance_monitor = PerformanceMonitor(metrics_collector)
    
    return _global_performance_monitor


def get_agent_metrics(agent_id: str) -> AgentMetrics:
    """Get agent-specific metrics tracker.
    
    Args:
        agent_id: ID of the agent
        
    Returns:
        Agent metrics tracker
    """
    metrics_collector = get_metrics_collector()
    return AgentMetrics(metrics_collector, agent_id)


# Decorator for automatic timing
def timed_operation(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to automatically time function execution.
    
    Args:
        metric_name: Name of the timing metric
        labels: Optional labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics_collector = get_metrics_collector()
            with Timer(metrics_collector, metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def start_monitoring() -> None:
    """Start the global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.start()


async def stop_monitoring() -> None:
    """Stop the global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.stop()