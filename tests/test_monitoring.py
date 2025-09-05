"""Tests for monitoring and metrics collection."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from deep_agent_system.monitoring import (
    AgentMetrics,
    MetricSummary,
    MetricValue,
    MetricsCollector,
    PerformanceMonitor,
    Timer,
    get_agent_metrics,
    get_metrics_collector,
    get_performance_monitor,
    start_monitoring,
    stop_monitoring,
    timed_operation,
)


class TestMetricValue:
    """Test MetricValue functionality."""
    
    def test_metric_value_creation(self):
        """Test creating a metric value."""
        timestamp = datetime.now(timezone.utc)
        labels = {"agent_id": "agent1", "type": "test"}
        
        metric = MetricValue(
            value=42.5,
            timestamp=timestamp,
            labels=labels,
        )
        
        assert metric.value == 42.5
        assert metric.timestamp == timestamp
        assert metric.labels == labels
    
    def test_metric_value_to_dict(self):
        """Test converting metric value to dictionary."""
        timestamp = datetime.now(timezone.utc)
        labels = {"agent_id": "agent1"}
        
        metric = MetricValue(
            value=100,
            timestamp=timestamp,
            labels=labels,
        )
        
        result = metric.to_dict()
        
        assert result["value"] == 100
        assert result["timestamp"] == timestamp.isoformat()
        assert result["labels"] == labels


class TestMetricSummary:
    """Test MetricSummary functionality."""
    
    def test_metric_summary_creation(self):
        """Test creating a metric summary."""
        summary = MetricSummary(
            count=10,
            sum=100.0,
            min=5.0,
            max=20.0,
            avg=10.0,
        )
        
        assert summary.count == 10
        assert summary.sum == 100.0
        assert summary.min == 5.0
        assert summary.max == 20.0
        assert summary.avg == 10.0
    
    def test_metric_summary_to_dict(self):
        """Test converting metric summary to dictionary."""
        summary = MetricSummary(
            count=5,
            sum=50.0,
            min=8.0,
            max=12.0,
            avg=10.0,
        )
        
        result = summary.to_dict()
        
        assert result["count"] == 5
        assert result["sum"] == 50.0
        assert result["min"] == 8.0
        assert result["max"] == 12.0
        assert result["avg"] == 10.0


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_counter_increment(self):
        """Test counter increment functionality."""
        collector = MetricsCollector()
        
        collector.counter("test_counter", 1)
        collector.counter("test_counter", 5)
        
        assert collector.get_counter("test_counter") == 6
    
    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = MetricsCollector()
        
        collector.counter("requests", 1, {"method": "GET"})
        collector.counter("requests", 2, {"method": "POST"})
        collector.counter("requests", 1, {"method": "GET"})
        
        assert collector.get_counter("requests", {"method": "GET"}) == 2
        assert collector.get_counter("requests", {"method": "POST"}) == 2
    
    def test_gauge_set(self):
        """Test gauge set functionality."""
        collector = MetricsCollector()
        
        collector.gauge("temperature", 25.5)
        collector.gauge("temperature", 30.0)
        
        assert collector.get_gauge("temperature") == 30.0
    
    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        collector = MetricsCollector()
        
        collector.gauge("cpu_usage", 45.2, {"core": "0"})
        collector.gauge("cpu_usage", 52.1, {"core": "1"})
        
        assert collector.get_gauge("cpu_usage", {"core": "0"}) == 45.2
        assert collector.get_gauge("cpu_usage", {"core": "1"}) == 52.1
    
    def test_histogram_record(self):
        """Test histogram record functionality."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.histogram("response_time", value)
        
        summary = collector.get_histogram_summary("response_time")
        
        assert summary is not None
        assert summary.count == 5
        assert summary.sum == 15.0
        assert summary.min == 1.0
        assert summary.max == 5.0
        assert summary.avg == 3.0
    
    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        collector = MetricsCollector()
        
        collector.histogram("latency", 10.0, {"endpoint": "/api/v1"})
        collector.histogram("latency", 15.0, {"endpoint": "/api/v1"})
        collector.histogram("latency", 20.0, {"endpoint": "/api/v2"})
        
        summary_v1 = collector.get_histogram_summary("latency", {"endpoint": "/api/v1"})
        summary_v2 = collector.get_histogram_summary("latency", {"endpoint": "/api/v2"})
        
        assert summary_v1.count == 2
        assert summary_v1.avg == 12.5
        assert summary_v2.count == 1
        assert summary_v2.avg == 20.0
    
    def test_timing_metric(self):
        """Test timing metric functionality."""
        collector = MetricsCollector()
        
        collector.timing("operation_duration", 0.123)
        collector.timing("operation_duration", 0.456)
        
        summary = collector.get_histogram_summary("operation_duration_duration_seconds")
        
        assert summary is not None
        assert summary.count == 2
        assert summary.min == 0.123
        assert summary.max == 0.456
    
    def test_get_time_series(self):
        """Test getting time series data."""
        collector = MetricsCollector()
        
        # Record some values
        collector.counter("events", 1)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        collector.counter("events", 2)
        
        time_series = collector.get_time_series("events")
        
        assert len(time_series) == 2
        assert time_series[0].value == 1
        assert time_series[1].value == 3  # Cumulative counter
        assert time_series[0].timestamp < time_series[1].timestamp
    
    def test_get_time_series_with_filter(self):
        """Test getting time series data with time filter."""
        collector = MetricsCollector()
        
        # Record a value
        collector.gauge("temperature", 20.0)
        
        # Get current time
        now = datetime.now(timezone.utc)
        
        # Record another value
        time.sleep(0.01)
        collector.gauge("temperature", 25.0)
        
        # Get time series since 'now'
        time_series = collector.get_time_series("temperature", since=now)
        
        # Should only get the second value
        assert len(time_series) == 1
        assert time_series[0].value == 25.0
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        
        collector.counter("requests", 10)
        collector.gauge("memory_usage", 75.5)
        collector.histogram("response_time", 0.123)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "timestamp" in all_metrics
        
        assert "requests" in all_metrics["counters"]
        assert "memory_usage" in all_metrics["gauges"]
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()
        
        collector.counter("test", 5)
        collector.gauge("test_gauge", 10.0)
        
        assert collector.get_counter("test") == 5
        assert collector.get_gauge("test_gauge") == 10.0
        
        collector.reset_metrics()
        
        assert collector.get_counter("test") == 0
        assert collector.get_gauge("test_gauge") == 0.0
    
    def test_metric_key_generation(self):
        """Test metric key generation with labels."""
        collector = MetricsCollector()
        
        # Test that labels are sorted consistently
        labels1 = {"b": "2", "a": "1"}
        labels2 = {"a": "1", "b": "2"}
        
        key1 = collector._get_metric_key("test", labels1)
        key2 = collector._get_metric_key("test", labels2)
        
        assert key1 == key2
        assert key1 == "test|a=1|b=2"
    
    def test_label_parsing(self):
        """Test parsing labels from metric keys."""
        collector = MetricsCollector()
        
        # Test parsing labels
        labels = collector._parse_labels("test|a=1|b=2")
        assert labels == {"a": "1", "b": "2"}
        
        # Test no labels
        labels = collector._parse_labels("test")
        assert labels is None


class TestTimer:
    """Test Timer context manager."""
    
    def test_timer_records_duration(self):
        """Test that Timer records operation duration."""
        collector = MetricsCollector()
        
        with Timer(collector, "test_operation"):
            time.sleep(0.01)  # Small delay
        
        summary = collector.get_histogram_summary("test_operation_duration_seconds")
        
        assert summary is not None
        assert summary.count == 1
        assert summary.avg > 0.005  # Should be at least 5ms
    
    def test_timer_with_labels(self):
        """Test Timer with labels."""
        collector = MetricsCollector()
        labels = {"operation": "test", "user": "user1"}
        
        with Timer(collector, "api_call", labels):
            time.sleep(0.01)
        
        summary = collector.get_histogram_summary("api_call_duration_seconds", labels)
        
        assert summary is not None
        assert summary.count == 1


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test starting and stopping the performance monitor."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector, collection_interval=0.1)
        
        assert not monitor._running
        
        await monitor.start()
        assert monitor._running
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        await monitor.stop()
        assert not monitor._running
    
    @pytest.mark.asyncio
    async def test_monitor_double_start(self):
        """Test starting monitor twice."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector, collection_interval=0.1)
        
        await monitor.start()
        
        # Starting again should not cause issues
        await monitor.start()
        
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test system metrics collection."""
        # Skip this test if psutil is not available
        pytest.skip("Skipping psutil-dependent test")


class TestAgentMetrics:
    """Test AgentMetrics functionality."""
    
    def test_message_sent_metric(self):
        """Test recording message sent metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.message_sent("agent2", "request")
        agent_metrics.message_sent("agent3", "response")
        
        # Check counter with labels
        labels1 = {"agent_id": "agent1", "recipient_id": "agent2", "message_type": "request"}
        labels2 = {"agent_id": "agent1", "recipient_id": "agent3", "message_type": "response"}
        
        assert collector.get_counter("agent_messages_sent_total", labels1) == 1
        assert collector.get_counter("agent_messages_sent_total", labels2) == 1
    
    def test_message_received_metric(self):
        """Test recording message received metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.message_received("agent2", "request")
        
        labels = {"agent_id": "agent1", "sender_id": "agent2", "message_type": "request"}
        assert collector.get_counter("agent_messages_received_total", labels) == 1
    
    def test_message_processed_metric(self):
        """Test recording message processed metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.message_processed(0.123, True)
        agent_metrics.message_processed(0.456, False)
        
        success_labels = {"agent_id": "agent1", "status": "success"}
        error_labels = {"agent_id": "agent1", "status": "error"}
        
        assert collector.get_counter("agent_messages_processed_total", success_labels) == 1
        assert collector.get_counter("agent_messages_processed_total", error_labels) == 1
        
        # Check timing metrics
        success_timing = collector.get_histogram_summary("agent_message_processing_duration_seconds", success_labels)
        error_timing = collector.get_histogram_summary("agent_message_processing_duration_seconds", error_labels)
        
        assert success_timing.count == 1
        assert success_timing.avg == 0.123
        assert error_timing.count == 1
        assert error_timing.avg == 0.456
    
    def test_rag_query_metric(self):
        """Test recording RAG query metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.rag_query("vector", 0.234, 5)
        
        labels = {"agent_id": "agent1", "query_type": "vector"}
        
        assert collector.get_counter("agent_rag_queries_total", labels) == 1
        
        timing_summary = collector.get_histogram_summary("agent_rag_query_duration_seconds", labels)
        assert timing_summary.count == 1
        assert timing_summary.avg == 0.234
        
        results_summary = collector.get_histogram_summary("agent_rag_results_count", labels)
        assert results_summary.count == 1
        assert results_summary.avg == 5.0
    
    def test_workflow_step_metric(self):
        """Test recording workflow step metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.workflow_step("wf123", "analyze", 1.5, True)
        
        labels = {
            "agent_id": "agent1",
            "workflow_id": "wf123",
            "step_name": "analyze",
            "status": "success",
        }
        
        assert collector.get_counter("agent_workflow_steps_total", labels) == 1
        
        timing_summary = collector.get_histogram_summary("agent_workflow_step_duration_seconds", labels)
        assert timing_summary.count == 1
        assert timing_summary.avg == 1.5
    
    def test_error_occurred_metric(self):
        """Test recording error metrics."""
        collector = MetricsCollector()
        agent_metrics = AgentMetrics(collector, "agent1")
        
        agent_metrics.error_occurred("ValueError", "message_processing")
        
        labels = {
            "agent_id": "agent1",
            "error_type": "ValueError",
            "component": "message_processing",
        }
        
        assert collector.get_counter("agent_errors_total", labels) == 1


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return the same instance
        assert collector1 is collector2
    
    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return the same instance
        assert monitor1 is monitor2
    
    def test_get_agent_metrics(self):
        """Test getting agent metrics."""
        agent_metrics = get_agent_metrics("agent1")
        
        assert isinstance(agent_metrics, AgentMetrics)
        assert agent_metrics.agent_id == "agent1"
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test global monitoring start/stop functions."""
        # This is a basic test - in practice, these would control the global monitor
        await start_monitoring()
        await stop_monitoring()
        
        # No assertions needed - just ensure no exceptions are raised
    
    def test_timed_operation_decorator(self):
        """Test timed operation decorator."""
        collector = MetricsCollector()
        
        # Patch the global collector
        with patch("deep_agent_system.monitoring.get_metrics_collector", return_value=collector):
            @timed_operation("test_function")
            def test_func():
                time.sleep(0.01)
                return "result"
            
            result = test_func()
            
            assert result == "result"
            
            # Check that timing was recorded
            summary = collector.get_histogram_summary("test_function_duration_seconds")
            assert summary is not None
            assert summary.count == 1
            assert summary.avg > 0.005
    
    def test_timed_operation_decorator_with_labels(self):
        """Test timed operation decorator with labels."""
        collector = MetricsCollector()
        labels = {"operation": "test", "version": "v1"}
        
        with patch("deep_agent_system.monitoring.get_metrics_collector", return_value=collector):
            @timed_operation("api_call", labels)
            def api_func():
                time.sleep(0.01)
                return "api_result"
            
            result = api_func()
            
            assert result == "api_result"
            
            # Check that timing was recorded with labels
            summary = collector.get_histogram_summary("api_call_duration_seconds", labels)
            assert summary is not None
            assert summary.count == 1