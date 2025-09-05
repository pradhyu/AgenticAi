"""Comprehensive unit tests for monitoring components."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from deep_agent_system.monitoring import (
    SystemMonitor,
    MetricsCollector,
    PerformanceTracker,
    HealthChecker,
    AlertManager,
    MetricType,
    AlertLevel,
    HealthStatus
)


class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    def test_metrics_collector_creation(self):
        """Test creating MetricsCollector."""
        collector = MetricsCollector()
        
        assert collector.metrics == {}
        assert isinstance(collector.start_time, datetime)
    
    def test_record_counter_metric(self):
        """Test recording counter metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("requests", 1, MetricType.COUNTER)
        collector.record_metric("requests", 2, MetricType.COUNTER)
        collector.record_metric("requests", 3, MetricType.COUNTER)
        
        metrics = collector.get_metrics()
        assert "requests" in metrics
        assert metrics["requests"]["value"] == 6  # Sum of counter values
        assert metrics["requests"]["type"] == MetricType.COUNTER
        assert metrics["requests"]["count"] == 3
    
    def test_record_gauge_metric(self):
        """Test recording gauge metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("cpu_usage", 50.0, MetricType.GAUGE)
        collector.record_metric("cpu_usage", 75.0, MetricType.GAUGE)
        collector.record_metric("cpu_usage", 60.0, MetricType.GAUGE)
        
        metrics = collector.get_metrics()
        assert "cpu_usage" in metrics
        assert metrics["cpu_usage"]["value"] == 60.0  # Latest gauge value
        assert metrics["cpu_usage"]["type"] == MetricType.GAUGE
        assert metrics["cpu_usage"]["count"] == 3
    
    def test_record_histogram_metric(self):
        """Test recording histogram metrics."""
        collector = MetricsCollector()
        
        values = [100, 200, 150, 300, 250]
        for value in values:
            collector.record_metric("response_time", value, MetricType.HISTOGRAM)
        
        metrics = collector.get_metrics()
        assert "response_time" in metrics
        
        histogram_data = metrics["response_time"]
        assert histogram_data["type"] == MetricType.HISTOGRAM
        assert histogram_data["count"] == 5
        assert histogram_data["sum"] == sum(values)
        assert histogram_data["min"] == min(values)
        assert histogram_data["max"] == max(values)
        assert histogram_data["avg"] == sum(values) / len(values)
    
    def test_record_metric_with_labels(self):
        """Test recording metrics with labels."""
        collector = MetricsCollector()
        
        collector.record_metric(
            "requests", 
            1, 
            MetricType.COUNTER, 
            labels={"method": "GET", "status": "200"}
        )
        collector.record_metric(
            "requests", 
            1, 
            MetricType.COUNTER, 
            labels={"method": "POST", "status": "201"}
        )
        
        metrics = collector.get_metrics()
        
        # Should create separate metric entries for different label combinations
        assert len([k for k in metrics.keys() if k.startswith("requests")]) >= 1
    
    def test_get_metric_by_name(self):
        """Test getting specific metric by name."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42, MetricType.GAUGE)
        
        metric = collector.get_metric("test_metric")
        assert metric is not None
        assert metric["value"] == 42
        assert metric["type"] == MetricType.GAUGE
        
        # Non-existent metric should return None
        assert collector.get_metric("non_existent") is None
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("metric1", 1, MetricType.COUNTER)
        collector.record_metric("metric2", 2, MetricType.GAUGE)
        
        assert len(collector.get_metrics()) == 2
        
        collector.clear_metrics()
        
        assert len(collector.get_metrics()) == 0
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()
        
        collector.record_metric("counter1", 5, MetricType.COUNTER)
        collector.record_metric("gauge1", 10.5, MetricType.GAUGE)
        collector.record_metric("histogram1", 100, MetricType.HISTOGRAM)
        collector.record_metric("histogram1", 200, MetricType.HISTOGRAM)
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_metrics"] == 3
        assert summary["counter_metrics"] == 1
        assert summary["gauge_metrics"] == 1
        assert summary["histogram_metrics"] == 1
        assert isinstance(summary["collection_duration"], float)


class TestPerformanceTracker:
    """Test cases for PerformanceTracker."""
    
    def test_performance_tracker_creation(self):
        """Test creating PerformanceTracker."""
        tracker = PerformanceTracker()
        
        assert tracker.active_operations == {}
        assert isinstance(tracker.metrics_collector, MetricsCollector)
    
    def test_start_operation_tracking(self):
        """Test starting operation tracking."""
        tracker = PerformanceTracker()
        
        operation_id = tracker.start_operation("test_operation")
        
        assert operation_id in tracker.active_operations
        assert tracker.active_operations[operation_id]["operation_name"] == "test_operation"
        assert isinstance(tracker.active_operations[operation_id]["start_time"], datetime)
    
    def test_end_operation_tracking(self):
        """Test ending operation tracking."""
        tracker = PerformanceTracker()
        
        operation_id = tracker.start_operation("test_operation")
        
        # Add small delay to ensure measurable duration
        time.sleep(0.01)
        
        duration = tracker.end_operation(operation_id)
        
        assert operation_id not in tracker.active_operations
        assert duration > 0
        
        # Should record metrics
        metrics = tracker.metrics_collector.get_metrics()
        assert "operation_duration_test_operation" in metrics
        assert "operation_count_test_operation" in metrics
    
    def test_end_non_existent_operation(self):
        """Test ending non-existent operation."""
        tracker = PerformanceTracker()
        
        # Should return None for non-existent operation
        duration = tracker.end_operation("non_existent_id")
        assert duration is None
    
    def test_context_manager_tracking(self):
        """Test using performance tracker as context manager."""
        tracker = PerformanceTracker()
        
        with tracker.track_operation("context_operation") as operation_id:
            assert operation_id in tracker.active_operations
            time.sleep(0.01)
        
        # Operation should be automatically ended
        assert operation_id not in tracker.active_operations
        
        # Should have recorded metrics
        metrics = tracker.metrics_collector.get_metrics()
        assert "operation_duration_context_operation" in metrics
    
    def test_get_active_operations(self):
        """Test getting active operations."""
        tracker = PerformanceTracker()
        
        op1 = tracker.start_operation("operation1")
        op2 = tracker.start_operation("operation2")
        
        active_ops = tracker.get_active_operations()
        
        assert len(active_ops) == 2
        assert op1 in active_ops
        assert op2 in active_ops
        assert active_ops[op1]["operation_name"] == "operation1"
        assert active_ops[op2]["operation_name"] == "operation2"
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        tracker = PerformanceTracker()
        
        # Track some operations
        op1 = tracker.start_operation("fast_op")
        time.sleep(0.01)
        tracker.end_operation(op1)
        
        op2 = tracker.start_operation("slow_op")
        time.sleep(0.02)
        tracker.end_operation(op2)
        
        summary = tracker.get_performance_summary()
        
        assert "total_operations" in summary
        assert "active_operations" in summary
        assert "average_duration" in summary
        assert summary["total_operations"] >= 2
        assert summary["active_operations"] == 0


class TestHealthChecker:
    """Test cases for HealthChecker."""
    
    def test_health_checker_creation(self):
        """Test creating HealthChecker."""
        checker = HealthChecker()
        
        assert checker.health_checks == {}
        assert checker.last_check_time is None
    
    def test_register_health_check(self):
        """Test registering health check."""
        checker = HealthChecker()
        
        def dummy_check():
            return True, "All good"
        
        checker.register_health_check("test_check", dummy_check)
        
        assert "test_check" in checker.health_checks
        assert checker.health_checks["test_check"] == dummy_check
    
    def test_unregister_health_check(self):
        """Test unregistering health check."""
        checker = HealthChecker()
        
        def dummy_check():
            return True, "All good"
        
        checker.register_health_check("test_check", dummy_check)
        assert "test_check" in checker.health_checks
        
        checker.unregister_health_check("test_check")
        assert "test_check" not in checker.health_checks
    
    def test_run_health_checks_all_healthy(self):
        """Test running health checks when all are healthy."""
        checker = HealthChecker()
        
        def healthy_check1():
            return True, "Service 1 OK"
        
        def healthy_check2():
            return True, "Service 2 OK"
        
        checker.register_health_check("service1", healthy_check1)
        checker.register_health_check("service2", healthy_check2)
        
        status, results = checker.run_health_checks()
        
        assert status == HealthStatus.HEALTHY
        assert len(results) == 2
        assert results["service1"]["status"] == HealthStatus.HEALTHY
        assert results["service2"]["status"] == HealthStatus.HEALTHY
        assert results["service1"]["message"] == "Service 1 OK"
        assert results["service2"]["message"] == "Service 2 OK"
    
    def test_run_health_checks_with_unhealthy(self):
        """Test running health checks with some unhealthy services."""
        checker = HealthChecker()
        
        def healthy_check():
            return True, "Service OK"
        
        def unhealthy_check():
            return False, "Service down"
        
        checker.register_health_check("healthy_service", healthy_check)
        checker.register_health_check("unhealthy_service", unhealthy_check)
        
        status, results = checker.run_health_checks()
        
        assert status == HealthStatus.UNHEALTHY
        assert results["healthy_service"]["status"] == HealthStatus.HEALTHY
        assert results["unhealthy_service"]["status"] == HealthStatus.UNHEALTHY
        assert results["unhealthy_service"]["message"] == "Service down"
    
    def test_run_health_checks_with_exception(self):
        """Test running health checks when check function raises exception."""
        checker = HealthChecker()
        
        def failing_check():
            raise Exception("Check failed")
        
        checker.register_health_check("failing_service", failing_check)
        
        status, results = checker.run_health_checks()
        
        assert status == HealthStatus.UNHEALTHY
        assert results["failing_service"]["status"] == HealthStatus.UNHEALTHY
        assert "Check failed" in results["failing_service"]["message"]
    
    def test_get_health_status(self):
        """Test getting current health status."""
        checker = HealthChecker()
        
        def healthy_check():
            return True, "OK"
        
        checker.register_health_check("test_service", healthy_check)
        
        # Before running checks
        status = checker.get_health_status()
        assert status == HealthStatus.UNKNOWN
        
        # After running checks
        checker.run_health_checks()
        status = checker.get_health_status()
        assert status == HealthStatus.HEALTHY


class TestAlertManager:
    """Test cases for AlertManager."""
    
    def test_alert_manager_creation(self):
        """Test creating AlertManager."""
        manager = AlertManager()
        
        assert manager.alert_handlers == {}
        assert manager.alert_history == []
        assert manager.alert_rules == {}
    
    def test_register_alert_handler(self):
        """Test registering alert handler."""
        manager = AlertManager()
        
        def dummy_handler(alert):
            pass
        
        manager.register_alert_handler(AlertLevel.ERROR, dummy_handler)
        
        assert AlertLevel.ERROR in manager.alert_handlers
        assert dummy_handler in manager.alert_handlers[AlertLevel.ERROR]
    
    def test_unregister_alert_handler(self):
        """Test unregistering alert handler."""
        manager = AlertManager()
        
        def dummy_handler(alert):
            pass
        
        manager.register_alert_handler(AlertLevel.ERROR, dummy_handler)
        assert dummy_handler in manager.alert_handlers[AlertLevel.ERROR]
        
        manager.unregister_alert_handler(AlertLevel.ERROR, dummy_handler)
        assert dummy_handler not in manager.alert_handlers[AlertLevel.ERROR]
    
    def test_send_alert(self):
        """Test sending alert."""
        manager = AlertManager()
        
        handler_calls = []
        
        def test_handler(alert):
            handler_calls.append(alert)
        
        manager.register_alert_handler(AlertLevel.WARNING, test_handler)
        
        manager.send_alert(
            level=AlertLevel.WARNING,
            message="Test alert",
            source="test_component",
            metadata={"key": "value"}
        )
        
        # Check alert was handled
        assert len(handler_calls) == 1
        alert = handler_calls[0]
        assert alert["level"] == AlertLevel.WARNING
        assert alert["message"] == "Test alert"
        assert alert["source"] == "test_component"
        assert alert["metadata"] == {"key": "value"}
        
        # Check alert was added to history
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0] == alert
    
    def test_add_alert_rule(self):
        """Test adding alert rule."""
        manager = AlertManager()
        
        def test_condition(metrics):
            return metrics.get("cpu_usage", 0) > 80
        
        manager.add_alert_rule(
            rule_id="high_cpu",
            condition=test_condition,
            level=AlertLevel.WARNING,
            message="High CPU usage detected"
        )
        
        assert "high_cpu" in manager.alert_rules
        rule = manager.alert_rules["high_cpu"]
        assert rule["condition"] == test_condition
        assert rule["level"] == AlertLevel.WARNING
        assert rule["message"] == "High CPU usage detected"
    
    def test_check_alert_rules(self):
        """Test checking alert rules against metrics."""
        manager = AlertManager()
        
        alert_sent = []
        
        def test_handler(alert):
            alert_sent.append(alert)
        
        manager.register_alert_handler(AlertLevel.WARNING, test_handler)
        
        # Add rule that triggers on high CPU
        def high_cpu_condition(metrics):
            return metrics.get("cpu_usage", 0) > 80
        
        manager.add_alert_rule(
            rule_id="high_cpu",
            condition=high_cpu_condition,
            level=AlertLevel.WARNING,
            message="High CPU usage: {cpu_usage}%"
        )
        
        # Test with low CPU (should not trigger)
        metrics = {"cpu_usage": 50}
        manager.check_alert_rules(metrics)
        assert len(alert_sent) == 0
        
        # Test with high CPU (should trigger)
        metrics = {"cpu_usage": 90}
        manager.check_alert_rules(metrics)
        assert len(alert_sent) == 1
        assert "High CPU usage: 90%" in alert_sent[0]["message"]
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = AlertManager()
        
        # Send some alerts
        manager.send_alert(AlertLevel.INFO, "Info alert", "component1")
        manager.send_alert(AlertLevel.WARNING, "Warning alert", "component2")
        manager.send_alert(AlertLevel.ERROR, "Error alert", "component3")
        
        # Get all history
        history = manager.get_alert_history()
        assert len(history) == 3
        
        # Get filtered history
        warning_history = manager.get_alert_history(level=AlertLevel.WARNING)
        assert len(warning_history) == 1
        assert warning_history[0]["level"] == AlertLevel.WARNING
        
        # Get recent history
        recent_history = manager.get_alert_history(limit=2)
        assert len(recent_history) == 2
    
    def test_clear_alert_history(self):
        """Test clearing alert history."""
        manager = AlertManager()
        
        manager.send_alert(AlertLevel.INFO, "Test alert", "test")
        assert len(manager.alert_history) == 1
        
        manager.clear_alert_history()
        assert len(manager.alert_history) == 0


class TestSystemMonitor:
    """Test cases for SystemMonitor."""
    
    def test_system_monitor_creation(self):
        """Test creating SystemMonitor."""
        monitor = SystemMonitor()
        
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        assert isinstance(monitor.performance_tracker, PerformanceTracker)
        assert isinstance(monitor.health_checker, HealthChecker)
        assert isinstance(monitor.alert_manager, AlertManager)
        assert monitor.monitoring_active is False
    
    def test_start_monitoring(self):
        """Test starting monitoring."""
        monitor = SystemMonitor()
        
        monitor.start_monitoring()
        
        assert monitor.monitoring_active is True
        
        # Clean up
        monitor.stop_monitoring()
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        monitor = SystemMonitor()
        
        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
    
    def test_record_metric(self):
        """Test recording metric through system monitor."""
        monitor = SystemMonitor()
        
        monitor.record_metric("test_metric", 42, MetricType.GAUGE)
        
        metrics = monitor.get_metrics()
        assert "test_metric" in metrics
        assert metrics["test_metric"]["value"] == 42
    
    def test_track_operation(self):
        """Test tracking operation through system monitor."""
        monitor = SystemMonitor()
        
        with monitor.track_operation("test_operation") as operation_id:
            time.sleep(0.01)
        
        metrics = monitor.get_metrics()
        assert "operation_duration_test_operation" in metrics
        assert "operation_count_test_operation" in metrics
    
    def test_register_health_check(self):
        """Test registering health check through system monitor."""
        monitor = SystemMonitor()
        
        def test_check():
            return True, "OK"
        
        monitor.register_health_check("test_service", test_check)
        
        status, results = monitor.run_health_checks()
        assert status == HealthStatus.HEALTHY
        assert "test_service" in results
    
    def test_send_alert(self):
        """Test sending alert through system monitor."""
        monitor = SystemMonitor()
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.register_alert_handler(AlertLevel.INFO, alert_handler)
        monitor.send_alert(AlertLevel.INFO, "Test alert", "test_component")
        
        assert len(alerts_received) == 1
        assert alerts_received[0]["message"] == "Test alert"
    
    def test_get_system_status(self):
        """Test getting comprehensive system status."""
        monitor = SystemMonitor()
        
        # Add some test data
        monitor.record_metric("cpu_usage", 75.0, MetricType.GAUGE)
        monitor.record_metric("memory_usage", 60.0, MetricType.GAUGE)
        
        def test_health_check():
            return True, "Service OK"
        
        monitor.register_health_check("test_service", test_health_check)
        
        status = monitor.get_system_status()
        
        assert "metrics" in status
        assert "health" in status
        assert "performance" in status
        assert "alerts" in status
        assert "monitoring_active" in status
        
        assert status["metrics"]["cpu_usage"]["value"] == 75.0
        assert status["health"]["overall_status"] == HealthStatus.HEALTHY
        assert status["monitoring_active"] is False


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow."""
        monitor = SystemMonitor()
        
        # Set up alert handler
        alerts = []
        
        def alert_handler(alert):
            alerts.append(alert)
        
        monitor.register_alert_handler(AlertLevel.WARNING, alert_handler)
        
        # Add alert rule for high CPU
        def high_cpu_rule(metrics):
            cpu_metric = metrics.get("cpu_usage")
            if cpu_metric:
                return cpu_metric["value"] > 80
            return False
        
        monitor.add_alert_rule(
            "high_cpu",
            high_cpu_rule,
            AlertLevel.WARNING,
            "High CPU usage detected"
        )
        
        # Record normal CPU usage (should not trigger alert)
        monitor.record_metric("cpu_usage", 50.0, MetricType.GAUGE)
        monitor.check_alert_rules()
        assert len(alerts) == 0
        
        # Record high CPU usage (should trigger alert)
        monitor.record_metric("cpu_usage", 90.0, MetricType.GAUGE)
        monitor.check_alert_rules()
        assert len(alerts) == 1
        assert "High CPU usage detected" in alerts[0]["message"]
    
    def test_concurrent_monitoring_operations(self):
        """Test monitoring under concurrent operations."""
        monitor = SystemMonitor()
        
        def worker_function(worker_id):
            for i in range(10):
                with monitor.track_operation(f"worker_{worker_id}_operation"):
                    time.sleep(0.001)
                    monitor.record_metric(f"worker_{worker_id}_counter", 1, MetricType.COUNTER)
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify metrics were recorded correctly
        metrics = monitor.get_metrics()
        
        # Should have counter metrics for each worker
        for i in range(5):
            counter_key = f"worker_{i}_counter"
            assert counter_key in metrics
            assert metrics[counter_key]["value"] == 10  # 10 increments per worker
        
        # Should have operation duration metrics
        operation_metrics = [k for k in metrics.keys() if "operation_duration" in k]
        assert len(operation_metrics) >= 5  # At least one per worker


if __name__ == "__main__":
    pytest.main([__file__])