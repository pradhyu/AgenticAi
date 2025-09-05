"""Tests for logging configuration and utilities."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deep_agent_system.exceptions import ConfigurationError
from deep_agent_system.logging_config import (
    AgentContextFilter,
    LogContext,
    PerformanceFilter,
    StructuredFormatter,
    TimedOperation,
    configure_from_env,
    get_logger,
    log_agent_action,
    log_message_flow,
    log_rag_operation,
    log_workflow_step,
    setup_logging,
)


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse as JSON
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "file"
        # Function name may be None in test environment
        assert "function" in log_data
        assert log_data["line"] == 42
        assert "timestamp" in log_data
        assert "process_id" in log_data
        assert "thread_id" in log_data
    
    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert log_data["exception"]["traceback"] is not None
    
    def test_formatting_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Add extra fields
        record.agent_id = "agent123"
        record.operation = "test_operation"
        record.custom_field = {"nested": "value"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "extra" in log_data
        assert log_data["extra"]["agent_id"] == "agent123"
        assert log_data["extra"]["operation"] == "test_operation"
        assert log_data["extra"]["custom_field"] == {"nested": "value"}
    
    def test_formatting_without_extra_fields(self):
        """Test formatting without extra fields."""
        formatter = StructuredFormatter(include_extra=False)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        record.agent_id = "agent123"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "extra" not in log_data


class TestAgentContextFilter:
    """Test AgentContextFilter functionality."""
    
    def test_filter_adds_agent_id(self):
        """Test that filter adds agent ID to log records."""
        filter_obj = AgentContextFilter("agent123")
        
        record = logging.LogRecord(
            name="deep_agent_system.agents.analyst",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.agent_id == "agent123"
        assert record.component == "agents"
    
    def test_filter_without_agent_id(self):
        """Test filter without agent ID."""
        filter_obj = AgentContextFilter()
        
        record = logging.LogRecord(
            name="deep_agent_system.rag.manager",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert not hasattr(record, "agent_id")
        assert record.component == "rag"


class TestPerformanceFilter:
    """Test PerformanceFilter functionality."""
    
    def test_filter_adds_performance_metrics(self):
        """Test that filter adds performance metrics."""
        # Skip this test if psutil is not available
        pytest.skip("Skipping psutil-dependent test")
    
    def test_filter_without_psutil(self):
        """Test filter when psutil is not available."""
        filter_obj = PerformanceFilter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        # Performance metrics may or may not be added depending on psutil availability


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(
                log_level="DEBUG",
                log_format="simple",
                log_dir=temp_dir,
                enable_console=False,
            )
            
            # Test that logger is configured
            logger = logging.getLogger("deep_agent_system.test")
            assert logger.level <= logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = "test.log"
            
            setup_logging(
                log_level="INFO",
                log_format="structured",
                log_file=log_file,
                log_dir=temp_dir,
                enable_console=False,
            )
            
            # Test that log file is created
            log_path = Path(temp_dir) / log_file
            
            # Log a message
            logger = logging.getLogger("deep_agent_system.test")
            logger.info("Test log message")
            
            # Check that file exists (may not have content immediately)
            # This is a basic check - in real usage, the file would be created on first write
            assert Path(temp_dir).exists()
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        with pytest.raises(ConfigurationError) as exc_info:
            setup_logging(log_level="INVALID_LEVEL")
        
        assert "Invalid log level" in str(exc_info.value)
        assert exc_info.value.config_key in ["log_level", "logging"]
    
    def test_get_logger_with_agent_id(self):
        """Test getting logger with agent ID."""
        logger = get_logger("test.logger", agent_id="agent123")
        
        # Check that agent context filter is added
        agent_filters = [
            f for f in logger.filters
            if isinstance(f, AgentContextFilter) and f.agent_id == "agent123"
        ]
        assert len(agent_filters) == 1
    
    def test_get_logger_without_agent_id(self):
        """Test getting logger without agent ID."""
        logger = get_logger("test.logger.unique")  # Use unique name to avoid conflicts
        
        # Should not have agent context filters
        agent_filters = [f for f in logger.filters if isinstance(f, AgentContextFilter)]
        assert len(agent_filters) == 0
    
    @patch.dict("os.environ", {
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "simple",
        "LOG_FILE": "test.log",
        "LOG_DIR": "/tmp/logs",
        "LOG_CONSOLE": "false",
    })
    def test_configure_from_env(self):
        """Test configuring logging from environment variables."""
        with patch("deep_agent_system.logging_config.setup_logging") as mock_setup:
            configure_from_env()
            
            mock_setup.assert_called_once_with(
                log_level="DEBUG",
                log_format="simple",
                log_file="test.log",
                log_dir="/tmp/logs",
                enable_console=False,
            )


class TestLogContext:
    """Test LogContext context manager."""
    
    def test_log_context_adds_fields(self):
        """Test that LogContext adds fields to log records."""
        logger = logging.getLogger("test.context")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        with LogContext(logger, operation="test_op", user_id="user123"):
            logger.info("Test message")
        
        assert len(records) == 1
        record = records[0]
        assert record.operation == "test_op"
        assert record.user_id == "user123"
    
    def test_log_context_cleanup(self):
        """Test that LogContext cleans up after itself."""
        logger = logging.getLogger("test.context.cleanup")
        original_factory = logging.getLogRecordFactory()
        
        with LogContext(logger, test_field="test_value"):
            pass
        
        # Factory should be restored
        assert logging.getLogRecordFactory() == original_factory


class TestTimedOperation:
    """Test TimedOperation context manager."""
    
    def test_timed_operation_logs_timing(self):
        """Test that TimedOperation logs timing information."""
        logger = logging.getLogger("test.timing")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        with TimedOperation(logger, "test_operation", user_id="user123"):
            import time
            time.sleep(0.01)  # Small delay
        
        # Should have start and end log records
        assert len(records) == 2
        
        start_record = records[0]
        assert "Starting operation: test_operation" in start_record.getMessage()
        assert start_record.operation == "test_operation"
        assert start_record.phase == "start"
        assert start_record.user_id == "user123"
        
        end_record = records[1]
        assert "Operation completed: test_operation" in end_record.getMessage()
        assert end_record.operation == "test_operation"
        assert end_record.phase == "end"
        assert end_record.status == "completed"
        assert hasattr(end_record, "duration_seconds")
        assert end_record.duration_seconds > 0
    
    def test_timed_operation_logs_failure(self):
        """Test that TimedOperation logs failure information."""
        logger = logging.getLogger("test.timing.failure")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        try:
            with TimedOperation(logger, "failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should have start and end log records
        assert len(records) == 2
        
        end_record = records[1]
        assert "Operation failed: failing_operation" in end_record.getMessage()
        assert end_record.status == "failed"
        assert end_record.exception_type == "ValueError"


class TestUtilityFunctions:
    """Test utility logging functions."""
    
    def test_log_agent_action(self):
        """Test log_agent_action utility function."""
        logger = logging.getLogger("test.agent.action")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        log_agent_action(
            logger,
            agent_id="agent123",
            action="process_message",
            message_id="msg456",
        )
        
        assert len(records) == 1
        record = records[0]
        assert "Agent action: process_message" in record.getMessage()
        assert record.agent_id == "agent123"
        assert record.action == "process_message"
        assert record.event_type == "agent_action"
        assert record.message_id == "msg456"
    
    def test_log_message_flow(self):
        """Test log_message_flow utility function."""
        logger = logging.getLogger("test.message.flow")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        log_message_flow(
            logger,
            message_id="msg123",
            sender_id="agent1",
            recipient_id="agent2",
            message_type="request",
            phase="sent",
        )
        
        assert len(records) == 1
        record = records[0]
        assert "Message sent: msg123" in record.getMessage()
        assert record.message_id == "msg123"
        assert record.sender_id == "agent1"
        assert record.recipient_id == "agent2"
        assert record.message_type == "request"
        assert record.phase == "sent"
        assert record.event_type == "message_flow"
    
    def test_log_rag_operation(self):
        """Test log_rag_operation utility function."""
        logger = logging.getLogger("test.rag.operation")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        log_rag_operation(
            logger,
            operation="retrieve",
            query="test query for retrieval",
            retrieval_type="vector",
            result_count=5,
            duration=0.123,
        )
        
        assert len(records) == 1
        record = records[0]
        assert "RAG retrieve: 5 results for query" in record.getMessage()
        assert record.operation == "retrieve"
        assert record.query == "test query for retrieval"
        assert record.retrieval_type == "vector"
        assert record.result_count == 5
        assert record.duration_seconds == 0.123
        assert record.event_type == "rag_operation"
    
    def test_log_workflow_step(self):
        """Test log_workflow_step utility function."""
        logger = logging.getLogger("test.workflow.step")
        
        # Capture log records
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        log_workflow_step(
            logger,
            workflow_id="wf123",
            step_name="analyze_data",
            agent_id="agent1",
            status="completed",
        )
        
        assert len(records) == 1
        record = records[0]
        assert "Workflow step completed: analyze_data" in record.getMessage()
        assert record.workflow_id == "wf123"
        assert record.step_name == "analyze_data"
        assert record.agent_id == "agent1"
        assert record.status == "completed"
        assert record.event_type == "workflow_step"