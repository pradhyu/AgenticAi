"""Logging configuration and setup for the Deep Agent System."""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from deep_agent_system.exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process and thread info
        log_data["process_id"] = record.process
        log_data["thread_id"] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            # Get all extra attributes (those not in standard LogRecord)
            standard_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'extra'
            }
            
            extra_data = {}
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith('_'):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_data[key] = value
                    except (TypeError, ValueError):
                        extra_data[key] = str(value)
            
            if extra_data:
                log_data["extra"] = extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class AgentContextFilter(logging.Filter):
    """Filter to add agent context to log records."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the agent context filter.
        
        Args:
            agent_id: ID of the agent to add to log records
        """
        super().__init__()
        self.agent_id = agent_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add agent context to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record to be logged
        """
        if self.agent_id:
            record.agent_id = self.agent_id
        
        # Add component context based on logger name
        logger_parts = record.name.split('.')
        if len(logger_parts) >= 2:
            record.component = logger_parts[1]  # e.g., 'agents', 'rag', 'communication'
        
        return True


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        """Initialize the performance filter."""
        super().__init__()
        self._start_times: Dict[str, float] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record to be logged
        """
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
            record.cpu_percent = process.cpu_percent()
        except (ImportError, Exception):
            pass
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    agent_id: Optional[str] = None,
) -> None:
    """Set up logging configuration for the Deep Agent System.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('structured' for JSON, 'simple' for text)
        log_file: Optional log file name
        log_dir: Optional log directory (defaults to 'logs')
        enable_console: Whether to enable console logging
        agent_id: Optional agent ID to include in logs
        
    Raises:
        ConfigurationError: If logging configuration is invalid
    """
    try:
        # Validate log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ConfigurationError(
                f"Invalid log level: {log_level}",
                config_key="log_level",
            )
        
        # Create log directory if needed
        if log_file or log_dir:
            log_directory = Path(log_dir or "logs")
            log_directory.mkdir(exist_ok=True)
        
        # Configure formatters
        formatters = {}
        
        if log_format == "structured":
            formatters["structured"] = {
                "()": "deep_agent_system.logging_config.StructuredFormatter",
                "include_extra": True,
            }
        else:
            formatters["simple"] = {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        
        # Configure filters
        filters = {}
        
        if agent_id:
            filters["agent_context"] = {
                "()": "deep_agent_system.logging_config.AgentContextFilter",
                "agent_id": agent_id,
            }
        
        filters["performance"] = {
            "()": "deep_agent_system.logging_config.PerformanceFilter",
        }
        
        # Configure handlers
        handlers = {}
        
        if enable_console:
            handlers["console"] = {
                "class": "logging.StreamHandler",
                "level": log_level.upper(),
                "formatter": "structured" if log_format == "structured" else "simple",
                "stream": "ext://sys.stdout",
                "filters": list(filters.keys()),
            }
        
        if log_file:
            log_path = Path(log_dir or "logs") / log_file
            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level.upper(),
                "formatter": "structured" if log_format == "structured" else "simple",
                "filename": str(log_path),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "filters": list(filters.keys()),
            }
        
        # Configure loggers
        loggers = {
            "deep_agent_system": {
                "level": log_level.upper(),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            # Reduce noise from external libraries
            "langchain": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "chromadb": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "neo4j": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
        }
        
        # Build logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "filters": filters,
            "handlers": handlers,
            "loggers": loggers,
            "root": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
            },
        }
        
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Log successful configuration
        logger = logging.getLogger("deep_agent_system.logging")
        logger.info(
            "Logging configured successfully",
            extra={
                "log_level": log_level,
                "log_format": log_format,
                "log_file": log_file,
                "enable_console": enable_console,
                "agent_id": agent_id,
            }
        )
        
    except Exception as e:
        raise ConfigurationError(
            f"Failed to configure logging: {e}",
            config_key="logging",
            cause=e,
        )


def get_logger(name: str, agent_id: Optional[str] = None) -> logging.Logger:
    """Get a logger with optional agent context.
    
    Args:
        name: Logger name
        agent_id: Optional agent ID to include in logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add agent context filter if agent_id is provided
    if agent_id:
        # Check if agent context filter already exists
        has_agent_filter = any(
            isinstance(f, AgentContextFilter) and f.agent_id == agent_id
            for f in logger.filters
        )
        
        if not has_agent_filter:
            agent_filter = AgentContextFilter(agent_id)
            logger.addFilter(agent_filter)
    
    return logger


def configure_from_env() -> None:
    """Configure logging from environment variables."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "structured")
    log_file = os.getenv("LOG_FILE")
    log_dir = os.getenv("LOG_DIR", "logs")
    enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"
    
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        log_dir=log_dir,
        enable_console=enable_console,
    )


# Context managers for structured logging

class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize log context.
        
        Args:
            logger: Logger to add context to
            **context: Context fields to add to log records
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter the log context."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the log context."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


class TimedOperation:
    """Context manager for timing operations and logging performance."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        log_level: int = logging.INFO,
        **context
    ):
        """Initialize timed operation.
        
        Args:
            logger: Logger to use for timing logs
            operation_name: Name of the operation being timed
            log_level: Log level for timing messages
            **context: Additional context to include in logs
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Start timing the operation."""
        import time
        self.start_time = time.time()
        
        self.logger.log(
            self.log_level,
            f"Starting operation: {self.operation_name}",
            extra={
                "operation": self.operation_name,
                "phase": "start",
                **self.context,
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing the operation."""
        import time
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        status = "failed" if exc_type else "completed"
        
        self.logger.log(
            self.log_level,
            f"Operation {status}: {self.operation_name} (duration: {duration:.3f}s)",
            extra={
                "operation": self.operation_name,
                "phase": "end",
                "status": status,
                "duration_seconds": duration,
                "exception_type": exc_type.__name__ if exc_type else None,
                **self.context,
            }
        )


# Utility functions for common logging patterns

def log_agent_action(
    logger: logging.Logger,
    agent_id: str,
    action: str,
    **context
) -> None:
    """Log an agent action with structured context.
    
    Args:
        logger: Logger to use
        agent_id: ID of the agent performing the action
        action: Description of the action
        **context: Additional context fields
    """
    logger.info(
        f"Agent action: {action}",
        extra={
            "agent_id": agent_id,
            "action": action,
            "event_type": "agent_action",
            **context,
        }
    )


def log_message_flow(
    logger: logging.Logger,
    message_id: str,
    sender_id: str,
    recipient_id: str,
    message_type: str,
    phase: str,
    **context
) -> None:
    """Log message flow between agents.
    
    Args:
        logger: Logger to use
        message_id: ID of the message
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        message_type: Type of the message
        phase: Phase of message processing (sent, received, processed, etc.)
        **context: Additional context fields
    """
    logger.debug(
        f"Message {phase}: {message_id}",
        extra={
            "message_id": message_id,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "message_type": message_type,
            "phase": phase,
            "event_type": "message_flow",
            **context,
        }
    )


def log_rag_operation(
    logger: logging.Logger,
    operation: str,
    query: str,
    retrieval_type: str,
    result_count: int,
    duration: float,
    **context
) -> None:
    """Log RAG retrieval operations.
    
    Args:
        logger: Logger to use
        operation: Type of RAG operation
        query: Query that was processed
        retrieval_type: Type of retrieval (vector, graph, hybrid)
        result_count: Number of results retrieved
        duration: Duration of the operation in seconds
        **context: Additional context fields
    """
    logger.info(
        f"RAG {operation}: {result_count} results for query",
        extra={
            "operation": operation,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "retrieval_type": retrieval_type,
            "result_count": result_count,
            "duration_seconds": duration,
            "event_type": "rag_operation",
            **context,
        }
    )


def log_workflow_step(
    logger: logging.Logger,
    workflow_id: str,
    step_name: str,
    agent_id: str,
    status: str,
    **context
) -> None:
    """Log workflow step execution.
    
    Args:
        logger: Logger to use
        workflow_id: ID of the workflow
        step_name: Name of the workflow step
        agent_id: ID of the agent executing the step
        status: Status of the step (started, completed, failed)
        **context: Additional context fields
    """
    logger.info(
        f"Workflow step {status}: {step_name}",
        extra={
            "workflow_id": workflow_id,
            "step_name": step_name,
            "agent_id": agent_id,
            "status": status,
            "event_type": "workflow_step",
            **context,
        }
    )