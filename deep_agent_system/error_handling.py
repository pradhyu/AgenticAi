"""Error handling utilities and recovery strategies for the Deep Agent System."""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from deep_agent_system.exceptions import (
    AgentCommunicationError,
    AgentNotFoundError,
    DeepAgentException,
    MessageTimeoutError,
    RAGRetrievalError,
    WorkflowExecutionError,
)


logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry strategies."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: List of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            AgentCommunicationError,
            MessageTimeoutError,
            RAGRetrievalError,
            ConnectionError,
            TimeoutError,
        ]
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_attempts:
            return False
        
        # Check if exception type is retryable
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if attempt <= 1:
            return 0.0
        
        # Calculate exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 2))
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """Initialize circuit breaker configuration.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise DeepAgentException(
                    "Circuit breaker is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                )
            else:
                self.state = "half-open"
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker recovered and closed")
            
            return result
            
        except self.config.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
            
            raise e


class FallbackStrategy:
    """Base class for fallback strategies."""
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the fallback strategy.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Fallback result
        """
        raise NotImplementedError("Subclasses must implement execute method")


class DefaultValueFallback(FallbackStrategy):
    """Fallback strategy that returns a default value."""
    
    def __init__(self, default_value: Any):
        """Initialize with default value.
        
        Args:
            default_value: Value to return as fallback
        """
        self.default_value = default_value
    
    def execute(self, *args, **kwargs) -> Any:
        """Return the default value.
        
        Returns:
            Default value
        """
        return self.default_value


class FunctionFallback(FallbackStrategy):
    """Fallback strategy that calls an alternative function."""
    
    def __init__(self, fallback_func: Callable):
        """Initialize with fallback function.
        
        Args:
            fallback_func: Function to call as fallback
        """
        self.fallback_func = fallback_func
    
    def execute(self, *args, **kwargs) -> Any:
        """Call the fallback function.
        
        Args:
            *args: Positional arguments to pass to fallback function
            **kwargs: Keyword arguments to pass to fallback function
            
        Returns:
            Result of fallback function
        """
        return self.fallback_func(*args, **kwargs)


class ErrorHandler:
    """Centralized error handling with retry and fallback strategies."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ):
        """Initialize error handler.
        
        Args:
            retry_config: Configuration for retry behavior
            circuit_breaker_config: Configuration for circuit breaker
            fallback_strategy: Fallback strategy to use when all else fails
        """
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = (
            CircuitBreaker(circuit_breaker_config)
            if circuit_breaker_config
            else None
        )
        self.fallback_strategy = fallback_strategy
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or fallback value
            
        Raises:
            Exception: If all retry attempts fail and no fallback is available
        """
        last_exception = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                logger.warning(
                    f"Attempt {attempt}/{self.retry_config.max_attempts} failed: {e}"
                )
                
                # Check if we should retry
                if not self.retry_config.should_retry(e, attempt):
                    break
                
                # Wait before retry (except on last attempt)
                if attempt < self.retry_config.max_attempts:
                    delay = self.retry_config.get_delay(attempt)
                    if delay > 0:
                        logger.debug(f"Waiting {delay:.2f}s before retry")
                        time.sleep(delay)
        
        # All retries failed, try fallback
        if self.fallback_strategy:
            try:
                logger.info("Executing fallback strategy")
                return self.fallback_strategy.execute(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback strategy failed: {fallback_error}")
                # Raise original exception, not fallback error
        
        # No fallback or fallback failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise DeepAgentException("Function execution failed with no exception details")


class AsyncErrorHandler:
    """Async version of ErrorHandler."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ):
        """Initialize async error handler.
        
        Args:
            retry_config: Configuration for retry behavior
            fallback_strategy: Fallback strategy to use when all else fails
        """
        self.retry_config = retry_config or RetryConfig()
        self.fallback_strategy = fallback_strategy
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute an async function with error handling.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or fallback value
            
        Raises:
            Exception: If all retry attempts fail and no fallback is available
        """
        last_exception = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                logger.warning(
                    f"Async attempt {attempt}/{self.retry_config.max_attempts} failed: {e}"
                )
                
                # Check if we should retry
                if not self.retry_config.should_retry(e, attempt):
                    break
                
                # Wait before retry (except on last attempt)
                if attempt < self.retry_config.max_attempts:
                    delay = self.retry_config.get_delay(attempt)
                    if delay > 0:
                        logger.debug(f"Waiting {delay:.2f}s before async retry")
                        await asyncio.sleep(delay)
        
        # All retries failed, try fallback
        if self.fallback_strategy:
            try:
                logger.info("Executing async fallback strategy")
                return self.fallback_strategy.execute(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Async fallback strategy failed: {fallback_error}")
        
        # No fallback or fallback failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise DeepAgentException("Async function execution failed with no exception details")


# Decorator functions for easy error handling

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
):
    """Decorator to add retry behavior to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        retryable_exceptions: List of exception types that should trigger retries
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions,
        )
        error_handler = ErrorHandler(retry_config=retry_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return error_handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


def with_fallback(fallback_strategy: FallbackStrategy):
    """Decorator to add fallback behavior to a function.
    
    Args:
        fallback_strategy: Fallback strategy to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        error_handler = ErrorHandler(fallback_strategy=fallback_strategy)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return error_handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
):
    """Decorator to add circuit breaker behavior to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
        error_handler = ErrorHandler(circuit_breaker_config=circuit_breaker_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return error_handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Specific error recovery strategies for different components

class AgentCommunicationRecovery:
    """Recovery strategies for agent communication failures."""
    
    @staticmethod
    def create_retry_config() -> RetryConfig:
        """Create retry configuration for agent communication.
        
        Returns:
            RetryConfig optimized for agent communication
        """
        return RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            retryable_exceptions=[
                AgentCommunicationError,
                MessageTimeoutError,
                ConnectionError,
            ],
        )
    
    @staticmethod
    def create_fallback_strategy(alternative_agents: List[str]) -> FallbackStrategy:
        """Create fallback strategy for agent communication.
        
        Args:
            alternative_agents: List of alternative agent IDs to try
            
        Returns:
            FallbackStrategy for agent communication
        """
        def fallback_func(*args, **kwargs):
            # This would be implemented by the communication manager
            # to try alternative agents
            raise NotImplementedError("Fallback to alternative agents not implemented")
        
        return FunctionFallback(fallback_func)


class RAGRetrievalRecovery:
    """Recovery strategies for RAG retrieval failures."""
    
    @staticmethod
    def create_retry_config() -> RetryConfig:
        """Create retry configuration for RAG retrieval.
        
        Returns:
            RetryConfig optimized for RAG retrieval
        """
        return RetryConfig(
            max_attempts=2,
            base_delay=1.0,
            max_delay=10.0,
            retryable_exceptions=[
                RAGRetrievalError,
                ConnectionError,
                TimeoutError,
            ],
        )
    
    @staticmethod
    def create_fallback_strategy(default_context: str = "") -> FallbackStrategy:
        """Create fallback strategy for RAG retrieval.
        
        Args:
            default_context: Default context to return when retrieval fails
            
        Returns:
            FallbackStrategy for RAG retrieval
        """
        return DefaultValueFallback(default_context)


class WorkflowRecovery:
    """Recovery strategies for workflow execution failures."""
    
    @staticmethod
    def create_retry_config() -> RetryConfig:
        """Create retry configuration for workflow execution.
        
        Returns:
            RetryConfig optimized for workflow execution
        """
        return RetryConfig(
            max_attempts=2,
            base_delay=2.0,
            max_delay=30.0,
            retryable_exceptions=[
                WorkflowExecutionError,
                AgentCommunicationError,
            ],
        )
    
    @staticmethod
    def create_circuit_breaker_config() -> CircuitBreakerConfig:
        """Create circuit breaker configuration for workflow execution.
        
        Returns:
            CircuitBreakerConfig for workflow execution
        """
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=120.0,
            expected_exception=WorkflowExecutionError,
        )