"""Tests for error handling utilities and recovery strategies."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from deep_agent_system.error_handling import (
    AsyncErrorHandler,
    CircuitBreaker,
    CircuitBreakerConfig,
    DefaultValueFallback,
    ErrorHandler,
    FunctionFallback,
    RetryConfig,
    with_circuit_breaker,
    with_fallback,
    with_retry,
)
from deep_agent_system.exceptions import (
    AgentCommunicationError,
    DeepAgentException,
    MessageTimeoutError,
    RAGRetrievalError,
)


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert AgentCommunicationError in config.retryable_exceptions
    
    def test_should_retry_with_retryable_exception(self):
        """Test should_retry with retryable exception."""
        config = RetryConfig(max_attempts=3)
        
        # Should retry on first attempt with retryable exception
        assert config.should_retry(AgentCommunicationError("test"), 1) is True
        assert config.should_retry(MessageTimeoutError(5.0), 2) is True
        
        # Should not retry after max attempts
        assert config.should_retry(AgentCommunicationError("test"), 3) is False
    
    def test_should_retry_with_non_retryable_exception(self):
        """Test should_retry with non-retryable exception."""
        config = RetryConfig(retryable_exceptions=[ValueError])
        
        # Should not retry with non-retryable exception
        assert config.should_retry(TypeError("test"), 1) is False
        assert config.should_retry(ValueError("test"), 1) is True
    
    def test_get_delay_calculation(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.get_delay(1) == 0.0  # No delay on first attempt
        assert config.get_delay(2) == 1.0  # Base delay on second attempt
        assert config.get_delay(3) == 2.0  # Exponential backoff
        assert config.get_delay(4) == 4.0
    
    def test_get_delay_with_max_delay(self):
        """Test delay calculation with max delay limit."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential_base=2.0, jitter=False)
        
        assert config.get_delay(2) == 10.0
        assert config.get_delay(3) == 15.0  # Capped at max_delay
        assert config.get_delay(4) == 15.0  # Still capped


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        # Should allow calls in closed state
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == "closed"
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2, expected_exception=ValueError)
        breaker = CircuitBreaker(config)
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        assert breaker.state == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        assert breaker.state == "open"
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1.0)
        breaker = CircuitBreaker(config)
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        
        # Should reject calls in open state
        with pytest.raises(DeepAgentException) as exc_info:
            breaker.call(lambda: "success")
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        assert breaker.state == "open"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should allow call and close circuit on success
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == "closed"


class TestFallbackStrategies:
    """Test fallback strategy implementations."""
    
    def test_default_value_fallback(self):
        """Test DefaultValueFallback strategy."""
        fallback = DefaultValueFallback("default_value")
        
        result = fallback.execute()
        assert result == "default_value"
    
    def test_function_fallback(self):
        """Test FunctionFallback strategy."""
        def fallback_func(x, y=None):
            return f"fallback_{x}_{y}"
        
        fallback = FunctionFallback(fallback_func)
        
        result = fallback.execute("test", y="value")
        assert result == "fallback_test_value"


class TestErrorHandler:
    """Test ErrorHandler functionality."""
    
    def test_successful_execution(self):
        """Test successful function execution."""
        handler = ErrorHandler()
        
        result = handler.execute(lambda: "success")
        assert result == "success"
    
    def test_retry_on_failure(self):
        """Test retry behavior on failure."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentCommunicationError("test error")
            return "success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = ErrorHandler(retry_config=config)
        
        result = handler.execute(failing_func)
        assert result == "success"
        assert call_count == 3
    
    def test_fallback_on_failure(self):
        """Test fallback execution on failure."""
        def failing_func():
            raise AgentCommunicationError("test error")
        
        fallback = DefaultValueFallback("fallback_result")
        config = RetryConfig(max_attempts=1)
        handler = ErrorHandler(retry_config=config, fallback_strategy=fallback)
        
        result = handler.execute(failing_func)
        assert result == "fallback_result"
    
    def test_no_retry_for_non_retryable_exception(self):
        """Test no retry for non-retryable exceptions."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("non-retryable error")
        
        config = RetryConfig(max_attempts=3, retryable_exceptions=[ValueError])
        handler = ErrorHandler(retry_config=config)
        
        with pytest.raises(TypeError):
            handler.execute(failing_func)
        
        assert call_count == 1  # Should not retry


class TestAsyncErrorHandler:
    """Test AsyncErrorHandler functionality."""
    
    @pytest.mark.asyncio
    async def test_async_successful_execution(self):
        """Test successful async function execution."""
        handler = AsyncErrorHandler()
        
        async def async_func():
            return "async_success"
        
        result = await handler.execute(async_func)
        assert result == "async_success"
    
    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Test async retry behavior on failure."""
        call_count = 0
        
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MessageTimeoutError(5.0)
            return "async_success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = AsyncErrorHandler(retry_config=config)
        
        result = await handler.execute(failing_async_func)
        assert result == "async_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_fallback_on_failure(self):
        """Test async fallback execution on failure."""
        async def failing_async_func():
            raise RAGRetrievalError("test error")
        
        fallback = DefaultValueFallback("async_fallback_result")
        config = RetryConfig(max_attempts=1)
        handler = AsyncErrorHandler(retry_config=config, fallback_strategy=fallback)
        
        result = await handler.execute(failing_async_func)
        assert result == "async_fallback_result"


class TestDecorators:
    """Test error handling decorators."""
    
    def test_with_retry_decorator(self):
        """Test with_retry decorator."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentCommunicationError("test error")
            return "decorated_success"
        
        result = failing_func()
        assert result == "decorated_success"
        assert call_count == 3
    
    def test_with_fallback_decorator(self):
        """Test with_fallback decorator."""
        @with_fallback(DefaultValueFallback("decorated_fallback"))
        def failing_func():
            raise ValueError("test error")
        
        result = failing_func()
        assert result == "decorated_fallback"
    
    def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""
        call_count = 0
        
        @with_circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def sometimes_failing_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("test error")
            return "circuit_success"
        
        # First two calls should fail and open circuit
        with pytest.raises(ValueError):
            sometimes_failing_func()
        
        with pytest.raises(ValueError):
            sometimes_failing_func()
        
        # Third call should be blocked by open circuit
        with pytest.raises(DeepAgentException):
            sometimes_failing_func()


class TestSpecificRecoveryStrategies:
    """Test specific recovery strategies for different components."""
    
    def test_rag_retrieval_recovery_config(self):
        """Test RAG retrieval recovery configuration."""
        from deep_agent_system.error_handling import RAGRetrievalRecovery
        
        config = RAGRetrievalRecovery.create_retry_config()
        
        assert config.max_attempts == 2
        assert RAGRetrievalError in config.retryable_exceptions
        assert ConnectionError in config.retryable_exceptions
    
    def test_rag_retrieval_fallback_strategy(self):
        """Test RAG retrieval fallback strategy."""
        from deep_agent_system.error_handling import RAGRetrievalRecovery
        
        fallback = RAGRetrievalRecovery.create_fallback_strategy("default_context")
        
        result = fallback.execute()
        assert result == "default_context"
    
    def test_agent_communication_recovery_config(self):
        """Test agent communication recovery configuration."""
        from deep_agent_system.error_handling import AgentCommunicationRecovery
        
        config = AgentCommunicationRecovery.create_retry_config()
        
        assert config.max_attempts == 3
        assert AgentCommunicationError in config.retryable_exceptions
        assert MessageTimeoutError in config.retryable_exceptions
    
    def test_workflow_recovery_config(self):
        """Test workflow recovery configuration."""
        from deep_agent_system.error_handling import WorkflowRecovery
        
        config = WorkflowRecovery.create_retry_config()
        circuit_config = WorkflowRecovery.create_circuit_breaker_config()
        
        assert config.max_attempts == 2
        assert circuit_config.failure_threshold == 3
        assert circuit_config.recovery_timeout == 120.0


class TestErrorHandlingIntegration:
    """Test error handling integration scenarios."""
    
    def test_combined_retry_and_fallback(self):
        """Test combining retry and fallback strategies."""
        call_count = 0
        
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise AgentCommunicationError("persistent error")
        
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        fallback = DefaultValueFallback("combined_fallback")
        handler = ErrorHandler(retry_config=config, fallback_strategy=fallback)
        
        result = handler.execute(always_failing_func)
        assert result == "combined_fallback"
        assert call_count == 2  # Should retry once, then fallback
    
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("test error")
        
        circuit_config = CircuitBreakerConfig(failure_threshold=2, expected_exception=ValueError)
        retry_config = RetryConfig(max_attempts=1, base_delay=0.01)  # Only 1 attempt to trigger circuit faster
        handler = ErrorHandler(
            retry_config=retry_config,
            circuit_breaker_config=circuit_config,
        )
        
        # First failure
        with pytest.raises(ValueError):
            handler.execute(failing_func)
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            handler.execute(failing_func)
        
        # Circuit should be open now
        with pytest.raises(DeepAgentException):
            handler.execute(lambda: "success")
    
    @pytest.mark.asyncio
    async def test_async_error_handling_with_timeout(self):
        """Test async error handling with timeout scenarios."""
        async def timeout_func():
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("operation timed out")
        
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=[asyncio.TimeoutError],
        )
        fallback = DefaultValueFallback("timeout_fallback")
        handler = AsyncErrorHandler(retry_config=config, fallback_strategy=fallback)
        
        result = await handler.execute(timeout_func)
        assert result == "timeout_fallback"