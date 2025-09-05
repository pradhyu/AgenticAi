"""Comprehensive unit tests for error handling components."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from deep_agent_system.error_handling import (
    ErrorHandler,
    RetryConfig,
    FallbackStrategy,
    DefaultValueFallback,
    CircuitBreaker,
    CircuitBreakerState,
    RAGRetrievalRecovery,
    WorkflowRecovery,
    AgentCommunicationRecovery
)
from deep_agent_system.exceptions import (
    DeepAgentException,
    RAGRetrievalError,
    WorkflowExecutionError,
    AgentCommunicationError
)


class TestRetryConfig:
    """Test cases for RetryConfig."""
    
    def test_retry_config_creation(self):
        """Test creating RetryConfig with custom parameters."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=2.5,
            jitter=True,
            retryable_exceptions=[ValueError, TypeError]
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 2.5
        assert config.jitter is True
        assert ValueError in config.retryable_exceptions
        assert TypeError in config.retryable_exceptions
    
    def test_retry_config_defaults(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is False
        assert config.retryable_exceptions == [Exception]
    
    def test_should_retry_retryable_exception(self):
        """Test should_retry with retryable exception."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ValueError, TypeError]
        )
        
        # Should retry for retryable exceptions within max attempts
        assert config.should_retry(ValueError("test"), 1) is True
        assert config.should_retry(TypeError("test"), 2) is True
        
        # Should not retry beyond max attempts
        assert config.should_retry(ValueError("test"), 3) is False
    
    def test_should_retry_non_retryable_exception(self):
        """Test should_retry with non-retryable exception."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ValueError]
        )
        
        # Should not retry for non-retryable exceptions
        assert config.should_retry(TypeError("test"), 1) is False
        assert config.should_retry(RuntimeError("test"), 1) is False
    
    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        
        # Test exponential backoff
        assert config.get_delay(1) == 1.0  # base_delay
        assert config.get_delay(2) == 2.0  # base_delay * multiplier
        assert config.get_delay(3) == 4.0  # base_delay * multiplier^2
        
        # Test max delay cap
        assert config.get_delay(10) == 10.0  # Should be capped at max_delay
    
    def test_get_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            jitter=True
        )
        
        # With jitter, delay should vary but be within expected range
        delay1 = config.get_delay(1)
        delay2 = config.get_delay(1)
        
        # Should be around base_delay but with some variation
        assert 0.5 <= delay1 <= 1.5
        assert 0.5 <= delay2 <= 1.5
        
        # Multiple calls should potentially give different results
        delays = [config.get_delay(1) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have some variation


class TestDefaultValueFallback:
    """Test cases for DefaultValueFallback."""
    
    def test_default_value_fallback_creation(self):
        """Test creating DefaultValueFallback."""
        fallback = DefaultValueFallback("default_value")
        assert fallback.default_value == "default_value"
    
    def test_default_value_fallback_execute(self):
        """Test executing DefaultValueFallback."""
        fallback = DefaultValueFallback(42)
        result = fallback.execute("arg1", "arg2", kwarg1="value1")
        assert result == 42
    
    def test_default_value_fallback_with_none(self):
        """Test DefaultValueFallback with None value."""
        fallback = DefaultValueFallback(None)
        result = fallback.execute()
        assert result is None
    
    def test_default_value_fallback_with_complex_object(self):
        """Test DefaultValueFallback with complex object."""
        default_dict = {"key": "value", "number": 123}
        fallback = DefaultValueFallback(default_dict)
        result = fallback.execute()
        assert result == default_dict


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    def test_circuit_breaker_creation(self):
        """Test creating CircuitBreaker."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=ValueError
        )
        
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception == ValueError
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_successful_call(self):
        """Test successful function call through circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def successful_func():
            return "success"
        
        result = breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_failure_accumulation(self):
        """Test failure accumulation in circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)
        
        def failing_func():
            raise ValueError("test error")
        
        # First two failures should not open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 1
        
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 2
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.failure_count == 3
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker behavior in open state."""
        breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)
        
        def failing_func():
            raise ValueError("test error")
        
        # Trigger circuit to open
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Subsequent calls should raise CircuitBreakerOpenError
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            breaker.call(failing_func)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test transition to half-open state after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
            expected_exception=ValueError
        )
        
        def failing_func():
            raise ValueError("test error")
        
        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should transition to half-open
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        # Note: The exact behavior depends on implementation details
    
    def test_circuit_breaker_reset_on_success(self):
        """Test circuit breaker reset on successful call."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)
        
        def failing_func():
            raise ValueError("test error")
        
        def successful_func():
            return "success"
        
        # Accumulate some failures
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.failure_count == 2
        
        # Successful call should reset failure count
        result = breaker.call(successful_func)
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == CircuitBreakerState.CLOSED


class TestErrorHandler:
    """Test cases for ErrorHandler."""
    
    def test_error_handler_creation_minimal(self):
        """Test creating ErrorHandler with minimal configuration."""
        handler = ErrorHandler()
        
        assert handler.retry_config is not None
        assert handler.fallback_strategy is None
        assert handler.circuit_breaker is None
    
    def test_error_handler_creation_full(self):
        """Test creating ErrorHandler with full configuration."""
        retry_config = RetryConfig(max_attempts=5)
        fallback_strategy = DefaultValueFallback("fallback")
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        handler = ErrorHandler(
            retry_config=retry_config,
            fallback_strategy=fallback_strategy,
            circuit_breaker=circuit_breaker
        )
        
        assert handler.retry_config == retry_config
        assert handler.fallback_strategy == fallback_strategy
        assert handler.circuit_breaker == circuit_breaker
    
    def test_error_handler_successful_execution(self):
        """Test successful function execution through error handler."""
        handler = ErrorHandler()
        
        def successful_func(x, y):
            return x + y
        
        result = handler.execute(successful_func, 2, 3)
        assert result == 5
    
    def test_error_handler_retry_on_failure(self):
        """Test retry behavior on function failure."""
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,  # Short delay for testing
            retryable_exceptions=[ValueError]
        )
        handler = ErrorHandler(retry_config=retry_config)
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"
        
        result = handler.execute(failing_func)
        assert result == "success"
        assert call_count == 3
    
    def test_error_handler_fallback_on_exhausted_retries(self):
        """Test fallback execution when retries are exhausted."""
        retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=[ValueError]
        )
        fallback_strategy = DefaultValueFallback("fallback_result")
        handler = ErrorHandler(
            retry_config=retry_config,
            fallback_strategy=fallback_strategy
        )
        
        def always_failing_func():
            raise ValueError("persistent failure")
        
        result = handler.execute(always_failing_func)
        assert result == "fallback_result"
    
    def test_error_handler_no_retry_for_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        retry_config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ValueError]
        )
        handler = ErrorHandler(retry_config=retry_config)
        
        call_count = 0
        
        def func_with_non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("non-retryable error")
        
        with pytest.raises(TypeError):
            handler.execute(func_with_non_retryable_error)
        
        # Should only be called once (no retries)
        assert call_count == 1
    
    def test_error_handler_with_circuit_breaker(self):
        """Test error handler with circuit breaker integration."""
        retry_config = RetryConfig(max_attempts=1)
        circuit_breaker = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)
        handler = ErrorHandler(
            retry_config=retry_config,
            circuit_breaker=circuit_breaker
        )
        
        def failing_func():
            raise ValueError("test error")
        
        # First call should fail and open circuit
        with pytest.raises(ValueError):
            handler.execute(failing_func)
        
        # Second call should be blocked by circuit breaker
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            handler.execute(failing_func)


class TestRecoveryStrategies:
    """Test cases for recovery strategy factories."""
    
    def test_rag_retrieval_recovery_retry_config(self):
        """Test RAG retrieval retry configuration."""
        config = RAGRetrievalRecovery.create_retry_config()
        
        assert config.max_attempts == 2
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        assert RAGRetrievalError in config.retryable_exceptions
        assert ConnectionError in config.retryable_exceptions
        assert TimeoutError in config.retryable_exceptions
    
    def test_rag_retrieval_recovery_fallback_strategy(self):
        """Test RAG retrieval fallback strategy."""
        strategy = RAGRetrievalRecovery.create_fallback_strategy()
        
        assert isinstance(strategy, DefaultValueFallback)
        assert strategy.default_value == ""
        
        # Test with custom default
        custom_strategy = RAGRetrievalRecovery.create_fallback_strategy("custom_default")
        assert custom_strategy.default_value == "custom_default"
    
    def test_workflow_recovery_retry_config(self):
        """Test workflow recovery retry configuration."""
        config = WorkflowRecovery.create_retry_config()
        
        assert config.max_attempts == 2
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert WorkflowExecutionError in config.retryable_exceptions
        assert AgentCommunicationError in config.retryable_exceptions
    
    def test_agent_communication_recovery_retry_config(self):
        """Test agent communication recovery retry configuration."""
        config = AgentCommunicationRecovery.create_retry_config()
        
        assert config.max_attempts == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 5.0
        assert AgentCommunicationError in config.retryable_exceptions
        assert ConnectionError in config.retryable_exceptions
    
    def test_agent_communication_recovery_circuit_breaker(self):
        """Test agent communication circuit breaker."""
        breaker = AgentCommunicationRecovery.create_circuit_breaker()
        
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.expected_exception == AgentCommunicationError


class TestErrorHandlerIntegration:
    """Integration tests for error handling components."""
    
    def test_complete_error_handling_flow(self):
        """Test complete error handling flow with all components."""
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ValueError]
        )
        fallback_strategy = DefaultValueFallback("fallback_result")
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            expected_exception=ValueError
        )
        
        handler = ErrorHandler(
            retry_config=retry_config,
            fallback_strategy=fallback_strategy,
            circuit_breaker=circuit_breaker
        )
        
        call_count = 0
        
        def intermittent_failure_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("temporary failure")
            return "success"
        
        # First execution should succeed after retries
        result = handler.execute(intermittent_failure_func)
        assert result == "success"
        assert call_count == 3
    
    def test_error_handling_with_logging(self):
        """Test error handling with logging integration."""
        retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=[ValueError]
        )
        handler = ErrorHandler(retry_config=retry_config)
        
        def failing_func():
            raise ValueError("test error")
        
        with patch('deep_agent_system.error_handling.logger') as mock_logger:
            with pytest.raises(ValueError):
                handler.execute(failing_func)
            
            # Should have logged retry attempts
            assert mock_logger.warning.call_count >= 1
    
    def test_error_handling_performance(self):
        """Test error handling performance characteristics."""
        retry_config = RetryConfig(
            max_attempts=1,  # No retries for performance test
            base_delay=0.0
        )
        handler = ErrorHandler(retry_config=retry_config)
        
        def fast_func():
            return "result"
        
        # Measure execution time
        start_time = time.time()
        for _ in range(1000):
            result = handler.execute(fast_func)
            assert result == "result"
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 1000 calls)
        execution_time = end_time - start_time
        assert execution_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])