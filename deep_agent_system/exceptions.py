"""Custom exception classes for the Deep Agent System."""

from typing import Any, Dict, Optional


class DeepAgentException(Exception):
    """Base exception class for all Deep Agent System errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [self.message]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class AgentCommunicationError(DeepAgentException):
    """Exception raised when agent communication fails."""
    
    def __init__(
        self,
        message: str,
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the communication error.
        
        Args:
            message: Error message
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            message_id: ID of the message that failed
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "message_id": message_id,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_id = message_id


class AgentNotFoundError(AgentCommunicationError):
    """Exception raised when trying to communicate with a non-existent agent."""
    
    def __init__(self, agent_id: str, **kwargs):
        """Initialize the agent not found error.
        
        Args:
            agent_id: ID of the agent that was not found
            **kwargs: Additional arguments for base exception
        """
        message = f"Agent '{agent_id}' not found or not registered"
        super().__init__(message, error_code="AGENT_NOT_FOUND", **kwargs)
        self.agent_id = agent_id


class MessageTimeoutError(AgentCommunicationError):
    """Exception raised when message delivery times out."""
    
    def __init__(self, timeout_seconds: float, **kwargs):
        """Initialize the timeout error.
        
        Args:
            timeout_seconds: Number of seconds that elapsed before timeout
            **kwargs: Additional arguments for base exception
        """
        message = f"Message delivery timed out after {timeout_seconds} seconds"
        super().__init__(message, error_code="MESSAGE_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class MessageQueueFullError(AgentCommunicationError):
    """Exception raised when an agent's message queue is full."""
    
    def __init__(self, agent_id: str, queue_size: int, **kwargs):
        """Initialize the queue full error.
        
        Args:
            agent_id: ID of the agent whose queue is full
            queue_size: Current size of the queue
            **kwargs: Additional arguments for base exception
        """
        message = f"Message queue for agent '{agent_id}' is full (size: {queue_size})"
        super().__init__(message, error_code="QUEUE_FULL", **kwargs)
        self.agent_id = agent_id
        self.queue_size = queue_size


class PromptLoadError(DeepAgentException):
    """Exception raised when prompt loading fails."""
    
    def __init__(
        self,
        message: str,
        prompt_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        prompt_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the prompt load error.
        
        Args:
            message: Error message
            prompt_name: Name of the prompt that failed to load
            agent_type: Type of agent the prompt was for
            prompt_path: Path to the prompt file
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "prompt_name": prompt_name,
            "agent_type": agent_type,
            "prompt_path": prompt_path,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.prompt_name = prompt_name
        self.agent_type = agent_type
        self.prompt_path = prompt_path


class PromptNotFoundError(PromptLoadError):
    """Exception raised when a required prompt is not found."""
    
    def __init__(self, prompt_name: str, agent_type: str, **kwargs):
        """Initialize the prompt not found error.
        
        Args:
            prompt_name: Name of the prompt that was not found
            agent_type: Type of agent the prompt was for
            **kwargs: Additional arguments for base exception
        """
        message = f"Prompt '{prompt_name}' not found for agent type '{agent_type}'"
        super().__init__(
            message,
            prompt_name=prompt_name,
            agent_type=agent_type,
            error_code="PROMPT_NOT_FOUND",
            **kwargs,
        )


class PromptValidationError(PromptLoadError):
    """Exception raised when prompt validation fails."""
    
    def __init__(self, prompt_name: str, validation_errors: list, **kwargs):
        """Initialize the prompt validation error.
        
        Args:
            prompt_name: Name of the prompt that failed validation
            validation_errors: List of validation error messages
            **kwargs: Additional arguments for base exception
        """
        message = f"Prompt '{prompt_name}' failed validation: {'; '.join(validation_errors)}"
        details = kwargs.get("details", {})
        details["validation_errors"] = validation_errors
        kwargs["details"] = details
        super().__init__(
            message,
            prompt_name=prompt_name,
            error_code="PROMPT_VALIDATION_FAILED",
            **kwargs,
        )
        self.validation_errors = validation_errors


class RAGRetrievalError(DeepAgentException):
    """Exception raised when RAG retrieval fails."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        retrieval_type: Optional[str] = None,
        store_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the RAG retrieval error.
        
        Args:
            message: Error message
            query: Query that failed to retrieve results
            retrieval_type: Type of retrieval (vector, graph, hybrid)
            store_type: Type of store (chromadb, neo4j)
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "query": query,
            "retrieval_type": retrieval_type,
            "store_type": store_type,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.query = query
        self.retrieval_type = retrieval_type
        self.store_type = store_type


class VectorStoreError(RAGRetrievalError):
    """Exception raised when vector store operations fail."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize the vector store error.
        
        Args:
            message: Error message
            **kwargs: Additional arguments for base exception
        """
        super().__init__(
            message,
            store_type="vector",
            error_code="VECTOR_STORE_ERROR",
            **kwargs,
        )


class GraphStoreError(RAGRetrievalError):
    """Exception raised when graph store operations fail."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize the graph store error.
        
        Args:
            message: Error message
            **kwargs: Additional arguments for base exception
        """
        super().__init__(
            message,
            store_type="graph",
            error_code="GRAPH_STORE_ERROR",
            **kwargs,
        )


class WorkflowExecutionError(DeepAgentException):
    """Exception raised when workflow execution fails."""
    
    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        step_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the workflow execution error.
        
        Args:
            message: Error message
            workflow_id: ID of the workflow that failed
            step_name: Name of the step that failed
            agent_id: ID of the agent that was executing the step
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "workflow_id": workflow_id,
            "step_name": step_name,
            "agent_id": agent_id,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.workflow_id = workflow_id
        self.step_name = step_name
        self.agent_id = agent_id


class WorkflowValidationError(WorkflowExecutionError):
    """Exception raised when workflow validation fails."""
    
    def __init__(self, workflow_id: str, validation_errors: list, **kwargs):
        """Initialize the workflow validation error.
        
        Args:
            workflow_id: ID of the workflow that failed validation
            validation_errors: List of validation error messages
            **kwargs: Additional arguments for base exception
        """
        message = f"Workflow '{workflow_id}' failed validation: {'; '.join(validation_errors)}"
        details = kwargs.get("details", {})
        details["validation_errors"] = validation_errors
        kwargs["details"] = details
        super().__init__(
            message,
            workflow_id=workflow_id,
            error_code="WORKFLOW_VALIDATION_FAILED",
            **kwargs,
        )
        self.validation_errors = validation_errors


class ConfigurationError(DeepAgentException):
    """Exception raised when configuration loading or validation fails."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_file: Configuration file that caused the error
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "config_key": config_key,
            "config_file": config_file,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.config_key = config_key
        self.config_file = config_file


class MissingConfigurationError(ConfigurationError):
    """Exception raised when required configuration is missing."""
    
    def __init__(self, config_key: str, **kwargs):
        """Initialize the missing configuration error.
        
        Args:
            config_key: Configuration key that is missing
            **kwargs: Additional arguments for base exception
        """
        message = f"Required configuration '{config_key}' is missing"
        super().__init__(
            message,
            config_key=config_key,
            error_code="MISSING_CONFIGURATION",
            **kwargs,
        )


class InvalidConfigurationError(ConfigurationError):
    """Exception raised when configuration values are invalid."""
    
    def __init__(self, config_key: str, value: Any, reason: str, **kwargs):
        """Initialize the invalid configuration error.
        
        Args:
            config_key: Configuration key with invalid value
            value: The invalid value
            reason: Reason why the value is invalid
            **kwargs: Additional arguments for base exception
        """
        message = f"Invalid configuration '{config_key}' = '{value}': {reason}"
        details = kwargs.get("details", {})
        details.update({
            "invalid_value": str(value),
            "reason": reason,
        })
        kwargs["details"] = details
        super().__init__(
            message,
            config_key=config_key,
            error_code="INVALID_CONFIGURATION",
            **kwargs,
        )
        self.value = value
        self.reason = reason


class AgentInitializationError(DeepAgentException):
    """Exception raised when agent initialization fails."""
    
    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the agent initialization error.
        
        Args:
            message: Error message
            agent_id: ID of the agent that failed to initialize
            agent_type: Type of the agent that failed to initialize
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details.update({
            "agent_id": agent_id,
            "agent_type": agent_type,
        })
        kwargs["details"] = details
        super().__init__(message, **kwargs)
        
        self.agent_id = agent_id
        self.agent_type = agent_type


class SystemShutdownError(DeepAgentException):
    """Exception raised during system shutdown operations."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        """Initialize the system shutdown error.
        
        Args:
            message: Error message
            component: Component that failed during shutdown
            **kwargs: Additional arguments for base exception
        """
        details = kwargs.get("details", {})
        details["component"] = component
        kwargs["details"] = details
        super().__init__(message, error_code="SHUTDOWN_ERROR", **kwargs)
        self.component = component


# Convenience functions for creating common exceptions

def agent_not_found(agent_id: str) -> AgentNotFoundError:
    """Create an AgentNotFoundError.
    
    Args:
        agent_id: ID of the agent that was not found
        
    Returns:
        AgentNotFoundError instance
    """
    return AgentNotFoundError(agent_id)


def message_timeout(timeout_seconds: float, sender_id: str, recipient_id: str) -> MessageTimeoutError:
    """Create a MessageTimeoutError.
    
    Args:
        timeout_seconds: Number of seconds that elapsed before timeout
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent
        
    Returns:
        MessageTimeoutError instance
    """
    return MessageTimeoutError(
        timeout_seconds,
        sender_id=sender_id,
        recipient_id=recipient_id,
    )


def prompt_not_found(prompt_name: str, agent_type: str) -> PromptNotFoundError:
    """Create a PromptNotFoundError.
    
    Args:
        prompt_name: Name of the prompt that was not found
        agent_type: Type of agent the prompt was for
        
    Returns:
        PromptNotFoundError instance
    """
    return PromptNotFoundError(prompt_name, agent_type)


def rag_retrieval_failed(query: str, retrieval_type: str, cause: Exception) -> RAGRetrievalError:
    """Create a RAGRetrievalError.
    
    Args:
        query: Query that failed to retrieve results
        retrieval_type: Type of retrieval that failed
        cause: Underlying exception that caused the failure
        
    Returns:
        RAGRetrievalError instance
    """
    return RAGRetrievalError(
        f"RAG retrieval failed for query: {query}",
        query=query,
        retrieval_type=retrieval_type,
        cause=cause,
    )


def missing_config(config_key: str) -> MissingConfigurationError:
    """Create a MissingConfigurationError.
    
    Args:
        config_key: Configuration key that is missing
        
    Returns:
        MissingConfigurationError instance
    """
    return MissingConfigurationError(config_key)