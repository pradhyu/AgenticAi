"""Base agent class providing core functionality for all agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from deep_agent_system.config.models import SystemConfig
from deep_agent_system.models.agents import AgentConfig, AgentState, Capability
from deep_agent_system.models.messages import (
    Context,
    ConversationHistory,
    Message,
    MessageType,
    Response,
    RetrievalType,
)
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        config: AgentConfig,
        prompt_manager: PromptManager,
        rag_manager: Optional[RAGManager] = None,
        system_config: Optional[SystemConfig] = None,
    ):
        """Initialize the base agent.
        
        Args:
            config: Agent configuration
            prompt_manager: Manager for loading and handling prompts
            rag_manager: Optional RAG manager for context retrieval
            system_config: Optional system-wide configuration
        """
        self.config = config
        self.prompt_manager = prompt_manager
        self.rag_manager = rag_manager
        self.system_config = system_config
        
        # Initialize agent state
        self.state = AgentState(agent_id=config.agent_id)
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory()
        
        # Agent registry for communication
        self._agent_registry: Dict[str, "BaseAgent"] = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {config.agent_id}")
    
    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self.config.agent_id
    
    @property
    def agent_type(self) -> str:
        """Get the agent's type."""
        return self.config.agent_type.value
    
    def register_agent(self, agent: "BaseAgent") -> None:
        """Register another agent for communication.
        
        Args:
            agent: The agent to register
        """
        self._agent_registry[agent.agent_id] = agent
        logger.debug(f"Registered agent {agent.agent_id} with {self.agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]
            logger.debug(f"Unregistered agent {agent_id} from {self.agent_id}")
    
    def has_capability(self, capability: Capability) -> bool:
        """Check if the agent has a specific capability.
        
        Args:
            capability: The capability to check for
            
        Returns:
            True if the agent has the capability, False otherwise
        """
        return self.config.has_capability(capability)
    
    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt template for this agent.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            
        Returns:
            The prompt template string
            
        Raises:
            ValueError: If the prompt is not found
        """
        # First try to get from agent-specific prompts
        prompt = self.config.get_prompt_template(prompt_name)
        if prompt:
            return prompt
        
        # Fall back to prompt manager
        try:
            return self.prompt_manager.get_prompt(self.agent_type, prompt_name)
        except Exception as e:
            raise ValueError(f"Prompt '{prompt_name}' not found for agent type '{self.agent_type}': {e}")
    
    def retrieve_context(
        self,
        query: str,
        retrieval_type: RetrievalType = RetrievalType.HYBRID,
        k: int = 5
    ) -> Optional[Context]:
        """Retrieve context using RAG.
        
        Args:
            query: The query to search for
            retrieval_type: Type of retrieval to perform
            k: Number of documents to retrieve
            
        Returns:
            Context object with retrieved information, or None if RAG is disabled
        """
        if not self.rag_manager:
            logger.warning(f"RAG manager not available for agent {self.agent_id}")
            return None
        
        if not self.config.rag_enabled:
            logger.debug(f"RAG disabled for agent {self.agent_id}")
            return None
        
        try:
            self.state.update_activity(f"Retrieving context for: {query[:50]}...")
            
            if retrieval_type == RetrievalType.VECTOR:
                documents = self.rag_manager.retrieve_vector_context(query, k=k)
                return Context(
                    query=query,
                    documents=documents,
                    retrieval_method=RetrievalType.VECTOR
                )
            elif retrieval_type == RetrievalType.GRAPH and self.config.graph_rag_enabled:
                graph_data = self.rag_manager.retrieve_graph_context(query)
                return Context(
                    query=query,
                    graph_data=graph_data,
                    retrieval_method=RetrievalType.GRAPH
                )
            elif retrieval_type == RetrievalType.HYBRID:
                combined_context = self.rag_manager.hybrid_retrieve(query, k=k)
                return combined_context
            else:
                logger.warning(f"Unsupported retrieval type: {retrieval_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving context for agent {self.agent_id}: {e}")
            return None
        finally:
            self.state.clear_task()
    
    def communicate_with_agent(self, agent_id: str, message: Message) -> Optional[Response]:
        """Send a message to another agent and get a response.
        
        Args:
            agent_id: ID of the target agent
            message: Message to send
            
        Returns:
            Response from the target agent, or None if communication failed
        """
        if agent_id not in self._agent_registry:
            logger.error(f"Agent {agent_id} not registered with {self.agent_id}")
            return None
        
        target_agent = self._agent_registry[agent_id]
        
        try:
            logger.debug(f"Agent {self.agent_id} sending message to {agent_id}")
            response = target_agent.process_message(message)
            
            # Add to conversation history
            self.conversation_history.add_message(message)
            if response:
                self.conversation_history.add_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error communicating with agent {agent_id}: {e}")
            return None
    
    def process_message(self, message: Message) -> Response:
        """Process an incoming message and generate a response.
        
        Args:
            message: The message to process
            
        Returns:
            Response to the message
        """
        try:
            self.state.update_activity(f"Processing message: {message.message_type.value}")
            
            # Add message to conversation history
            self.conversation_history.add_message(message)
            
            # Delegate to the specific agent implementation
            response = self._process_message_impl(message)
            
            # Add response to conversation history
            self.conversation_history.add_response(response)
            
            logger.debug(f"Agent {self.agent_id} processed message {message.id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in agent {self.agent_id}: {e}")
            
            # Return error response
            error_response = Response(
                message_id=message.id,
                agent_id=self.agent_id,
                content=f"Error processing message: {str(e)}",
                confidence_score=0.0,
                metadata={"error": True, "error_type": type(e).__name__}
            )
            
            self.conversation_history.add_response(error_response)
            return error_response
        finally:
            self.state.clear_task()
    
    @abstractmethod
    def _process_message_impl(self, message: Message) -> Response:
        """Implementation-specific message processing.
        
        This method must be implemented by concrete agent classes.
        
        Args:
            message: The message to process
            
        Returns:
            Response to the message
        """
        pass
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for the agent.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of conversation context items
        """
        context = []
        
        recent_messages = self.conversation_history.get_recent_messages(max_messages)
        recent_responses = self.conversation_history.get_recent_responses(max_messages)
        
        # Combine messages and responses chronologically
        all_items = []
        
        for msg in recent_messages:
            all_items.append(("message", msg.timestamp, msg))
        
        for resp in recent_responses:
            all_items.append(("response", resp.timestamp, resp))
        
        # Sort by timestamp
        all_items.sort(key=lambda x: x[1])
        
        # Convert to context format
        for item_type, timestamp, item in all_items:
            if item_type == "message":
                context.append({
                    "type": "message",
                    "sender": item.sender_id,
                    "recipient": item.recipient_id,
                    "content": item.content,
                    "timestamp": timestamp.isoformat(),
                })
            else:  # response
                context.append({
                    "type": "response",
                    "agent": item.agent_id,
                    "content": item.content,
                    "confidence": item.confidence_score,
                    "timestamp": timestamp.isoformat(),
                })
        
        return context[-max_messages:] if max_messages > 0 else context
    
    def clear_conversation_history(self) -> None:
        """Clear the agent's conversation history."""
        self.conversation_history.clear_history()
        logger.debug(f"Cleared conversation history for agent {self.agent_id}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for the agent.
        
        Returns:
            Dictionary containing agent state information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_active": self.state.is_active,
            "current_task": self.state.current_task,
            "last_activity": self.state.last_activity,
            "conversation_count": self.state.conversation_count,
            "capabilities": [cap.value for cap in self.config.capabilities],
            "rag_enabled": self.config.rag_enabled,
            "graph_rag_enabled": self.config.graph_rag_enabled,
        }
    
    def shutdown(self) -> None:
        """Shutdown the agent and clean up resources."""
        self.state.is_active = False
        self.state.clear_task()
        self._agent_registry.clear()
        logger.info(f"Agent {self.agent_id} shutdown complete")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"