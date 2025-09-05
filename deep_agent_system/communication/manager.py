"""Agent communication manager for handling inter-agent messaging."""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.error_handling import AsyncErrorHandler, RetryConfig
from deep_agent_system.exceptions import (
    AgentCommunicationError,
    AgentNotFoundError,
    MessageQueueFullError,
    MessageTimeoutError,
)
from deep_agent_system.models.messages import (
    ConversationHistory,
    Message,
    MessageType,
    Response,
)


logger = logging.getLogger(__name__)


class MessageQueue:
    """Thread-safe message queue for agent communication."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the message queue.
        
        Args:
            max_size: Maximum number of messages to store
        """
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def put(self, message: Message) -> None:
        """Add a message to the queue.
        
        Args:
            message: Message to add
        """
        async with self._lock:
            self._queue.append(message)
            logger.debug(f"Added message {message.id} to queue")
    
    async def get(self) -> Optional[Message]:
        """Get the next message from the queue.
        
        Returns:
            Next message or None if queue is empty
        """
        async with self._lock:
            if self._queue:
                message = self._queue.popleft()
                logger.debug(f"Retrieved message {message.id} from queue")
                return message
            return None
    
    async def peek(self) -> Optional[Message]:
        """Peek at the next message without removing it.
        
        Returns:
            Next message or None if queue is empty
        """
        async with self._lock:
            return self._queue[0] if self._queue else None
    
    async def size(self) -> int:
        """Get the current queue size.
        
        Returns:
            Number of messages in queue
        """
        async with self._lock:
            return len(self._queue)
    
    async def clear(self) -> None:
        """Clear all messages from the queue."""
        async with self._lock:
            self._queue.clear()
            logger.debug("Cleared message queue")


class AgentCommunicationManager:
    """Manages communication between agents in the system."""
    
    def __init__(self, max_queue_size: int = 1000, message_timeout: int = 30):
        """Initialize the communication manager.
        
        Args:
            max_queue_size: Maximum size for agent message queues
            message_timeout: Timeout in seconds for message delivery
        """
        self.max_queue_size = max_queue_size
        self.message_timeout = message_timeout
        
        # Agent registry
        self._agents: Dict[str, BaseAgent] = {}
        
        # Message queues for each agent
        self._message_queues: Dict[str, MessageQueue] = {}
        
        # Message routing table
        self._routing_table: Dict[str, str] = {}
        
        # Conversation contexts shared between agents
        self._shared_contexts: Dict[str, ConversationHistory] = {}
        
        # Message delivery callbacks
        self._delivery_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Pending message responses
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
        # Message delivery statistics
        self._delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "average_delivery_time": 0.0,
        }
        
        logger.info("AgentCommunicationManager initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the communication manager.
        
        Args:
            agent: Agent to register
        """
        agent_id = agent.agent_id
        
        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already registered, updating registration")
        
        self._agents[agent_id] = agent
        self._message_queues[agent_id] = MessageQueue(self.max_queue_size)
        
        # Register this agent with all other agents for direct communication
        for other_agent in self._agents.values():
            if other_agent.agent_id != agent_id:
                other_agent.register_agent(agent)
                agent.register_agent(other_agent)
        
        logger.info(f"Registered agent {agent_id} with communication manager")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the communication manager.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not registered")
            return
        
        # Unregister from all other agents
        for other_agent in self._agents.values():
            if other_agent.agent_id != agent_id:
                other_agent.unregister_agent(agent_id)
        
        # Clean up resources
        del self._agents[agent_id]
        if agent_id in self._message_queues:
            del self._message_queues[agent_id]
        if agent_id in self._routing_table:
            del self._routing_table[agent_id]
        
        # Clean up shared contexts where this agent was involved
        contexts_to_remove = []
        for context_id, context in self._shared_contexts.items():
            if agent_id in context_id:
                contexts_to_remove.append(context_id)
        
        for context_id in contexts_to_remove:
            del self._shared_contexts[context_id]
        
        logger.info(f"Unregistered agent {agent_id} from communication manager")
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs.
        
        Returns:
            List of agent IDs
        """
        return list(self._agents.keys())
    
    def is_agent_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            True if agent is registered, False otherwise
        """
        return agent_id in self._agents
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        message_type: MessageType = MessageType.AGENT_REQUEST,
        metadata: Optional[Dict[str, Any]] = None,
        wait_for_response: bool = False,
        timeout: Optional[int] = None,
    ) -> Optional[Response]:
        """Send a message from one agent to another with error handling.
        
        Args:
            sender_id: ID of sending agent
            recipient_id: ID of receiving agent
            content: Message content
            message_type: Type of message
            metadata: Optional message metadata
            wait_for_response: Whether to wait for a response
            timeout: Timeout for response (uses default if None)
            
        Returns:
            Response if wait_for_response is True, None otherwise
            
        Raises:
            AgentNotFoundError: If sender or recipient agent is not registered
            MessageTimeoutError: If message delivery times out
            AgentCommunicationError: If message delivery fails
        """
        if sender_id not in self._agents:
            raise AgentNotFoundError(sender_id)
        
        if recipient_id not in self._agents:
            raise AgentNotFoundError(recipient_id)
        
        # Create message
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        
        # Create error handler for message delivery
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            retryable_exceptions=[
                AgentCommunicationError,
                MessageTimeoutError,
                ConnectionError,
            ],
        )
        error_handler = AsyncErrorHandler(retry_config=retry_config)
        
        async def _send_message_impl():
            start_time = datetime.utcnow()
            
            try:
                # Check queue capacity
                queue_size = await self._message_queues[recipient_id].size()
                if queue_size >= self.max_queue_size:
                    raise MessageQueueFullError(recipient_id, queue_size)
                
                # Add to recipient's message queue
                await self._message_queues[recipient_id].put(message)
                
                # Update statistics
                self._delivery_stats["messages_sent"] += 1
                
                # Deliver message to recipient
                response = await self._deliver_message(message, wait_for_response, timeout)
                
                # Update delivery time statistics
                delivery_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_delivery_time(delivery_time)
                
                self._delivery_stats["messages_delivered"] += 1
                
                # Call delivery callbacks
                for callback in self._delivery_callbacks.get(recipient_id, []):
                    try:
                        callback(message, response)
                    except Exception as e:
                        logger.error(f"Error in delivery callback: {e}")
                
                logger.debug(f"Message {message.id} delivered from {sender_id} to {recipient_id}")
                return response
                
            except (MessageQueueFullError, MessageTimeoutError):
                # Re-raise specific exceptions without wrapping
                raise
            except Exception as e:
                self._delivery_stats["messages_failed"] += 1
                raise AgentCommunicationError(
                    f"Failed to deliver message from {sender_id} to {recipient_id}",
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    message_id=message.id,
                    cause=e,
                )
        
        try:
            return await error_handler.execute(_send_message_impl)
        except Exception as e:
            logger.error(f"Failed to send message from {sender_id} to {recipient_id}: {e}")
            raise
    
    async def _deliver_message(
        self,
        message: Message,
        wait_for_response: bool = False,
        timeout: Optional[int] = None,
    ) -> Optional[Response]:
        """Deliver a message to the recipient agent with error handling.
        
        Args:
            message: Message to deliver
            wait_for_response: Whether to wait for a response
            timeout: Timeout for response
            
        Returns:
            Response if wait_for_response is True, None otherwise
            
        Raises:
            MessageTimeoutError: If waiting for response times out
            AgentCommunicationError: If message processing fails
        """
        recipient_agent = self._agents[message.recipient_id]
        
        if wait_for_response:
            # Create future for response
            response_future = asyncio.Future()
            self._pending_responses[message.id] = response_future
            
            try:
                # Process message in recipient agent
                response = recipient_agent.process_message(message)
                
                # Complete the future
                if not response_future.done():
                    response_future.set_result(response)
                
                # Wait for response with timeout
                actual_timeout = timeout or self.message_timeout
                return await asyncio.wait_for(response_future, timeout=actual_timeout)
                
            except asyncio.TimeoutError:
                logger.error(f"Message {message.id} timed out waiting for response")
                if not response_future.done():
                    response_future.cancel()
                raise MessageTimeoutError(
                    timeout_seconds=actual_timeout,
                    sender_id=message.sender_id,
                    recipient_id=message.recipient_id,
                    message_id=message.id,
                )
            except Exception as e:
                logger.error(f"Error processing message {message.id}: {e}")
                raise AgentCommunicationError(
                    f"Failed to process message {message.id}",
                    sender_id=message.sender_id,
                    recipient_id=message.recipient_id,
                    message_id=message.id,
                    cause=e,
                )
            finally:
                # Clean up pending response
                if message.id in self._pending_responses:
                    del self._pending_responses[message.id]
        else:
            # Fire and forget
            try:
                response = recipient_agent.process_message(message)
                return response
            except Exception as e:
                logger.error(f"Error processing fire-and-forget message {message.id}: {e}")
                raise AgentCommunicationError(
                    f"Failed to process message {message.id}",
                    sender_id=message.sender_id,
                    recipient_id=message.recipient_id,
                    message_id=message.id,
                    cause=e,
                )
    
    async def broadcast_message(
        self,
        sender_id: str,
        content: str,
        message_type: MessageType = MessageType.SYSTEM_MESSAGE,
        metadata: Optional[Dict[str, Any]] = None,
        exclude_agents: Optional[Set[str]] = None,
    ) -> List[str]:
        """Broadcast a message to all registered agents.
        
        Args:
            sender_id: ID of sending agent
            content: Message content
            message_type: Type of message
            metadata: Optional message metadata
            exclude_agents: Set of agent IDs to exclude from broadcast
            
        Returns:
            List of agent IDs that received the message
        """
        if sender_id not in self._agents:
            raise ValueError(f"Sender agent {sender_id} not registered")
        
        exclude_agents = exclude_agents or set()
        exclude_agents.add(sender_id)  # Don't send to self
        
        recipients = []
        tasks = []
        
        for agent_id in self._agents:
            if agent_id not in exclude_agents:
                try:
                    task = asyncio.create_task(self.send_message(
                        sender_id=sender_id,
                        recipient_id=agent_id,
                        content=content,
                        message_type=message_type,
                        metadata=metadata,
                    ))
                    tasks.append(task)
                    recipients.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to create broadcast task for agent {agent_id}: {e}")
        
        # Wait for all broadcast messages to complete
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during broadcast execution: {e}")
        
        logger.info(f"Broadcast message from {sender_id} to {len(recipients)} agents")
        return recipients
    
    def create_shared_context(self, agent_ids: List[str], context_id: Optional[str] = None) -> str:
        """Create a shared conversation context between agents.
        
        Args:
            agent_ids: List of agent IDs to share context
            context_id: Optional custom context ID
            
        Returns:
            Context ID for the shared context
        """
        if not agent_ids:
            raise ValueError("At least one agent ID required for shared context")
        
        # Validate all agents are registered
        for agent_id in agent_ids:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not registered")
        
        # Generate context ID if not provided
        if not context_id:
            context_id = f"shared_context_{uuid4().hex[:8]}"
        
        # Create shared conversation history
        shared_context = ConversationHistory(session_id=context_id)
        self._shared_contexts[context_id] = shared_context
        
        logger.info(f"Created shared context {context_id} for agents: {agent_ids}")
        return context_id
    
    def add_to_shared_context(self, context_id: str, message: Message) -> None:
        """Add a message to a shared context.
        
        Args:
            context_id: ID of the shared context
            message: Message to add
        """
        if context_id not in self._shared_contexts:
            raise ValueError(f"Shared context {context_id} not found")
        
        self._shared_contexts[context_id].add_message(message)
        logger.debug(f"Added message {message.id} to shared context {context_id}")
    
    def add_response_to_shared_context(self, context_id: str, response: Response) -> None:
        """Add a response to a shared context.
        
        Args:
            context_id: ID of the shared context
            response: Response to add
        """
        if context_id not in self._shared_contexts:
            raise ValueError(f"Shared context {context_id} not found")
        
        self._shared_contexts[context_id].add_response(response)
        logger.debug(f"Added response from {response.agent_id} to shared context {context_id}")
    
    def get_shared_context(self, context_id: str) -> Optional[ConversationHistory]:
        """Get a shared conversation context.
        
        Args:
            context_id: ID of the shared context
            
        Returns:
            Shared conversation history or None if not found
        """
        return self._shared_contexts.get(context_id)
    
    def remove_shared_context(self, context_id: str) -> None:
        """Remove a shared conversation context.
        
        Args:
            context_id: ID of the shared context to remove
        """
        if context_id in self._shared_contexts:
            del self._shared_contexts[context_id]
            logger.info(f"Removed shared context {context_id}")
    
    def add_delivery_callback(self, agent_id: str, callback: Callable) -> None:
        """Add a callback for message delivery to an agent.
        
        Args:
            agent_id: Agent ID to add callback for
            callback: Callback function (message, response) -> None
        """
        self._delivery_callbacks[agent_id].append(callback)
        logger.debug(f"Added delivery callback for agent {agent_id}")
    
    def remove_delivery_callback(self, agent_id: str, callback: Callable) -> None:
        """Remove a delivery callback for an agent.
        
        Args:
            agent_id: Agent ID to remove callback from
            callback: Callback function to remove
        """
        if agent_id in self._delivery_callbacks:
            try:
                self._delivery_callbacks[agent_id].remove(callback)
                logger.debug(f"Removed delivery callback for agent {agent_id}")
            except ValueError:
                logger.warning(f"Callback not found for agent {agent_id}")
    
    async def get_queue_size(self, agent_id: str) -> int:
        """Get the message queue size for an agent.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            Queue size
        """
        if agent_id not in self._message_queues:
            raise ValueError(f"Agent {agent_id} not registered")
        
        return await self._message_queues[agent_id].size()
    
    async def clear_queue(self, agent_id: str) -> None:
        """Clear the message queue for an agent.
        
        Args:
            agent_id: Agent ID to clear queue for
        """
        if agent_id not in self._message_queues:
            raise ValueError(f"Agent {agent_id} not registered")
        
        await self._message_queues[agent_id].clear()
        logger.info(f"Cleared message queue for agent {agent_id}")
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get message delivery statistics.
        
        Returns:
            Dictionary containing delivery statistics
        """
        return self._delivery_stats.copy()
    
    def _update_delivery_time(self, delivery_time: float) -> None:
        """Update average delivery time statistics.
        
        Args:
            delivery_time: Time taken for message delivery in seconds
        """
        current_avg = self._delivery_stats["average_delivery_time"]
        delivered_count = self._delivery_stats["messages_delivered"]
        
        if delivered_count <= 1:
            self._delivery_stats["average_delivery_time"] = delivery_time
        else:
            # Calculate running average
            new_avg = ((current_avg * (delivered_count - 1)) + delivery_time) / delivered_count
            self._delivery_stats["average_delivery_time"] = new_avg
    
    def reset_statistics(self) -> None:
        """Reset delivery statistics."""
        self._delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "average_delivery_time": 0.0,
        }
        logger.info("Reset delivery statistics")
    
    def shutdown(self) -> None:
        """Shutdown the communication manager and clean up resources."""
        logger.info("Shutting down AgentCommunicationManager")
        
        # Cancel all pending responses
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        
        # Clear all data structures
        self._agents.clear()
        self._message_queues.clear()
        self._routing_table.clear()
        self._shared_contexts.clear()
        self._delivery_callbacks.clear()
        self._pending_responses.clear()
        
        logger.info("AgentCommunicationManager shutdown complete")