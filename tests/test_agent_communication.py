"""Integration tests for agent communication system."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.communication.manager import AgentCommunicationManager, MessageQueue
from deep_agent_system.config.models import SystemConfig
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.prompts.manager import PromptManager


class MockAgent(BaseAgent):
    """Mock agent for testing communication."""
    
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.ANALYST):
        # Create appropriate capabilities for each agent type
        capabilities_map = {
            AgentType.ANALYST: [Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
            AgentType.ARCHITECT: [Capability.ARCHITECTURE_DESIGN, Capability.SOLUTION_EXPLANATION],
            AgentType.DEVELOPER: [Capability.CODE_GENERATION],
            AgentType.CODE_REVIEWER: [Capability.CODE_REVIEW],
            AgentType.TESTER: [Capability.TEST_CREATION],
        }
        
        # Create mock LLM config
        llm_config = LLMConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            llm_config=llm_config,
            capabilities=capabilities_map.get(agent_type, []),
            rag_enabled=False,
            graph_rag_enabled=False,
        )
        
        # Mock prompt manager
        prompt_manager = Mock(spec=PromptManager)
        prompt_manager.get_prompt.return_value = "Test prompt"
        
        super().__init__(config, prompt_manager)
        
        # Track processed messages for testing
        self.processed_messages = []
        self.response_content = f"Response from {agent_id}"
    
    def _process_message_impl(self, message: Message) -> Response:
        """Mock message processing implementation."""
        self.processed_messages.append(message)
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=self.response_content,
            confidence_score=0.9,
        )


class TestMessageQueue:
    """Test cases for MessageQueue class."""
    
    @pytest.mark.asyncio
    async def test_message_queue_operations(self):
        """Test basic message queue operations."""
        queue = MessageQueue(max_size=3)
        
        # Test empty queue
        assert await queue.size() == 0
        assert await queue.get() is None
        assert await queue.peek() is None
        
        # Add messages
        message1 = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test message 1",
            message_type=MessageType.AGENT_REQUEST,
        )
        message2 = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test message 2",
            message_type=MessageType.AGENT_REQUEST,
        )
        
        await queue.put(message1)
        await queue.put(message2)
        
        assert await queue.size() == 2
        
        # Test peek
        peeked = await queue.peek()
        assert peeked.id == message1.id
        assert await queue.size() == 2  # Size unchanged after peek
        
        # Test get
        retrieved1 = await queue.get()
        assert retrieved1.id == message1.id
        assert await queue.size() == 1
        
        retrieved2 = await queue.get()
        assert retrieved2.id == message2.id
        assert await queue.size() == 0
    
    @pytest.mark.asyncio
    async def test_message_queue_max_size(self):
        """Test message queue max size enforcement."""
        queue = MessageQueue(max_size=2)
        
        # Add messages up to max size
        for i in range(3):  # Add one more than max size
            message = Message(
                sender_id="agent1",
                recipient_id="agent2",
                content=f"Test message {i}",
                message_type=MessageType.AGENT_REQUEST,
            )
            await queue.put(message)
        
        # Should only have 2 messages (max size)
        assert await queue.size() == 2
        
        # First message should be dropped (FIFO with max size)
        first_message = await queue.get()
        assert "Test message 1" in first_message.content
    
    @pytest.mark.asyncio
    async def test_message_queue_clear(self):
        """Test clearing message queue."""
        queue = MessageQueue()
        
        # Add some messages
        for i in range(3):
            message = Message(
                sender_id="agent1",
                recipient_id="agent2",
                content=f"Test message {i}",
                message_type=MessageType.AGENT_REQUEST,
            )
            await queue.put(message)
        
        assert await queue.size() == 3
        
        await queue.clear()
        assert await queue.size() == 0
        assert await queue.get() is None


class TestAgentCommunicationManager:
    """Test cases for AgentCommunicationManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comm_manager = AgentCommunicationManager(max_queue_size=10, message_timeout=5)
        
        # Create mock agents
        self.agent1 = MockAgent("agent1", AgentType.ANALYST)
        self.agent2 = MockAgent("agent2", AgentType.ARCHITECT)
        self.agent3 = MockAgent("agent3", AgentType.DEVELOPER)
    
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Test registration
        self.comm_manager.register_agent(self.agent1)
        assert self.comm_manager.is_agent_registered("agent1")
        assert "agent1" in self.comm_manager.get_registered_agents()
        
        # Test multiple registrations
        self.comm_manager.register_agent(self.agent2)
        self.comm_manager.register_agent(self.agent3)
        
        registered_agents = self.comm_manager.get_registered_agents()
        assert len(registered_agents) == 3
        assert all(agent_id in registered_agents for agent_id in ["agent1", "agent2", "agent3"])
        
        # Test agents are registered with each other
        assert self.agent1._agent_registry["agent2"] == self.agent2
        assert self.agent1._agent_registry["agent3"] == self.agent3
        assert self.agent2._agent_registry["agent1"] == self.agent1
        
        # Test unregistration
        self.comm_manager.unregister_agent("agent2")
        assert not self.comm_manager.is_agent_registered("agent2")
        assert "agent2" not in self.comm_manager.get_registered_agents()
        assert "agent2" not in self.agent1._agent_registry
    
    @pytest.mark.asyncio
    async def test_send_message_basic(self):
        """Test basic message sending between agents."""
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        
        # Send message without waiting for response
        response = await self.comm_manager.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Hello agent2",
            message_type=MessageType.AGENT_REQUEST,
        )
        
        # Check that message was processed
        assert len(self.agent2.processed_messages) == 1
        processed_message = self.agent2.processed_messages[0]
        assert processed_message.sender_id == "agent1"
        assert processed_message.recipient_id == "agent2"
        assert processed_message.content == "Hello agent2"
        assert processed_message.message_type == MessageType.AGENT_REQUEST
        
        # Check response
        assert response is not None
        assert response.agent_id == "agent2"
        assert response.content == "Response from agent2"
    
    @pytest.mark.asyncio
    async def test_send_message_with_response(self):
        """Test sending message and waiting for response."""
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        
        # Send message and wait for response
        response = await self.comm_manager.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Request with response",
            wait_for_response=True,
            timeout=2,
        )
        
        assert response is not None
        assert response.agent_id == "agent2"
        assert response.content == "Response from agent2"
        assert response.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_send_message_invalid_agents(self):
        """Test sending messages with invalid agent IDs."""
        self.comm_manager.register_agent(self.agent1)
        
        # Test invalid sender
        with pytest.raises(ValueError, match="Sender agent invalid_sender not registered"):
            await self.comm_manager.send_message(
                sender_id="invalid_sender",
                recipient_id="agent1",
                content="Test message",
            )
        
        # Test invalid recipient
        with pytest.raises(ValueError, match="Recipient agent invalid_recipient not registered"):
            await self.comm_manager.send_message(
                sender_id="agent1",
                recipient_id="invalid_recipient",
                content="Test message",
            )
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting messages to multiple agents."""
        # Register agents
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        self.comm_manager.register_agent(self.agent3)
        
        # Broadcast message
        recipients = await self.comm_manager.broadcast_message(
            sender_id="agent1",
            content="Broadcast message",
            message_type=MessageType.SYSTEM_MESSAGE,
        )
        
        # Should send to all agents except sender
        assert len(recipients) == 2
        assert "agent2" in recipients
        assert "agent3" in recipients
        assert "agent1" not in recipients
        
        # Test with exclusions
        recipients = await self.comm_manager.broadcast_message(
            sender_id="agent1",
            content="Broadcast with exclusions",
            exclude_agents={"agent3"},
        )
        
        assert len(recipients) == 1
        assert "agent2" in recipients
        assert "agent3" not in recipients
    
    def test_shared_context_management(self):
        """Test shared conversation context management."""
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        
        # Create shared context
        context_id = self.comm_manager.create_shared_context(["agent1", "agent2"])
        assert context_id is not None
        
        # Get shared context
        shared_context = self.comm_manager.get_shared_context(context_id)
        assert shared_context is not None
        assert shared_context.session_id == context_id
        
        # Add message to shared context
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Shared message",
            message_type=MessageType.AGENT_REQUEST,
        )
        
        self.comm_manager.add_to_shared_context(context_id, message)
        
        # Check message was added
        updated_context = self.comm_manager.get_shared_context(context_id)
        assert len(updated_context.messages) == 1
        assert updated_context.messages[0].content == "Shared message"
        
        # Add response to shared context
        response = Response(
            message_id=message.id,
            agent_id="agent2",
            content="Shared response",
        )
        
        self.comm_manager.add_response_to_shared_context(context_id, response)
        
        # Check response was added
        updated_context = self.comm_manager.get_shared_context(context_id)
        assert len(updated_context.responses) == 1
        assert updated_context.responses[0].content == "Shared response"
        
        # Remove shared context
        self.comm_manager.remove_shared_context(context_id)
        assert self.comm_manager.get_shared_context(context_id) is None
    
    def test_shared_context_invalid_agents(self):
        """Test shared context creation with invalid agents."""
        # Test empty agent list
        with pytest.raises(ValueError, match="At least one agent ID required"):
            self.comm_manager.create_shared_context([])
        
        # Test unregistered agent
        with pytest.raises(ValueError, match="Agent invalid_agent not registered"):
            self.comm_manager.create_shared_context(["invalid_agent"])
    
    def test_delivery_callbacks(self):
        """Test message delivery callbacks."""
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        
        # Track callback invocations
        callback_calls = []
        
        def delivery_callback(message, response):
            callback_calls.append((message.id, response.agent_id if response else None))
        
        # Add callback
        self.comm_manager.add_delivery_callback("agent2", delivery_callback)
        
        # Send message (this will trigger callback in the actual async execution)
        # For this test, we'll simulate the callback call
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test message",
            message_type=MessageType.AGENT_REQUEST,
        )
        
        response = Response(
            message_id=message.id,
            agent_id="agent2",
            content="Test response",
        )
        
        # Simulate callback execution
        delivery_callback(message, response)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == message.id
        assert callback_calls[0][1] == "agent2"
        
        # Remove callback
        self.comm_manager.remove_delivery_callback("agent2", delivery_callback)
    
    @pytest.mark.asyncio
    async def test_queue_management(self):
        """Test message queue management operations."""
        self.comm_manager.register_agent(self.agent1)
        
        # Check initial queue size
        queue_size = await self.comm_manager.get_queue_size("agent1")
        assert queue_size == 0
        
        # Clear empty queue (should not raise error)
        await self.comm_manager.clear_queue("agent1")
        
        # Test with invalid agent
        with pytest.raises(ValueError, match="Agent invalid_agent not registered"):
            await self.comm_manager.get_queue_size("invalid_agent")
        
        with pytest.raises(ValueError, match="Agent invalid_agent not registered"):
            await self.comm_manager.clear_queue("invalid_agent")
    
    def test_delivery_statistics(self):
        """Test delivery statistics tracking."""
        # Get initial statistics
        stats = self.comm_manager.get_delivery_statistics()
        assert stats["messages_sent"] == 0
        assert stats["messages_delivered"] == 0
        assert stats["messages_failed"] == 0
        assert stats["average_delivery_time"] == 0.0
        
        # Reset statistics
        self.comm_manager.reset_statistics()
        stats = self.comm_manager.get_delivery_statistics()
        assert all(value == 0 or value == 0.0 for value in stats.values())
    
    def test_shutdown(self):
        """Test communication manager shutdown."""
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        
        # Create shared context
        context_id = self.comm_manager.create_shared_context(["agent1", "agent2"])
        
        # Verify setup
        assert len(self.comm_manager.get_registered_agents()) == 2
        assert self.comm_manager.get_shared_context(context_id) is not None
        
        # Shutdown
        self.comm_manager.shutdown()
        
        # Verify cleanup
        assert len(self.comm_manager.get_registered_agents()) == 0
        assert self.comm_manager.get_shared_context(context_id) is None


@pytest.mark.asyncio
async def test_integration_agent_communication_flow():
    """Integration test for complete agent communication flow."""
    comm_manager = AgentCommunicationManager()
    
    # Create and register agents
    analyst = MockAgent("analyst", AgentType.ANALYST)
    architect = MockAgent("architect", AgentType.ARCHITECT)
    developer = MockAgent("developer", AgentType.DEVELOPER)
    
    comm_manager.register_agent(analyst)
    comm_manager.register_agent(architect)
    comm_manager.register_agent(developer)
    
    # Create shared context for collaboration
    context_id = comm_manager.create_shared_context(["analyst", "architect", "developer"])
    
    # Simulate multi-agent workflow
    # 1. User question comes to analyst
    user_message = Message(
        sender_id="user",
        recipient_id="analyst",
        content="How should I design a microservices architecture?",
        message_type=MessageType.USER_QUESTION,
    )
    
    # Add to shared context
    comm_manager.add_to_shared_context(context_id, user_message)
    
    # 2. Analyst routes to architect
    architect_response = await comm_manager.send_message(
        sender_id="analyst",
        recipient_id="architect",
        content="Please provide microservices architecture design",
        message_type=MessageType.AGENT_REQUEST,
        wait_for_response=True,
    )
    
    assert architect_response is not None
    assert architect_response.agent_id == "architect"
    
    # Add response to shared context
    comm_manager.add_response_to_shared_context(context_id, architect_response)
    
    # 3. Architect collaborates with developer
    developer_response = await comm_manager.send_message(
        sender_id="architect",
        recipient_id="developer",
        content="Please provide implementation details for the architecture",
        message_type=MessageType.AGENT_REQUEST,
        wait_for_response=True,
    )
    
    assert developer_response is not None
    assert developer_response.agent_id == "developer"
    
    # Add response to shared context
    comm_manager.add_response_to_shared_context(context_id, developer_response)
    
    # Verify shared context contains all interactions
    shared_context = comm_manager.get_shared_context(context_id)
    assert len(shared_context.messages) == 1  # Original user message
    assert len(shared_context.responses) == 2  # Architect and developer responses
    
    # Verify agents processed messages
    assert len(architect.processed_messages) == 1
    assert len(developer.processed_messages) == 1
    
    # Check delivery statistics
    stats = comm_manager.get_delivery_statistics()
    assert stats["messages_sent"] == 2
    assert stats["messages_delivered"] == 2
    assert stats["messages_failed"] == 0
    
    # Cleanup
    comm_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])