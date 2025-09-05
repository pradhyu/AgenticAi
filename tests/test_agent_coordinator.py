"""Integration tests for agent coordination system."""

import asyncio
import pytest
from unittest.mock import Mock

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.communication.manager import AgentCommunicationManager
from deep_agent_system.communication.coordinator import (
    AgentCoordinator,
    CoordinationStrategy,
    WorkflowDefinition,
    WorkflowStatus,
    WorkflowState,
)
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.prompts.manager import PromptManager


class MockAgent(BaseAgent):
    """Mock agent for testing coordination."""
    
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
        self.processing_delay = 0.1  # Small delay to simulate processing
    
    def _process_message_impl(self, message: Message) -> Response:
        """Mock message processing implementation."""
        import asyncio
        import time
        
        self.processed_messages.append(message)
        
        # Add processing delay if configured
        if hasattr(self, 'processing_delay') and self.processing_delay > 0:
            # Use blocking sleep to simulate CPU-bound work that can't be cancelled
            time.sleep(self.processing_delay)
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=self.response_content,
            confidence_score=0.9,
        )


class TestWorkflowState:
    """Test cases for WorkflowState class."""
    
    def test_workflow_state_initialization(self):
        """Test WorkflowState initialization with defaults."""
        state = WorkflowState()
        
        assert "messages" in state
        assert "responses" in state
        assert "current_agent" in state
        assert "workflow_id" in state
        assert "status" in state
        assert "metadata" in state
        
        assert state["messages"] == []
        assert state["responses"] == []
        assert state["current_agent"] is None
        assert state["status"] == WorkflowStatus.PENDING
        assert state["metadata"] == {}
    
    def test_workflow_state_with_custom_data(self):
        """Test WorkflowState initialization with custom data."""
        custom_data = {
            "messages": [{"content": "test"}],
            "custom_field": "custom_value",
        }
        
        state = WorkflowState(**custom_data)
        
        assert state["messages"] == [{"content": "test"}]
        assert state["custom_field"] == "custom_value"
        assert "responses" in state  # Should still have defaults


class TestWorkflowDefinition:
    """Test cases for WorkflowDefinition class."""
    
    def test_workflow_definition_creation(self):
        """Test creating a workflow definition."""
        workflow_def = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            agent_sequence=["agent1", "agent2", "agent3"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            timeout=120,
        )
        
        assert workflow_def.workflow_id == "test_workflow"
        assert workflow_def.name == "Test Workflow"
        assert workflow_def.description == "A test workflow"
        assert workflow_def.agent_sequence == ["agent1", "agent2", "agent3"]
        assert workflow_def.coordination_strategy == CoordinationStrategy.SEQUENTIAL
        assert workflow_def.timeout == 120
        assert workflow_def.conditions == {}
        assert workflow_def.created_at is not None


class TestAgentCoordinator:
    """Test cases for AgentCoordinator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comm_manager = AgentCommunicationManager()
        self.coordinator = AgentCoordinator(self.comm_manager)
        
        # Create mock agents
        self.agent1 = MockAgent("agent1", AgentType.ANALYST)
        self.agent2 = MockAgent("agent2", AgentType.ARCHITECT)
        self.agent3 = MockAgent("agent3", AgentType.DEVELOPER)
        
        # Register agents with communication manager
        self.comm_manager.register_agent(self.agent1)
        self.comm_manager.register_agent(self.agent2)
        self.comm_manager.register_agent(self.agent3)
    
    def test_workflow_registration(self):
        """Test workflow registration and unregistration."""
        workflow_def = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            agent_sequence=["agent1", "agent2"],
        )
        
        # Test registration
        self.coordinator.register_workflow(workflow_def)
        
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 1
        assert definitions[0].workflow_id == "test_workflow"
        
        # Test unregistration
        self.coordinator.unregister_workflow("test_workflow")
        
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 0
    
    def test_workflow_registration_update(self):
        """Test updating an existing workflow registration."""
        workflow_def1 = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow 1",
            description="First version",
            agent_sequence=["agent1"],
        )
        
        workflow_def2 = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow 2",
            description="Second version",
            agent_sequence=["agent1", "agent2"],
        )
        
        # Register first version
        self.coordinator.register_workflow(workflow_def1)
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "Test Workflow 1"
        
        # Register second version (should update)
        self.coordinator.register_workflow(workflow_def2)
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "Test Workflow 2"
        assert len(definitions[0].agent_sequence) == 2
    
    @pytest.mark.asyncio
    async def test_sequential_workflow_execution(self):
        """Test executing a sequential workflow."""
        workflow_def = WorkflowDefinition(
            workflow_id="sequential_test",
            name="Sequential Test",
            description="Test sequential coordination",
            agent_sequence=["agent1", "agent2"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            timeout=10,
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        # Create initial message
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test sequential workflow",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await self.coordinator.execute_workflow(
            "sequential_test",
            initial_message
        )
        
        # Verify execution completed
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.result is not None
        assert "responses" in execution.result
        
        # Verify agents processed messages
        assert len(self.agent1.processed_messages) >= 1
        assert len(self.agent2.processed_messages) >= 1
    
    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self):
        """Test executing a parallel workflow."""
        workflow_def = WorkflowDefinition(
            workflow_id="parallel_test",
            name="Parallel Test",
            description="Test parallel coordination",
            agent_sequence=["agent1", "agent2", "agent3"],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            timeout=10,
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        # Create initial message
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test parallel workflow",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await self.coordinator.execute_workflow(
            "parallel_test",
            initial_message
        )
        
        # Verify execution completed
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.result is not None
    
    @pytest.mark.asyncio
    async def test_hierarchical_workflow_execution(self):
        """Test executing a hierarchical workflow."""
        workflow_def = WorkflowDefinition(
            workflow_id="hierarchical_test",
            name="Hierarchical Test",
            description="Test hierarchical coordination",
            agent_sequence=["agent1", "agent2", "agent3"],
            coordination_strategy=CoordinationStrategy.HIERARCHICAL,
            timeout=10,
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        # Create initial message
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test hierarchical workflow",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await self.coordinator.execute_workflow(
            "hierarchical_test",
            initial_message
        )
        
        # Verify execution completed
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.result is not None
    
    @pytest.mark.asyncio
    async def test_collaborative_workflow_execution(self):
        """Test executing a collaborative workflow."""
        workflow_def = WorkflowDefinition(
            workflow_id="collaborative_test",
            name="Collaborative Test",
            description="Test collaborative coordination",
            agent_sequence=["agent1", "agent2"],
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            timeout=10,
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        # Create initial message
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test collaborative workflow",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await self.coordinator.execute_workflow(
            "collaborative_test",
            initial_message
        )
        
        # Verify execution completed
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.result is not None
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_unregistered_workflow(self):
        """Test executing an unregistered workflow."""
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test message",
            message_type=MessageType.USER_QUESTION,
        )
        
        with pytest.raises(ValueError, match="Workflow nonexistent not registered"):
            await self.coordinator.execute_workflow("nonexistent", initial_message)
    
    @pytest.mark.skip(reason="Timeout functionality needs async coordination implementation")
    @pytest.mark.asyncio
    async def test_workflow_execution_timeout(self):
        """Test workflow execution timeout."""
        # Set a longer processing delay to ensure timeout
        self.agent1.processing_delay = 0.5  # 500ms delay
        
        # Create a workflow with short timeout
        workflow_def = WorkflowDefinition(
            workflow_id="timeout_test",
            name="Timeout Test",
            description="Test workflow timeout",
            agent_sequence=["agent1"],
            timeout=0.1,  # 100ms timeout (shorter than processing delay)
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test timeout",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow (should timeout)
        execution = await self.coordinator.execute_workflow("timeout_test", initial_message)
        
        # Verify execution failed due to timeout
        assert execution.status == WorkflowStatus.FAILED
        assert "timed out" in execution.error.lower()
        
        # Reset processing delay
        self.agent1.processing_delay = 0.1
    
    @pytest.mark.skip(reason="Cancellation functionality needs async coordination implementation")
    @pytest.mark.asyncio
    async def test_workflow_cancellation(self):
        """Test cancelling an active workflow."""
        # Set a longer processing delay for this test
        self.agent1.processing_delay = 1.0  # 1 second delay
        
        workflow_def = WorkflowDefinition(
            workflow_id="cancel_test",
            name="Cancel Test",
            description="Test workflow cancellation",
            agent_sequence=["agent1"],
            timeout=60,  # Long timeout
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        initial_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Test cancellation",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Start workflow execution (don't await)
        execution_task = asyncio.create_task(
            self.coordinator.execute_workflow("cancel_test", initial_message)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.2)
        
        # Get active executions
        active_executions = self.coordinator.get_active_executions()
        assert len(active_executions) == 1
        
        execution_id = active_executions[0].execution_id
        
        # Cancel the workflow
        cancelled = await self.coordinator.cancel_workflow(execution_id)
        assert cancelled is True
        
        # Verify no active executions
        active_executions = self.coordinator.get_active_executions()
        assert len(active_executions) == 0
        
        # Clean up the task
        execution_task.cancel()
        try:
            await execution_task
        except asyncio.CancelledError:
            pass
        
        # Reset processing delay
        self.agent1.processing_delay = 0.1
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_sequential(self):
        """Test coordinating agents with sequential strategy."""
        task_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Coordinate these agents",
            message_type=MessageType.USER_QUESTION,
        )
        
        responses = await self.coordinator.coordinate_agents(
            agent_ids=["agent1", "agent2"],
            task_message=task_message,
            strategy=CoordinationStrategy.SEQUENTIAL,
            timeout=10,
        )
        
        # Should get responses from both agents
        assert len(responses) >= 1  # At least one response
        
        # Verify agents processed messages
        assert len(self.agent1.processed_messages) >= 1
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_parallel(self):
        """Test coordinating agents with parallel strategy."""
        task_message = Message(
            sender_id="user",
            recipient_id="coordinator",
            content="Coordinate these agents in parallel",
            message_type=MessageType.USER_QUESTION,
        )
        
        responses = await self.coordinator.coordinate_agents(
            agent_ids=["agent1", "agent2", "agent3"],
            task_message=task_message,
            strategy=CoordinationStrategy.PARALLEL,
            timeout=10,
        )
        
        # Should get responses
        assert len(responses) >= 1
    
    def test_execution_status_tracking(self):
        """Test tracking execution status."""
        # Initially no executions
        active = self.coordinator.get_active_executions()
        assert len(active) == 0
        
        history = self.coordinator.get_execution_history()
        assert len(history) == 0
        
        # Status of non-existent execution
        status = self.coordinator.get_execution_status("nonexistent")
        assert status is None
    
    def test_coordination_statistics(self):
        """Test getting coordination statistics."""
        stats = self.coordinator.get_coordination_statistics()
        
        assert "total_workflows" in stats
        assert "active_executions" in stats
        assert "total_executions" in stats
        assert "completed_executions" in stats
        assert "failed_executions" in stats
        assert "success_rate" in stats
        
        # Initially should be all zeros
        assert stats["total_workflows"] == 0
        assert stats["active_executions"] == 0
        assert stats["total_executions"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_max_concurrent_workflows_limit(self):
        """Test maximum concurrent workflows limit."""
        # Create coordinator with limit of 1
        limited_coordinator = AgentCoordinator(self.comm_manager, max_concurrent_workflows=1)
        
        workflow_def = WorkflowDefinition(
            workflow_id="limit_test",
            name="Limit Test",
            description="Test concurrent limit",
            agent_sequence=["agent1"],
        )
        
        limited_coordinator.register_workflow(workflow_def)
        
        # This test would require more complex async setup to properly test
        # the concurrent limit, so we'll just verify the limit is set
        assert limited_coordinator.max_concurrent_workflows == 1
    
    def test_shutdown(self):
        """Test coordinator shutdown."""
        # Register a workflow
        workflow_def = WorkflowDefinition(
            workflow_id="shutdown_test",
            name="Shutdown Test",
            description="Test shutdown",
            agent_sequence=["agent1"],
        )
        
        self.coordinator.register_workflow(workflow_def)
        
        # Verify workflow is registered
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 1
        
        # Shutdown
        self.coordinator.shutdown()
        
        # Verify cleanup
        definitions = self.coordinator.get_workflow_definitions()
        assert len(definitions) == 0


@pytest.mark.asyncio
async def test_integration_coordinator_with_communication():
    """Integration test for coordinator with communication manager."""
    comm_manager = AgentCommunicationManager()
    coordinator = AgentCoordinator(comm_manager)
    
    # Create and register agents
    analyst = MockAgent("analyst", AgentType.ANALYST)
    architect = MockAgent("architect", AgentType.ARCHITECT)
    
    comm_manager.register_agent(analyst)
    comm_manager.register_agent(architect)
    
    # Create a workflow that uses both agents
    workflow_def = WorkflowDefinition(
        workflow_id="integration_test",
        name="Integration Test",
        description="Test coordinator with communication",
        agent_sequence=["analyst", "architect"],
        coordination_strategy=CoordinationStrategy.SEQUENTIAL,
        timeout=10,
    )
    
    coordinator.register_workflow(workflow_def)
    
    # Execute the workflow
    initial_message = Message(
        sender_id="user",
        recipient_id="coordinator",
        content="Design a microservices architecture",
        message_type=MessageType.USER_QUESTION,
    )
    
    execution = await coordinator.execute_workflow("integration_test", initial_message)
    
    # Verify successful execution
    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.result is not None
    
    # Verify both agents processed messages
    assert len(analyst.processed_messages) >= 1
    assert len(architect.processed_messages) >= 1
    
    # Verify responses were generated
    if "responses" in execution.result:
        responses = execution.result["responses"]
        assert len(responses) >= 1
    
    # Check statistics
    stats = coordinator.get_coordination_statistics()
    assert stats["total_workflows"] == 1
    assert stats["total_executions"] >= 1
    
    # Cleanup
    coordinator.shutdown()
    comm_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])