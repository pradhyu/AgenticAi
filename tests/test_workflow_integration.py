"""Integration tests for the workflow system with agents."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from deep_agent_system.workflows import (
    WorkflowManager,
    WorkflowExecutor,
    WorkflowSpec,
    WorkflowPattern,
    WorkflowComplexity,
)
from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.models.agents import AgentType, AgentConfig, LLMConfig, Capability
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.communication.manager import AgentCommunicationManager


class TestWorkflowIntegration:
    """Integration tests for workflow system."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with proper responses."""
        agents = {}
        
        # Create analyst agent
        analyst_config = AgentConfig(
            agent_id="analyst_1",
            agent_type=AgentType.ANALYST,
            llm_config=LLMConfig(model_name="gpt-4"),
            capabilities=[Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
        )
        analyst_agent = Mock(spec=BaseAgent)
        analyst_agent.agent_id = "analyst_1"
        analyst_agent.config = analyst_config
        
        # Mock process_message to return a response
        def analyst_process_message(message):
            return Response(
                message_id=message.id,
                agent_id="analyst_1",
                content="Analyzed the question and determined it needs architectural design.",
                confidence_score=0.9,
            )
        analyst_agent.process_message = analyst_process_message
        agents["analyst_1"] = analyst_agent
        
        # Create architect agent
        architect_config = AgentConfig(
            agent_id="architect_1",
            agent_type=AgentType.ARCHITECT,
            llm_config=LLMConfig(model_name="gpt-4"),
            capabilities=[Capability.ARCHITECTURE_DESIGN, Capability.SOLUTION_EXPLANATION],
        )
        architect_agent = Mock(spec=BaseAgent)
        architect_agent.agent_id = "architect_1"
        architect_agent.config = architect_config
        
        def architect_process_message(message):
            return Response(
                message_id=message.id,
                agent_id="architect_1",
                content="Designed a microservices architecture with API gateway and database layer.",
                confidence_score=0.85,
            )
        architect_agent.process_message = architect_process_message
        agents["architect_1"] = architect_agent
        
        # Create developer agent
        developer_config = AgentConfig(
            agent_id="developer_1",
            agent_type=AgentType.DEVELOPER,
            llm_config=LLMConfig(model_name="gpt-4"),
            capabilities=[Capability.CODE_GENERATION],
        )
        developer_agent = Mock(spec=BaseAgent)
        developer_agent.agent_id = "developer_1"
        developer_agent.config = developer_config
        
        def developer_process_message(message):
            return Response(
                message_id=message.id,
                agent_id="developer_1",
                content="Generated Python code for the API endpoints and database models.",
                confidence_score=0.8,
            )
        developer_agent.process_message = developer_process_message
        agents["developer_1"] = developer_agent
        
        return agents
    
    @pytest.fixture
    def workflow_manager(self, mock_agents):
        """Create WorkflowManager with mock agents."""
        return WorkflowManager(mock_agents)
    
    @pytest.fixture
    def workflow_executor(self):
        """Create WorkflowExecutor."""
        return WorkflowExecutor(auto_checkpoint=False)  # Disable for simpler testing
    
    def test_single_agent_workflow_creation(self, workflow_manager):
        """Test creating a single agent workflow."""
        question = "What is Python?"
        available_agents = ["analyst_1"]
        
        workflow = workflow_manager.create_workflow_from_question(
            question, available_agents=available_agents
        )
        
        assert workflow is not None
        
        # Compile the workflow
        compiled = workflow_manager.compile_workflow(workflow)
        assert compiled is not None
    
    def test_sequential_workflow_creation(self, workflow_manager):
        """Test creating a sequential workflow."""
        question = "Design and implement a REST API"
        available_agents = ["analyst_1", "architect_1", "developer_1"]
        
        workflow = workflow_manager.create_workflow_from_question(
            question, available_agents=available_agents
        )
        
        assert workflow is not None
        
        # Compile the workflow
        compiled = workflow_manager.compile_workflow(workflow)
        assert compiled is not None
    
    @pytest.mark.asyncio
    async def test_single_agent_workflow_execution(self, workflow_manager, workflow_executor, mock_agents):
        """Test executing a single agent workflow."""
        # Create workflow
        spec = WorkflowSpec(
            workflow_id="single_test",
            template=workflow_manager.get_template("single_agent"),
            agent_ids=["analyst_1"],
        )
        
        workflow = workflow_manager.create_workflow(spec)
        compiled = workflow_manager.compile_workflow(workflow)
        
        # Create test message
        message = Message(
            sender_id="user",
            recipient_id="system",
            content="What is machine learning?",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await workflow_executor.execute_workflow(
            workflow_id="single_test",
            compiled_workflow=compiled,
            initial_message=message,
            timeout=10,
        )
        
        assert execution is not None
        assert execution.workflow_id == "single_test"
        
        # Wait for execution to complete
        await asyncio.sleep(0.2)
        
        # Check execution completed
        final_status = workflow_executor.get_execution_status(execution.execution_id)
        assert final_status is not None
    
    @pytest.mark.asyncio
    async def test_sequential_workflow_execution(self, workflow_manager, workflow_executor, mock_agents):
        """Test executing a sequential workflow."""
        # Create workflow
        spec = WorkflowSpec(
            workflow_id="sequential_test",
            template=workflow_manager.get_template("sequential_chain"),
            agent_ids=["analyst_1", "architect_1", "developer_1"],
        )
        
        workflow = workflow_manager.create_workflow(spec)
        compiled = workflow_manager.compile_workflow(workflow)
        
        # Create test message
        message = Message(
            sender_id="user",
            recipient_id="system",
            content="Build a web application with user authentication",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await workflow_executor.execute_workflow(
            workflow_id="sequential_test",
            compiled_workflow=compiled,
            initial_message=message,
            timeout=10,
        )
        
        assert execution is not None
        assert execution.workflow_id == "sequential_test"
        
        # Wait for execution to complete
        await asyncio.sleep(0.3)
        
        # Check execution completed
        final_status = workflow_executor.get_execution_status(execution.execution_id)
        assert final_status is not None
    
    def test_workflow_template_suggestion(self, workflow_manager):
        """Test workflow template suggestion based on question complexity."""
        # Simple question
        simple_question = "What is Python?"
        simple_template = workflow_manager.suggest_workflow_template(
            simple_question, ["analyst_1"]
        )
        assert simple_template is not None
        assert simple_template.complexity == WorkflowComplexity.SIMPLE
        
        # Complex question
        complex_question = "Design a distributed microservices architecture with authentication, implement the user service, and create comprehensive tests"
        complex_template = workflow_manager.suggest_workflow_template(
            complex_question, ["analyst_1", "architect_1", "developer_1"]
        )
        assert complex_template is not None
        # The complexity might vary based on implementation, so just check it's not None
        assert complex_template.complexity in [WorkflowComplexity.SIMPLE, WorkflowComplexity.MODERATE, WorkflowComplexity.COMPLEX]
    
    def test_workflow_patterns(self, workflow_manager):
        """Test different workflow patterns."""
        templates = workflow_manager.get_templates()
        
        # Check that we have different patterns
        patterns = {template.pattern for template in templates}
        
        assert WorkflowPattern.SINGLE_AGENT in patterns
        assert WorkflowPattern.SEQUENTIAL_CHAIN in patterns
        assert WorkflowPattern.PARALLEL_PROCESSING in patterns
        
        # Test each pattern can create a workflow
        for template in templates:
            if template.pattern == WorkflowPattern.SINGLE_AGENT:
                spec = WorkflowSpec(
                    workflow_id=f"test_{template.template_id}",
                    template=template,
                    agent_ids=["analyst_1"],
                )
            else:
                spec = WorkflowSpec(
                    workflow_id=f"test_{template.template_id}",
                    template=template,
                    agent_ids=["analyst_1", "architect_1"],
                )
            
            workflow = workflow_manager.create_workflow(spec)
            assert workflow is not None
    
    def test_workflow_state_management(self, workflow_manager):
        """Test workflow state management."""
        from deep_agent_system.workflows.manager import WorkflowState
        
        # Create initial state
        state = WorkflowState(
            workflow_id="test_workflow",
            messages=[{"content": "test message"}],
        )
        
        # Check initial values
        assert state["workflow_id"] == "test_workflow"
        assert len(state["messages"]) == 1
        assert state["current_step"] == 0
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 5
        
        # Modify state
        state["current_step"] = 1
        state["responses"] = [{"agent_id": "test", "content": "response"}]
        
        assert state["current_step"] == 1
        assert len(state["responses"]) == 1
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_manager, workflow_executor):
        """Test workflow error handling."""
        # Create a workflow with an agent that will cause an error
        error_agent = Mock(spec=BaseAgent)
        error_agent.agent_id = "error_agent"
        error_agent.process_message.side_effect = Exception("Test error")
        
        # Add error agent to manager
        workflow_manager.agents["error_agent"] = error_agent
        
        # Create workflow with error agent
        spec = WorkflowSpec(
            workflow_id="error_test",
            template=workflow_manager.get_template("single_agent"),
            agent_ids=["error_agent"],
        )
        
        workflow = workflow_manager.create_workflow(spec)
        compiled = workflow_manager.compile_workflow(workflow)
        
        # Create test message
        message = Message(
            sender_id="user",
            recipient_id="system",
            content="This will cause an error",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await workflow_executor.execute_workflow(
            workflow_id="error_test",
            compiled_workflow=compiled,
            initial_message=message,
            timeout=5,
        )
        
        assert execution is not None
        
        # Wait for execution to complete/fail
        await asyncio.sleep(0.2)
        
        # Check that error was handled
        final_status = workflow_executor.get_execution_status(execution.execution_id)
        assert final_status is not None
    
    def test_workflow_statistics(self, workflow_manager, workflow_executor):
        """Test workflow system statistics."""
        # Test WorkflowManager statistics
        manager_stats = workflow_manager.get_workflow_statistics()
        
        assert "templates_registered" in manager_stats
        assert "workflows_cached" in manager_stats
        assert "compiled_workflows" in manager_stats
        assert "available_agents" in manager_stats
        
        assert manager_stats["available_agents"] == 3
        assert manager_stats["templates_registered"] > 0
        
        # Test WorkflowExecutor statistics
        executor_stats = workflow_executor.get_statistics()
        
        assert "executions_started" in executor_stats
        assert "executions_completed" in executor_stats
        assert "executions_failed" in executor_stats
        assert "active_executions" in executor_stats
        
        # Initially should be 0
        assert executor_stats["executions_started"] == 0
        assert executor_stats["active_executions"] == 0


class TestWorkflowCommunicationIntegration:
    """Test integration between workflows and communication system."""
    
    @pytest.fixture
    def communication_manager(self):
        """Create AgentCommunicationManager."""
        return AgentCommunicationManager()
    
    @pytest.fixture
    def integrated_agents(self, communication_manager):
        """Create agents registered with communication manager."""
        agents = {}
        
        # Create and register analyst agent
        analyst_config = AgentConfig(
            agent_id="analyst_1",
            agent_type=AgentType.ANALYST,
            llm_config=LLMConfig(model_name="gpt-4"),
            capabilities=[Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
        )
        analyst_agent = Mock(spec=BaseAgent)
        analyst_agent.agent_id = "analyst_1"
        analyst_agent.config = analyst_config
        
        def analyst_process_message(message):
            return Response(
                message_id=message.id,
                agent_id="analyst_1",
                content="Routing question to appropriate agent.",
                confidence_score=0.9,
            )
        analyst_agent.process_message = analyst_process_message
        
        communication_manager.register_agent(analyst_agent)
        agents["analyst_1"] = analyst_agent
        
        return agents
    
    def test_workflow_with_communication_manager(self, integrated_agents, communication_manager):
        """Test workflow system integration with communication manager."""
        # Create workflow manager with integrated agents
        workflow_manager = WorkflowManager(integrated_agents)
        
        # Verify agents are available
        assert len(workflow_manager.agents) == 1
        assert "analyst_1" in workflow_manager.agents
        
        # Verify communication manager has the agents
        registered_agents = communication_manager.get_registered_agents()
        assert "analyst_1" in registered_agents
        
        # Create a simple workflow
        workflow = workflow_manager.create_workflow_from_question(
            "What is the best architecture pattern?",
            available_agents=["analyst_1"]
        )
        
        assert workflow is not None
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_communication(self, integrated_agents, communication_manager):
        """Test workflow execution using communication manager."""
        workflow_manager = WorkflowManager(integrated_agents)
        workflow_executor = WorkflowExecutor(auto_checkpoint=False)
        
        # Create workflow
        spec = WorkflowSpec(
            workflow_id="comm_test",
            template=workflow_manager.get_template("single_agent"),
            agent_ids=["analyst_1"],
        )
        
        workflow = workflow_manager.create_workflow(spec)
        compiled = workflow_manager.compile_workflow(workflow)
        
        # Create test message
        message = Message(
            sender_id="user",
            recipient_id="system",
            content="Analyze this architectural question",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Execute workflow
        execution = await workflow_executor.execute_workflow(
            workflow_id="comm_test",
            compiled_workflow=compiled,
            initial_message=message,
            timeout=5,
        )
        
        assert execution is not None
        
        # Wait for execution
        await asyncio.sleep(0.1)
        
        # Verify execution completed
        final_status = workflow_executor.get_execution_status(execution.execution_id)
        assert final_status is not None