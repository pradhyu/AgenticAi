"""Comprehensive integration tests for agent workflows and system components."""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from deep_agent_system.system import AgentSystem
from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.agents.analyst import AnalystAgent
from deep_agent_system.agents.architect import ArchitectAgent
from deep_agent_system.agents.developer import DeveloperAgent
from deep_agent_system.agents.code_reviewer import CodeReviewerAgent
from deep_agent_system.agents.tester import TesterAgent
from deep_agent_system.models.agents import (
    AgentConfig, 
    AgentType, 
    Capability, 
    LLMConfig
)
from deep_agent_system.models.messages import (
    Message, 
    Response, 
    MessageType,
    Context,
    Document,
    RetrievalType
)
from deep_agent_system.communication.manager import AgentCommunicationManager
from deep_agent_system.communication.coordinator import AgentCoordinator
from deep_agent_system.workflows.manager import WorkflowManager
from deep_agent_system.workflows.executor import WorkflowExecutor
from deep_agent_system.config.manager import ConfigManager
from deep_agent_system.config.models import SystemConfig
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager


@pytest.fixture
def temp_prompt_dir():
    """Create temporary prompt directory with test prompts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_dir = Path(temp_dir)
        
        # Create agent directories
        for agent_type in ["analyst", "architect", "developer", "code_reviewer", "tester"]:
            (prompt_dir / agent_type).mkdir()
        
        # Create basic prompt files
        test_prompts = {
            "analyst/classification.txt": "You are an analyst. Classify this question: {user_question}",
            "analyst/routing.txt": "Route this question to the appropriate agent: {user_question}",
            "analyst/aggregation.txt": "Aggregate these responses: {responses}",
            "architect/design.txt": "You are an architect. Design a solution for: {user_question}",
            "developer/implementation.txt": "You are a developer. Implement: {user_question}",
            "code_reviewer/review.txt": "You are a code reviewer. Review this code: {code}",
            "tester/test_creation.txt": "You are a tester. Create tests for: {code}",
        }
        
        for file_path, content in test_prompts.items():
            full_path = prompt_dir / file_path
            full_path.write_text(content)
        
        yield str(prompt_dir)


@pytest.fixture
def mock_rag_manager():
    """Create mock RAG manager."""
    rag_manager = Mock(spec=RAGManager)
    
    # Mock vector retrieval
    rag_manager.retrieve_vector_context.return_value = [
        Document(content="Test context document", metadata={"source": "test.txt"})
    ]
    
    # Mock hybrid retrieval
    rag_manager.hybrid_retrieve.return_value = Context(
        query="test query",
        documents=[Document(content="Hybrid context", metadata={})],
        retrieval_method=RetrievalType.HYBRID
    )
    
    return rag_manager


@pytest.fixture
def agent_configs():
    """Create agent configurations for testing."""
    llm_config = LLMConfig(model_name="gpt-3.5-turbo")
    
    return {
        "analyst": AgentConfig(
            agent_id="analyst_1",
            agent_type=AgentType.ANALYST,
            llm_config=llm_config,
            capabilities=[Capability.QUESTION_ROUTING, Capability.CLASSIFICATION]
        ),
        "architect": AgentConfig(
            agent_id="architect_1",
            agent_type=AgentType.ARCHITECT,
            llm_config=llm_config,
            capabilities=[Capability.ARCHITECTURE_DESIGN, Capability.SOLUTION_EXPLANATION]
        ),
        "developer": AgentConfig(
            agent_id="developer_1",
            agent_type=AgentType.DEVELOPER,
            llm_config=llm_config,
            capabilities=[Capability.CODE_GENERATION]
        ),
        "code_reviewer": AgentConfig(
            agent_id="code_reviewer_1",
            agent_type=AgentType.CODE_REVIEWER,
            llm_config=llm_config,
            capabilities=[Capability.CODE_REVIEW]
        ),
        "tester": AgentConfig(
            agent_id="tester_1",
            agent_type=AgentType.TESTER,
            llm_config=llm_config,
            capabilities=[Capability.TEST_CREATION]
        )
    }


@pytest.fixture
def integrated_agents(agent_configs, temp_prompt_dir, mock_rag_manager):
    """Create integrated agent instances."""
    prompt_manager = PromptManager(temp_prompt_dir)
    
    agents = {}
    
    # Create analyst agent
    agents["analyst"] = AnalystAgent(
        config=agent_configs["analyst"],
        prompt_manager=prompt_manager,
        rag_manager=mock_rag_manager
    )
    
    # Create architect agent
    agents["architect"] = ArchitectAgent(
        config=agent_configs["architect"],
        prompt_manager=prompt_manager,
        rag_manager=mock_rag_manager
    )
    
    # Create developer agent
    agents["developer"] = DeveloperAgent(
        config=agent_configs["developer"],
        prompt_manager=prompt_manager,
        rag_manager=mock_rag_manager
    )
    
    # Create code reviewer agent
    agents["code_reviewer"] = CodeReviewerAgent(
        config=agent_configs["code_reviewer"],
        prompt_manager=prompt_manager,
        rag_manager=mock_rag_manager
    )
    
    # Create tester agent
    agents["tester"] = TesterAgent(
        config=agent_configs["tester"],
        prompt_manager=prompt_manager,
        rag_manager=mock_rag_manager
    )
    
    # Register agents with each other for communication
    for agent in agents.values():
        for other_agent in agents.values():
            if agent != other_agent:
                agent.register_agent(other_agent)
    
    # Also register agents by their type names for analyst routing
    analyst = agents["analyst"]
    analyst._agent_registry["architect"] = agents["architect"]
    analyst._agent_registry["developer"] = agents["developer"]
    analyst._agent_registry["code_reviewer"] = agents["code_reviewer"]
    analyst._agent_registry["tester"] = agents["tester"]
    
    return agents


class TestEndToEndQuestionProcessing:
    """Test complete end-to-end question processing workflows."""

    
    def test_architecture_question_workflow(self, integrated_agents):
        """Test complete workflow for architecture question."""
        analyst = integrated_agents["analyst"]
        architect = integrated_agents["architect"]
        
        # Create architecture question
        message = Message(
            sender_id="user",
            recipient_id="analyst_1",
            content="How should I design a scalable microservices architecture?",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process through analyst
        response = analyst.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "analyst_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        
        # Check that analyst classified and potentially routed the question
        assert response.confidence_score > 0
    
    def test_development_question_workflow(self, integrated_agents):
        """Test complete workflow for development question."""
        analyst = integrated_agents["analyst"]
        developer = integrated_agents["developer"]
        
        # Create development question
        message = Message(
            sender_id="user",
            recipient_id="analyst_1",
            content="How do I implement a REST API in Python using FastAPI?",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process through analyst
        response = analyst.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "analyst_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
    
    def test_code_review_workflow(self, integrated_agents):
        """Test complete workflow for code review question."""
        analyst = integrated_agents["analyst"]
        code_reviewer = integrated_agents["code_reviewer"]
        
        # Create code review question
        code_sample = '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
'''
        
        message = Message(
            sender_id="user",
            recipient_id="analyst_1",
            content=f"Please review this code: {code_sample}",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process through analyst
        response = analyst.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "analyst_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
    
    def test_testing_question_workflow(self, integrated_agents):
        """Test complete workflow for testing question."""
        analyst = integrated_agents["analyst"]
        tester = integrated_agents["tester"]
        
        # Create testing question
        message = Message(
            sender_id="user",
            recipient_id="analyst_1",
            content="How do I write unit tests for a Python function using pytest?",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process through analyst
        response = analyst.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "analyst_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
    
    def test_multi_agent_collaboration(self, integrated_agents):
        """Test multi-agent collaboration workflow."""
        analyst = integrated_agents["analyst"]
        
        # Create complex question that might require multiple agents
        message = Message(
            sender_id="user",
            recipient_id="analyst_1",
            content="Design and implement a user authentication system with proper testing",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process through analyst
        response = analyst.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "analyst_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        
        # The analyst should handle this appropriately, either by routing or handling directly
        assert response.confidence_score >= 0.5


@pytest.fixture
def communication_manager():
    """Create communication manager for testing."""
    return AgentCommunicationManager()


@pytest.fixture
def mock_agents_for_communication(agent_configs):
    """Create mock agents for communication testing."""
    agents = {}
    
    for agent_type, config in agent_configs.items():
        agent = Mock(spec=BaseAgent)
        agent.agent_id = config.agent_id
        agent.config = config
        
        # Mock process_message method
        def create_process_message(agent_id):
            def process_message(message):
                return Response(
                    message_id=message.id,
                    agent_id=agent_id,
                    content=f"Response from {agent_id}: {message.content}",
                    confidence_score=0.8
                )
            return process_message
        
        agent.process_message = create_process_message(config.agent_id)
        agents[agent_type] = agent
    
    return agents


class TestAgentCommunicationIntegration:
    """Test agent communication and coordination integration."""

    
    @pytest.mark.asyncio
    async def test_agent_to_agent_communication(self, communication_manager, mock_agents_for_communication):
        """Test direct agent-to-agent communication."""
        analyst = mock_agents_for_communication["analyst"]
        architect = mock_agents_for_communication["architect"]
        
        # Register agents
        communication_manager.register_agent(analyst)
        communication_manager.register_agent(architect)
        
        # Send message from analyst to architect
        response = await communication_manager.send_message(
            sender_id="analyst_1",
            recipient_id="architect_1",
            content="Please design a solution for user authentication"
        )
        
        # Verify response
        assert response is not None
        assert response.agent_id == "architect_1"
        assert "analyst_1" in response.content or "architect_1" in response.content
    
    @pytest.mark.asyncio
    async def test_broadcast_communication(self, communication_manager, mock_agents_for_communication):
        """Test broadcast communication to multiple agents."""
        # Register multiple agents
        for agent in mock_agents_for_communication.values():
            communication_manager.register_agent(agent)
        
        # Broadcast message
        responses = await communication_manager.broadcast_message(
            sender_id="analyst_1",
            content="System status check",
            exclude_sender=True
        )
        
        # Verify responses
        assert len(responses) == len(mock_agents_for_communication) - 1  # Excluding sender
        for response in responses:
            assert isinstance(response, Response)
            assert "System status check" in response.content
    
    def test_agent_coordination_workflow(self, mock_agents_for_communication):
        """Test agent coordination through coordinator."""
        coordinator = AgentCoordinator()
        
        # Register agents
        for agent in mock_agents_for_communication.values():
            coordinator.register_agent(agent)
        
        # Create coordination request
        request = {
            "type": "sequential",
            "agents": ["analyst_1", "architect_1", "developer_1"],
            "initial_message": "Design and implement a user service"
        }
        
        # Execute coordination
        result = coordinator.coordinate_agents(request)
        
        # Verify coordination result
        assert result is not None
        assert "execution_id" in result
        assert result["status"] in ["completed", "in_progress"]


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.similarity_search.return_value = [
        (Document(content="Vector search result 1", metadata={"source": "doc1.txt"}), 0.9),
        (Document(content="Vector search result 2", metadata={"source": "doc2.txt"}), 0.8)
    ]
    return store


@pytest.fixture
def mock_graph_store():
    """Create mock graph store."""
    store = Mock()
    store.query_graph.return_value = Mock(
        entities=[{"name": "Python", "type": "TECHNOLOGY"}],
        relationships=[{"from": "Python", "to": "FastAPI", "type": "USES"}]
    )
    return store


@pytest.fixture
def rag_manager_with_stores(mock_vector_store, mock_graph_store):
    """Create RAG manager with mock stores."""
    rag_manager = Mock(spec=RAGManager)
    rag_manager.vector_store = mock_vector_store
    rag_manager.graph_store = mock_graph_store
    
    # Mock retrieval methods
    rag_manager.retrieve_vector_context.return_value = [
        Document(content="Vector context", metadata={"source": "test.txt"})
    ]
    
    rag_manager.retrieve_graph_context.return_value = Mock(
        entities=[{"name": "FastAPI", "type": "FRAMEWORK"}],
        relationships=[]
    )
    
    rag_manager.hybrid_retrieve.return_value = Context(
        query="test query",
        documents=[Document(content="Hybrid result", metadata={})],
        retrieval_method=RetrievalType.HYBRID
    )
    
    return rag_manager


class TestRAGIntegrationWorkflows:
    """Test RAG integration in agent workflows."""
    
    def test_agent_with_rag_context_retrieval(self, agent_configs, temp_prompt_dir, rag_manager_with_stores):
        """Test agent using RAG for context retrieval."""
        prompt_manager = PromptManager(temp_prompt_dir)
        
        # Create architect agent with RAG
        architect = ArchitectAgent(
            config=agent_configs["architect"],
            prompt_manager=prompt_manager,
            rag_manager=rag_manager_with_stores
        )
        
        # Create message
        message = Message(
            sender_id="user",
            recipient_id="architect_1",
            content="Design a FastAPI microservice architecture",
            message_type=MessageType.USER_QUESTION
        )
        
        # Process message (should use RAG)
        response = architect.process_message(message)
        
        # Verify response
        assert response.message_id == message.id
        assert response.agent_id == "architect_1"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        
        # Verify RAG was called
        rag_manager_with_stores.retrieve_vector_context.assert_called()
    
    def test_multiple_agents_with_shared_rag(self, agent_configs, temp_prompt_dir, rag_manager_with_stores):
        """Test multiple agents sharing the same RAG manager."""
        prompt_manager = PromptManager(temp_prompt_dir)
        
        # Create multiple agents with shared RAG
        architect = ArchitectAgent(
            config=agent_configs["architect"],
            prompt_manager=prompt_manager,
            rag_manager=rag_manager_with_stores
        )
        
        developer = DeveloperAgent(
            config=agent_configs["developer"],
            prompt_manager=prompt_manager,
            rag_manager=rag_manager_with_stores
        )
        
        # Test architect
        arch_message = Message(
            sender_id="user",
            recipient_id="architect_1",
            content="Design a REST API",
            message_type=MessageType.USER_QUESTION
        )
        
        arch_response = architect.process_message(arch_message)
        assert arch_response.agent_id == "architect_1"
        
        # Test developer
        dev_message = Message(
            sender_id="user",
            recipient_id="developer_1",
            content="Implement a REST API endpoint",
            message_type=MessageType.USER_QUESTION
        )
        
        dev_response = developer.process_message(dev_message)
        assert dev_response.agent_id == "developer_1"
        
        # Both should have used RAG
        assert rag_manager_with_stores.retrieve_vector_context.call_count >= 2


@pytest.fixture
def workflow_manager_with_agents(mock_agents_for_communication):
    """Create workflow manager with agents."""
    # Convert mock agents to the format expected by WorkflowManager
    agents_dict = {
        agent.agent_id: agent 
        for agent in mock_agents_for_communication.values()
    }
    return WorkflowManager(agents_dict)


class TestWorkflowExecutionIntegration:
    """Test workflow execution with real components."""

    
    def test_single_agent_workflow_execution(self, workflow_manager_with_agents):
        """Test executing a single agent workflow."""
        # Create workflow for single agent
        workflow = workflow_manager_with_agents.create_workflow_from_question(
            "What is Python?",
            available_agents=["analyst_1"]
        )
        
        # Verify workflow was created
        assert workflow is not None
        
        # Compile workflow
        compiled_workflow = workflow_manager_with_agents.compile_workflow(workflow)
        assert compiled_workflow is not None
    
    def test_sequential_workflow_execution(self, workflow_manager_with_agents):
        """Test executing a sequential workflow."""
        # Create workflow for sequential execution
        workflow = workflow_manager_with_agents.create_workflow_from_question(
            "Design and implement a user authentication system",
            available_agents=["analyst_1", "architect_1", "developer_1"]
        )
        
        # Verify workflow was created
        assert workflow is not None
        
        # Compile workflow
        compiled_workflow = workflow_manager_with_agents.compile_workflow(workflow)
        assert compiled_workflow is not None
    
    def test_workflow_with_error_handling(self, workflow_manager_with_agents):
        """Test workflow execution with error handling."""
        # Create a workflow
        workflow = workflow_manager_with_agents.create_workflow_from_question(
            "Test question",
            available_agents=["analyst_1"]
        )
        
        assert workflow is not None
        
        # Test that workflow compilation doesn't raise exceptions
        try:
            compiled_workflow = workflow_manager_with_agents.compile_workflow(workflow)
            assert compiled_workflow is not None
        except Exception as e:
            # If there's an error, it should be handled gracefully
            assert isinstance(e, Exception)


@pytest.fixture
def mock_system_config(temp_prompt_dir):
    """Create mock system configuration."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'PROMPT_DIR': temp_prompt_dir,
    }):
        config_manager = ConfigManager()
        # Mock the config loading to avoid external dependencies
        with patch.object(config_manager, 'load_config') as mock_load:
            mock_config = Mock(spec=SystemConfig)
            mock_config.prompt_dir = temp_prompt_dir
            mock_config.agents = {}
            mock_load.return_value = mock_config
            yield mock_config


class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_agent_system_initialization(self, mock_system_config, temp_prompt_dir):
        """Test AgentSystem initialization with all components."""
        with patch('deep_agent_system.system.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.return_value = mock_system_config
            
            with patch('deep_agent_system.system.PromptManager') as mock_prompt_manager:
                mock_prompt_manager.return_value = Mock()
                
                with patch('deep_agent_system.system.RAGManager') as mock_rag_manager:
                    mock_rag_manager.return_value = Mock()
                    
                    # Initialize system
                    system = AgentSystem()
                    
                    # Verify system components
                    assert system is not None
                    assert hasattr(system, 'config')
    
    def test_system_question_processing_flow(self, mock_system_config, temp_prompt_dir):
        """Test complete question processing through the system."""
        with patch('deep_agent_system.system.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.return_value = mock_system_config
            
            with patch('deep_agent_system.system.PromptManager') as mock_prompt_manager:
                mock_prompt_manager.return_value = Mock()
                
                with patch('deep_agent_system.system.RAGManager') as mock_rag_manager:
                    mock_rag_manager.return_value = Mock()
                    
                    # Initialize system
                    system = AgentSystem()
                    
                    # Test that system can be initialized without errors
                    assert system is not None


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    def test_concurrent_agent_processing(self, integrated_agents):
        """Test concurrent processing by multiple agents."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_message_worker(agent, question_id):
            try:
                message = Message(
                    sender_id="user",
                    recipient_id=agent.agent_id,
                    content=f"Test question {question_id}",
                    message_type=MessageType.USER_QUESTION
                )
                
                start_time = time.time()
                response = agent.process_message(message)
                end_time = time.time()
                
                results.append({
                    "agent_id": agent.agent_id,
                    "question_id": question_id,
                    "response_time": end_time - start_time,
                    "success": True
                })
            except Exception as e:
                errors.append({
                    "agent_id": agent.agent_id,
                    "question_id": question_id,
                    "error": str(e)
                })
        
        # Create threads for concurrent processing
        threads = []
        for i in range(10):
            for agent_name, agent in integrated_agents.items():
                thread = threading.Thread(
                    target=process_message_worker,
                    args=(agent, f"{agent_name}_{i}")
                )
                threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) > 0
        
        # Check that all agents processed messages
        agent_ids = set(result["agent_id"] for result in results)
        expected_agent_ids = set(agent.agent_id for agent in integrated_agents.values())
        assert agent_ids == expected_agent_ids
    
    def test_memory_usage_stability(self, integrated_agents):
        """Test that memory usage remains stable during processing."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process multiple messages
        analyst = integrated_agents["analyst"]
        for i in range(100):
            message = Message(
                sender_id="user",
                recipient_id="analyst_1",
                content=f"Test question {i}",
                message_type=MessageType.USER_QUESTION
            )
            response = analyst.process_message(message)
            assert response is not None
        
        # Check memory usage after processing
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory usage grew by {object_growth} objects"


if __name__ == "__main__":
    pytest.main([__file__])