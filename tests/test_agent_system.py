"""Tests for the AgentSystem main orchestrator."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent_system.config.models import AgentType
from deep_agent_system.models.messages import MessageType
from deep_agent_system.models.messages import Message, Response
from deep_agent_system.system import AgentSystem, AgentSystemError


class TestAgentSystem:
    """Test cases for AgentSystem orchestrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        env_vars = {
            "OPENAI_API_KEY": "test-api-key",
            "NEO4J_PASSWORD": "test-password",
            "PROMPT_DIR": "./prompts",
        }
        
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    @pytest.fixture
    def agent_system(self, temp_dir, mock_env_vars):
        """Create AgentSystem instance for testing."""
        # Create mock .env file
        env_file = temp_dir / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test-api-key
NEO4J_PASSWORD=test-password
PROMPT_DIR=./prompts
""")
        
        return AgentSystem(config_file=str(env_file))
    
    @pytest.mark.asyncio
    async def test_agent_system_initialization(self, agent_system, temp_dir):
        """Test AgentSystem initialization."""
        # Create mock prompt directory
        prompt_dir = temp_dir / "prompts"
        prompt_dir.mkdir()
        
        # Create mock prompt files
        for agent_type in ["analyst", "architect", "developer", "code_reviewer", "tester"]:
            agent_dir = prompt_dir / agent_type
            agent_dir.mkdir()
            (agent_dir / "default.txt").write_text(f"Mock prompt for {agent_type}")
        
        # Mock the components that require external dependencies
        with patch('deep_agent_system.system.PromptManager') as mock_prompt_manager, \
             patch('deep_agent_system.system.RAGManager') as mock_rag_manager, \
             patch('deep_agent_system.system.AnalystAgent') as mock_analyst, \
             patch('deep_agent_system.system.ArchitectAgent') as mock_architect, \
             patch('deep_agent_system.system.DeveloperAgent') as mock_developer, \
             patch('deep_agent_system.system.CodeReviewerAgent') as mock_reviewer, \
             patch('deep_agent_system.system.TesterAgent') as mock_tester:
            
            # Setup mocks
            mock_prompt_manager.return_value.load_prompts.return_value = {"test": "prompt"}
            mock_rag_manager.return_value.initialize = AsyncMock()
            
            # Mock agent instances
            for mock_agent_class in [mock_analyst, mock_architect, mock_developer, mock_reviewer, mock_tester]:
                mock_agent = MagicMock()
                mock_agent.agent_id = "test_agent"
                mock_agent.agent_type = "test_type"
                mock_agent_class.return_value = mock_agent
            
            # Initialize system
            await agent_system.initialize()
            
            # Verify initialization
            assert agent_system.is_initialized
            assert agent_system.config is not None
            assert agent_system.config_manager is not None
            assert agent_system.prompt_manager is not None
            assert agent_system.communication_manager is not None
            assert agent_system.workflow_executor is not None
    
    @pytest.mark.asyncio
    async def test_agent_system_start_stop(self, agent_system):
        """Test AgentSystem start and stop functionality."""
        # Mock initialization
        agent_system.is_initialized = True
        agent_system.rag_manager = MagicMock()
        agent_system.rag_manager.start = AsyncMock()
        agent_system.rag_manager.stop = AsyncMock()
        agent_system.workflow_executor = MagicMock()
        agent_system.communication_manager = MagicMock()
        agent_system.agents = {"test_agent": MagicMock()}
        
        # Test start
        await agent_system.start()
        assert agent_system.is_running
        agent_system.rag_manager.start.assert_called_once()
        
        # Test stop
        await agent_system.stop()
        assert not agent_system.is_running
        agent_system.rag_manager.stop.assert_called_once()
        agent_system.workflow_executor.shutdown.assert_called_once()
        agent_system.communication_manager.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_system_start_without_initialization(self, agent_system):
        """Test that starting without initialization raises error."""
        with pytest.raises(AgentSystemError, match="System not initialized"):
            await agent_system.start()
    
    @pytest.mark.asyncio
    async def test_process_message(self, agent_system):
        """Test message processing through the system."""
        # Setup system as running
        agent_system.is_running = True
        
        # Mock analyst agent
        mock_analyst = MagicMock()
        mock_response = Response(
            message_id="test_msg",
            agent_id="analyst_agent",
            content="Test response",
            confidence_score=0.9
        )
        mock_analyst.process_message.return_value = mock_response
        mock_analyst.config = MagicMock()
        mock_analyst.config.agent_type = AgentType.ANALYST
        
        agent_system.agents = {"analyst_agent": mock_analyst}
        
        # Process message
        response = await agent_system.process_message("Test message")
        
        # Verify response
        assert response == mock_response
        mock_analyst.process_message.assert_called_once()
        
        # Verify message structure
        call_args = mock_analyst.process_message.call_args[0][0]
        assert isinstance(call_args, Message)
        assert call_args.content == "Test message"
        assert call_args.sender_id == "user"
        assert call_args.message_type == MessageType.USER_QUESTION
    
    @pytest.mark.asyncio
    async def test_process_message_without_analyst(self, agent_system):
        """Test message processing when no analyst agent is available."""
        agent_system.is_running = True
        agent_system.agents = {}
        
        with pytest.raises(AgentSystemError, match="Analyst agent not available"):
            await agent_system.process_message("Test message")
    
    @pytest.mark.asyncio
    async def test_process_message_not_running(self, agent_system):
        """Test message processing when system is not running."""
        with pytest.raises(AgentSystemError, match="System not running"):
            await agent_system.process_message("Test message")
    
    def test_get_agent(self, agent_system):
        """Test getting agent by ID."""
        mock_agent = MagicMock()
        agent_system.agents = {"test_agent": mock_agent}
        
        # Test existing agent
        result = agent_system.get_agent("test_agent")
        assert result == mock_agent
        
        # Test non-existing agent
        result = agent_system.get_agent("nonexistent")
        assert result is None
    
    def test_get_agents_by_type(self, agent_system):
        """Test getting agents by type."""
        # Create mock agents
        mock_analyst = MagicMock()
        mock_analyst.config = MagicMock()
        mock_analyst.config.agent_type = AgentType.ANALYST
        
        mock_architect = MagicMock()
        mock_architect.config = MagicMock()
        mock_architect.config.agent_type = AgentType.ARCHITECT
        
        mock_analyst2 = MagicMock()
        mock_analyst2.config = MagicMock()
        mock_analyst2.config.agent_type = AgentType.ANALYST
        
        agent_system.agents = {
            "analyst1": mock_analyst,
            "architect1": mock_architect,
            "analyst2": mock_analyst2,
        }
        
        # Test getting analysts
        analysts = agent_system.get_agents_by_type(AgentType.ANALYST)
        assert len(analysts) == 2
        assert mock_analyst in analysts
        assert mock_analyst2 in analysts
        
        # Test getting architects
        architects = agent_system.get_agents_by_type(AgentType.ARCHITECT)
        assert len(architects) == 1
        assert mock_architect in architects
        
        # Test getting non-existing type
        developers = agent_system.get_agents_by_type(AgentType.DEVELOPER)
        assert len(developers) == 0
    
    def test_list_agents(self, agent_system):
        """Test listing all agents."""
        # Create mock agents
        mock_agent1 = MagicMock()
        mock_agent1.agent_type = "analyst"
        mock_agent1.state = {"is_active": True}
        
        mock_agent2 = MagicMock()
        mock_agent2.agent_type = "architect"
        mock_agent2.state = {"is_active": False}
        
        agent_system.agents = {
            "agent1": mock_agent1,
            "agent2": mock_agent2,
        }
        
        # List agents
        agent_list = agent_system.list_agents()
        
        assert len(agent_list) == 2
        
        # Check first agent
        agent1_info = next(info for info in agent_list if info["agent_id"] == "agent1")
        assert agent1_info["agent_type"] == "analyst"
        assert agent1_info["is_active"] is True
        
        # Check second agent
        agent2_info = next(info for info in agent_list if info["agent_id"] == "agent2")
        assert agent2_info["agent_type"] == "architect"
        assert agent2_info["is_active"] is False
    
    def test_get_system_status(self, agent_system):
        """Test getting system status."""
        # Setup system state
        agent_system.is_initialized = True
        agent_system.is_running = True
        agent_system.config_manager = MagicMock()
        agent_system.prompt_manager = MagicMock()
        agent_system.rag_manager = MagicMock()
        agent_system.communication_manager = MagicMock()
        agent_system.communication_manager.get_delivery_statistics.return_value = {
            "messages_sent": 10,
            "messages_delivered": 9,
        }
        agent_system.workflow_executor = MagicMock()
        agent_system.workflow_executor.get_statistics.return_value = {
            "executions_completed": 5,
        }
        agent_system.agents = {"test_agent": MagicMock()}
        
        # Get status
        status = agent_system.get_system_status()
        
        # Verify status structure
        assert status["is_initialized"] is True
        assert status["is_running"] is True
        assert status["agents_count"] == 1
        assert "components" in status
        assert "agents" in status
        assert "communication_stats" in status
        assert "workflow_stats" in status
        
        # Verify components status
        components = status["components"]
        assert components["config_manager"] is True
        assert components["prompt_manager"] is True
        assert components["rag_manager"] is True
        assert components["communication_manager"] is True
        assert components["workflow_executor"] is True
    
    @pytest.mark.asyncio
    async def test_reload_prompts(self, agent_system):
        """Test reloading prompts."""
        # Test with prompt manager
        agent_system.prompt_manager = MagicMock()
        result = await agent_system.reload_prompts()
        assert result is True
        agent_system.prompt_manager.reload_prompts.assert_called_once()
        
        # Test without prompt manager
        agent_system.prompt_manager = None
        result = await agent_system.reload_prompts()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_reload_configuration(self, agent_system):
        """Test reloading configuration."""
        # Test with config manager
        mock_config = MagicMock()
        agent_system.config_manager = MagicMock()
        agent_system.config_manager.load_config.return_value = mock_config
        
        result = await agent_system.reload_configuration()
        assert result is True
        assert agent_system.config == mock_config
        agent_system.config_manager.load_config.assert_called_once()
        agent_system.config_manager.validate_config.assert_called_once_with(mock_config)
        
        # Test without config manager
        agent_system.config_manager = None
        result = await agent_system.reload_configuration()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, agent_system):
        """Test handling of initialization failures."""
        with patch('deep_agent_system.system.ConfigManager') as mock_config_manager:
            mock_config_manager.side_effect = Exception("Config error")
            
            with pytest.raises(AgentSystemError, match="Initialization failed"):
                await agent_system.initialize()
            
            assert not agent_system.is_initialized
    
    @pytest.mark.asyncio
    async def test_start_failure(self, agent_system):
        """Test handling of start failures."""
        agent_system.is_initialized = True
        agent_system.rag_manager = MagicMock()
        agent_system.rag_manager.start = AsyncMock(side_effect=Exception("Start error"))
        
        with pytest.raises(AgentSystemError, match="Start failed"):
            await agent_system.start()
        
        assert not agent_system.is_running
    
    def test_repr(self, agent_system):
        """Test string representation of AgentSystem."""
        agent_system.is_initialized = True
        agent_system.is_running = False
        agent_system.agents = {"agent1": MagicMock(), "agent2": MagicMock()}
        
        repr_str = repr(agent_system)
        assert "AgentSystem" in repr_str
        assert "initialized=True" in repr_str
        assert "running=False" in repr_str
        assert "agents=2" in repr_str


class TestAgentSystemIntegration:
    """Integration tests for AgentSystem."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_system_lifecycle(self, temp_dir):
        """Test complete system lifecycle with mocked dependencies."""
        # Create test environment
        env_file = temp_dir / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test-api-key
NEO4J_PASSWORD=test-password
PROMPT_DIR=./prompts
VECTOR_STORE_ENABLED=false
GRAPH_STORE_ENABLED=false
""")
        
        # Create prompt directory structure
        prompt_dir = temp_dir / "prompts"
        prompt_dir.mkdir()
        for agent_type in ["analyst", "architect", "developer", "code_reviewer", "tester"]:
            agent_dir = prompt_dir / agent_type
            agent_dir.mkdir()
            (agent_dir / "default.txt").write_text(f"Mock prompt for {agent_type}")
        
        # Mock external dependencies
        with patch('deep_agent_system.system.PromptManager') as mock_prompt_manager, \
             patch('deep_agent_system.system.AnalystAgent') as mock_analyst, \
             patch('deep_agent_system.system.ArchitectAgent') as mock_architect, \
             patch('deep_agent_system.system.DeveloperAgent') as mock_developer, \
             patch('deep_agent_system.system.CodeReviewerAgent') as mock_reviewer, \
             patch('deep_agent_system.system.TesterAgent') as mock_tester:
            
            # Setup mocks
            mock_prompt_manager.return_value.load_prompts.return_value = {"test": "prompt"}
            
            # Mock agent instances
            mock_agents = {}
            for agent_type, mock_agent_class in [
                (AgentType.ANALYST, mock_analyst),
                (AgentType.ARCHITECT, mock_architect),
                (AgentType.DEVELOPER, mock_developer),
                (AgentType.CODE_REVIEWER, mock_reviewer),
                (AgentType.TESTER, mock_tester),
            ]:
                mock_agent = MagicMock()
                mock_agent.agent_id = f"{agent_type.value}_agent"
                mock_agent.agent_type = agent_type.value
                mock_agent.config = MagicMock()
                mock_agent.config.agent_type = agent_type
                mock_agent.process_message.return_value = Response(
                    message_id="test",
                    agent_id=mock_agent.agent_id,
                    content=f"Response from {agent_type.value}",
                    confidence_score=0.9
                )
                mock_agents[agent_type] = mock_agent
                mock_agent_class.return_value = mock_agent
            
            # Create and test system
            system = AgentSystem(config_file=str(env_file))
            
            # Test initialization
            await system.initialize()
            assert system.is_initialized
            assert len(system.agents) == 5
            
            # Test start
            await system.start()
            assert system.is_running
            
            # Test message processing
            response = await system.process_message("Test message")
            assert response is not None
            assert "analyst" in response.content.lower()
            
            # Test system status
            status = system.get_system_status()
            assert status["is_initialized"]
            assert status["is_running"]
            assert status["agents_count"] == 5
            
            # Test stop
            await system.stop()
            assert not system.is_running
    
    @pytest.mark.asyncio
    async def test_agent_communication_flow(self, temp_dir):
        """Test communication flow between agents."""
        # Setup similar to previous test but focus on communication
        env_file = temp_dir / ".env"
        env_file.write_text("""
OPENAI_API_KEY=test-api-key
NEO4J_PASSWORD=test-password
PROMPT_DIR=./prompts
VECTOR_STORE_ENABLED=false
GRAPH_STORE_ENABLED=false
""")
        
        prompt_dir = temp_dir / "prompts"
        prompt_dir.mkdir()
        for agent_type in ["analyst", "architect"]:
            agent_dir = prompt_dir / agent_type
            agent_dir.mkdir()
            (agent_dir / "default.txt").write_text(f"Mock prompt for {agent_type}")
        
        with patch('deep_agent_system.system.PromptManager') as mock_prompt_manager, \
             patch('deep_agent_system.system.AnalystAgent') as mock_analyst, \
             patch('deep_agent_system.system.ArchitectAgent') as mock_architect:
            
            mock_prompt_manager.return_value.load_prompts.return_value = {"test": "prompt"}
            
            # Create mock agents that can communicate
            analyst_agent = MagicMock()
            analyst_agent.agent_id = "analyst_agent"
            analyst_agent.config = MagicMock()
            analyst_agent.config.agent_type = AgentType.ANALYST
            
            architect_agent = MagicMock()
            architect_agent.agent_id = "architect_agent"
            architect_agent.config = MagicMock()
            architect_agent.config.agent_type = AgentType.ARCHITECT
            
            mock_analyst.return_value = analyst_agent
            mock_architect.return_value = architect_agent
            
            # Setup system
            system = AgentSystem(config_file=str(env_file))
            await system.initialize()
            await system.start()
            
            # Verify agents are registered with communication manager
            assert system.communication_manager is not None
            registered_agents = system.communication_manager.get_registered_agents()
            assert "analyst_agent" in registered_agents
            assert "architect_agent" in registered_agents
            
            await system.stop()