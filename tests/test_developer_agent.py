"""Tests for DeveloperAgent."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from deep_agent_system.agents.developer import DeveloperAgent, CodeAnalysis, CodeImplementation
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
from deep_agent_system.models.messages import Message, MessageType, Response, Context, Document
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = Mock(spec=PromptManager)
    manager.get_prompt.return_value = "Test prompt template: {user_question}"
    return manager


@pytest.fixture
def mock_rag_manager():
    """Create a mock RAG manager."""
    manager = Mock(spec=RAGManager)
    
    # Mock documents for context
    mock_docs = [
        Document(
            content="Example Python code for API development",
            metadata={"language": "python", "type": "api"}
        ),
        Document(
            content="FastAPI implementation example",
            metadata={"language": "python", "framework": "fastapi"}
        )
    ]
    
    mock_context = Context(
        query="test query",
        documents=mock_docs,
        relevance_scores=[0.9, 0.8]
    )
    
    manager.retrieve_vector_context.return_value = mock_docs
    manager.hybrid_retrieve.return_value = mock_context
    
    return manager


@pytest.fixture
def agent_config():
    """Create agent configuration for testing."""
    return AgentConfig(
        agent_id="test_developer",
        agent_type=AgentType.DEVELOPER,
        llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
        capabilities=[Capability.CODE_GENERATION],
        rag_enabled=True,
        graph_rag_enabled=False
    )


@pytest.fixture
def developer_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create a DeveloperAgent instance for testing."""
    return DeveloperAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager
    )


class TestDeveloperAgent:
    """Test cases for DeveloperAgent."""
    
    def test_initialization(self, developer_agent):
        """Test agent initialization."""
        assert developer_agent.agent_id == "test_developer"
        assert developer_agent.agent_type == "developer"
        assert developer_agent.has_capability(Capability.CODE_GENERATION)
    
    def test_process_user_question(self, developer_agent):
        """Test processing a user question."""
        message = Message(
            sender_id="user",
            recipient_id="test_developer",
            content="Create a Python API endpoint for user management",
            message_type=MessageType.USER_QUESTION
        )
        
        response = developer_agent.process_message(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "test_developer"
        assert response.message_id == message.id
        assert "Implementation Strategy" in response.content
        assert "Code Implementation" in response.content
        assert response.confidence_score > 0
    
    def test_process_agent_request(self, developer_agent):
        """Test processing an agent request."""
        message = Message(
            sender_id="analyst",
            recipient_id="test_developer",
            content="Implement a REST API with authentication",
            message_type=MessageType.AGENT_REQUEST
        )
        
        response = developer_agent.process_message(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "test_developer"
        assert "python" in response.content.lower() or "javascript" in response.content.lower()
    
    def test_unsupported_message_type(self, developer_agent):
        """Test handling unsupported message types."""
        message = Message(
            sender_id="user",
            recipient_id="test_developer",
            content="Test message",
            message_type=MessageType.SYSTEM_MESSAGE
        )
        
        response = developer_agent.process_message(message)
        
        assert "Unsupported message type" in response.content
        assert response.confidence_score == 0.0
    
    def test_detect_programming_language_python(self, developer_agent):
        """Test Python language detection."""
        question = "Create a FastAPI endpoint with Pydantic models"
        context = None
        
        language = developer_agent._detect_programming_language(question, context)
        
        assert language == "python"
    
    def test_detect_programming_language_javascript(self, developer_agent):
        """Test JavaScript language detection."""
        question = "Build a Node.js Express API with React frontend"
        context = None
        
        language = developer_agent._detect_programming_language(question, context)
        
        assert language == "javascript"
    
    def test_detect_programming_language_default(self, developer_agent):
        """Test default language detection."""
        question = "Create a simple web application"
        context = None
        
        language = developer_agent._detect_programming_language(question, context)
        
        assert language == "python"  # Default
    
    def test_assess_complexity_high(self, developer_agent):
        """Test high complexity assessment."""
        question = "Build a distributed microservices architecture with real-time streaming"
        
        complexity = developer_agent._assess_complexity(question)
        
        assert complexity == "high"
    
    def test_assess_complexity_medium(self, developer_agent):
        """Test medium complexity assessment."""
        question = "Create an API with database integration and authentication"
        
        complexity = developer_agent._assess_complexity(question)
        
        assert complexity == "medium"
    
    def test_assess_complexity_low(self, developer_agent):
        """Test low complexity assessment."""
        question = "Write a simple function to calculate fibonacci numbers"
        
        complexity = developer_agent._assess_complexity(question)
        
        assert complexity == "low"
    
    def test_identify_frameworks_python(self, developer_agent):
        """Test framework identification for Python."""
        question = "Create a FastAPI application with pytest tests"
        language = "python"
        
        frameworks = developer_agent._identify_frameworks(question, language)
        
        assert "fastapi" in frameworks
        assert "pytest" in frameworks
    
    def test_identify_frameworks_javascript(self, developer_agent):
        """Test framework identification for JavaScript."""
        question = "Build an Express server with React frontend"
        language = "javascript"
        
        frameworks = developer_agent._identify_frameworks(question, language)
        
        assert "express" in frameworks
        assert "react" in frameworks
    
    def test_identify_key_components_api(self, developer_agent):
        """Test key component identification for API."""
        question = "Create a REST API with authentication"
        language = "python"
        
        components = developer_agent._identify_key_components(question, language)
        
        assert "API Router" in components
        assert "Authentication" in components
        assert "Error Handling" in components
    
    def test_identify_key_components_database(self, developer_agent):
        """Test key component identification for database."""
        question = "Implement database models and CRUD operations"
        language = "python"
        
        components = developer_agent._identify_key_components(question, language)
        
        assert "Data Models" in components
        assert "Database Connection" in components
        assert "CRUD Operations" in components
    
    def test_analyze_code_requirements(self, developer_agent):
        """Test code requirements analysis."""
        question = "Create a Python FastAPI application with database integration"
        context = None
        
        analysis = developer_agent._analyze_code_requirements(question, context)
        
        assert isinstance(analysis, CodeAnalysis)
        assert analysis.programming_language == "python"
        assert "fastapi" in analysis.framework_requirements
        assert analysis.complexity_level in ["low", "medium", "high"]
        assert len(analysis.key_components) > 0
    
    def test_generate_implementation_strategy(self, developer_agent):
        """Test implementation strategy generation."""
        question = "Create an API endpoint"
        analysis = CodeAnalysis(
            problem_understanding="API development request",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_components=["API Router", "Error Handling"]
        )
        
        strategy = developer_agent._generate_implementation_strategy(question, analysis)
        
        assert "Implementation approach for python development" in strategy
        assert "fastapi" in strategy
        assert "API Router" in strategy
        assert "Best Practices" in strategy
    
    def test_get_api_example_python(self, developer_agent):
        """Test Python API example generation."""
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_components=["API Router"]
        )
        
        example = developer_agent._get_api_example("python", analysis)
        
        assert example["filename"] == "api_example.py"
        assert "fastapi" in example["code"].lower()
        assert "FastAPI" in example["code"]
        assert "@app.post" in example["code"]
    
    def test_get_api_example_javascript(self, developer_agent):
        """Test JavaScript API example generation."""
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="javascript",
            framework_requirements=["express"],
            complexity_level="medium",
            key_components=["API Router"]
        )
        
        example = developer_agent._get_api_example("javascript", analysis)
        
        assert example["filename"] == "api_example.js"
        assert "express" in example["code"].lower()
        assert "app.post" in example["code"]
        assert "app.get" in example["code"]
    
    def test_get_model_example_python(self, developer_agent):
        """Test Python model example generation."""
        analysis = CodeAnalysis(
            problem_understanding="Data model creation",
            programming_language="python",
            framework_requirements=["pydantic"],
            complexity_level="low",
            key_components=["Data Models"]
        )
        
        example = developer_agent._get_model_example("python", analysis)
        
        assert example["filename"] == "models.py"
        assert "pydantic" in example["code"].lower()
        assert "BaseModel" in example["code"]
        assert "class User" in example["code"]
    
    def test_identify_dependencies_python(self, developer_agent):
        """Test dependency identification for Python."""
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="python",
            framework_requirements=["fastapi", "pytest"],
            complexity_level="medium",
            key_components=[]
        )
        
        dependencies = developer_agent._identify_dependencies(analysis)
        
        assert "python>=3.8" in dependencies
        assert "fastapi" in dependencies
        assert "pytest" in dependencies
    
    def test_generate_setup_instructions_python(self, developer_agent):
        """Test setup instructions generation for Python."""
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_components=[]
        )
        
        instructions = developer_agent._generate_setup_instructions(analysis)
        
        assert any("virtual environment" in instruction.lower() for instruction in instructions)
        assert any("pip install" in instruction.lower() for instruction in instructions)
    
    def test_calculate_confidence_with_context(self, developer_agent):
        """Test confidence calculation with context."""
        question = "Create a Python API"
        context = Context(
            query="test",
            documents=[Document(content="test", metadata={})],
            relevance_scores=[0.9]
        )
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="python",
            framework_requirements=[],
            complexity_level="low",
            key_components=[]
        )
        
        confidence = developer_agent._calculate_confidence(question, context, analysis)
        
        assert 0.7 <= confidence <= 1.0
    
    def test_calculate_confidence_high_complexity(self, developer_agent):
        """Test confidence calculation with high complexity."""
        question = "Create a distributed system"
        context = None
        analysis = CodeAnalysis(
            problem_understanding="Distributed system",
            programming_language="python",
            framework_requirements=[],
            complexity_level="high",
            key_components=[]
        )
        
        confidence = developer_agent._calculate_confidence(question, context, analysis)
        
        assert confidence < 0.8  # Should be lower due to high complexity
    
    def test_format_code_response(self, developer_agent):
        """Test code response formatting."""
        question = "Create an API"
        analysis = CodeAnalysis(
            problem_understanding="API development",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_components=["API Router"]
        )
        implementation = CodeImplementation(
            implementation_strategy="Test strategy",
            code_examples=[{
                "filename": "test.py",
                "description": "Test API",
                "code": "print('hello')"
            }],
            dependencies=["python", "fastapi"],
            setup_instructions=["Install Python"],
            usage_examples=["python test.py"],
            test_examples=["pytest test.py"]
        )
        
        response = developer_agent._format_code_response(question, analysis, implementation, None)
        
        assert "# Code Implementation Response" in response
        assert "## 1. Implementation Strategy" in response
        assert "## 2. Code Implementation" in response
        assert "## 3. Dependencies & Setup" in response
        assert "## 4. Usage Examples" in response
        assert "## 5. Testing Approach" in response
        assert "## 6. Best Practices" in response
    
    def test_error_handling(self, developer_agent):
        """Test error handling in message processing."""
        # Create a message that will cause an error
        message = Message(
            sender_id="user",
            recipient_id="test_developer",
            content="",  # Empty content should cause validation error
            message_type=MessageType.USER_QUESTION
        )
        
        # Mock the _handle_code_request method to raise an exception
        with patch.object(developer_agent, '_handle_code_request', side_effect=Exception("Test error")):
            response = developer_agent.process_message(message)
        
        assert "error" in response.content.lower()
        assert response.confidence_score == 0.0
        assert response.metadata.get("error") is True


class TestCodeAnalysis:
    """Test cases for CodeAnalysis class."""
    
    def test_code_analysis_creation(self):
        """Test CodeAnalysis creation and serialization."""
        analysis = CodeAnalysis(
            problem_understanding="Test problem",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_components=["API", "Database"]
        )
        
        assert analysis.problem_understanding == "Test problem"
        assert analysis.programming_language == "python"
        assert analysis.framework_requirements == ["fastapi"]
        assert analysis.complexity_level == "medium"
        assert analysis.key_components == ["API", "Database"]
        
        # Test serialization
        data = analysis.to_dict()
        assert data["problem_understanding"] == "Test problem"
        assert data["programming_language"] == "python"


class TestCodeImplementation:
    """Test cases for CodeImplementation class."""
    
    def test_code_implementation_creation(self):
        """Test CodeImplementation creation and serialization."""
        implementation = CodeImplementation(
            implementation_strategy="Test strategy",
            code_examples=[{"filename": "test.py", "code": "print('hello')"}],
            dependencies=["python"],
            setup_instructions=["Install Python"],
            usage_examples=["python test.py"],
            test_examples=["pytest test.py"]
        )
        
        assert implementation.implementation_strategy == "Test strategy"
        assert len(implementation.code_examples) == 1
        assert implementation.dependencies == ["python"]
        
        # Test serialization
        data = implementation.to_dict()
        assert data["implementation_strategy"] == "Test strategy"
        assert len(data["code_examples"]) == 1