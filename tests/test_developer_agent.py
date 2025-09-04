"""Tests for DeveloperAgent."""

import pytest
from unittest.mock import Mock, patch

from deep_agent_system.agents.developer import DeveloperAgent, CodeAnalysis, CodeImplementation
from deep_agent_system.config.models import SystemConfig
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager


@pytest.fixture
def mock_prompt_manager():
    """Create mock prompt manager."""
    mock_pm = Mock(spec=PromptManager)
    mock_pm.get_prompt.return_value = "Test prompt template: {user_question}"
    return mock_pm


@pytest.fixture
def mock_rag_manager():
    """Create mock RAG manager."""
    mock_rm = Mock(spec=RAGManager)
    return mock_rm


@pytest.fixture
def agent_config():
    """Create agent configuration for testing."""
    return AgentConfig(
        agent_id="developer_test",
        agent_type=AgentType.DEVELOPER,
        capabilities=[Capability.CODE_GENERATION, Capability.IMPLEMENTATION],
        rag_enabled=True,
        graph_rag_enabled=False
    )


@pytest.fixture
def developer_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create DeveloperAgent instance for testing."""
    return DeveloperAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager
    )


class TestDeveloperAgent:
    """Test cases for DeveloperAgent."""
    
    def test_initialization(self, developer_agent):
        """Test DeveloperAgent initialization."""
        assert developer_agent.agent_id == "developer_test"
        assert developer_agent.agent_type == AgentType.DEVELOPER.value
        assert developer_agent.has_capability(Capability.CODE_GENERATION)
        assert developer_agent.has_capability(Capability.IMPLEMENTATION)
    
    def test_detect_programming_language_python(self, developer_agent):
        """Test Python language detection."""
        code = "def hello_world():\n    print('Hello, World!')"
        language = developer_agent._detect_programming_language(code, None)
        assert language == "python"
    
    def test_detect_programming_language_javascript(self, developer_agent):
        """Test JavaScript language detection."""
        code = "function helloWorld() {\n    console.log('Hello, World!');\n}"
        language = developer_agent._detect_programming_language(code, None)
        assert language == "javascript"
    
    def test_detect_programming_language_java(self, developer_agent):
        """Test Java language detection."""
        code = "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
        language = developer_agent._detect_programming_language(code, None)
        assert language == "java"
    
    def test_assess_complexity_level_simple(self, developer_agent):
        """Test simple complexity assessment."""
        question = "Create a simple hello world function"
        complexity = developer_agent._assess_complexity_level(question)
        assert complexity == "simple"
    
    def test_assess_complexity_level_complex(self, developer_agent):
        """Test complex complexity assessment."""
        question = "Create a distributed microservice architecture with scalable components"
        complexity = developer_agent._assess_complexity_level(question)
        assert complexity == "complex"
    
    def test_extract_key_requirements(self, developer_agent):
        """Test key requirements extraction."""
        question = "Create a REST API with CRUD operations and authentication"
        requirements = developer_agent._extract_key_requirements(question)
        
        assert "RESTful API endpoints" in requirements
        assert "CRUD operations implementation" in requirements
        assert "Authentication and authorization" in requirements
    
    def test_analyze_development_requirements(self, developer_agent):
        """Test development requirements analysis."""
        question = "Create a Python FastAPI application with database integration"
        
        analysis = developer_agent._analyze_development_requirements(question, None)
        
        assert analysis is not None
        assert analysis.programming_language == "python"
        assert "fastapi" in analysis.framework_requirements
        assert "Database integration and operations" in analysis.key_requirements
    
    def test_generate_code_implementation_python(self, developer_agent):
        """Test Python code generation."""
        question = "Create a FastAPI application"
        analysis = CodeAnalysis(
            problem_understanding="FastAPI application development",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_requirements=["RESTful API endpoints"]
        )
        
        implementation = developer_agent._generate_code_implementation(question, analysis, None)
        
        assert implementation is not None
        assert len(implementation.code_blocks) > 0
        assert "fastapi" in implementation.dependencies
        assert any("main.py" in block["filename"] for block in implementation.code_blocks)
    
    def test_process_development_request(self, developer_agent, mock_rag_manager):
        """Test processing development request message."""
        message = Message(
            sender_id="user",
            recipient_id="developer_test",
            content="Create a Python function to calculate fibonacci numbers",
            message_type=MessageType.USER_QUESTION
        )
        
        # Mock RAG manager response
        mock_rag_manager.retrieve_vector_context.return_value = []
        
        response = developer_agent._handle_development_request(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "developer_test"
        assert response.confidence_score > 0
        assert "Implementation" in response.content
    
    def test_identify_dependencies_python(self, developer_agent):
        """Test Python dependency identification."""
        analysis = CodeAnalysis(
            problem_understanding="FastAPI development",
            programming_language="python",
            framework_requirements=["fastapi", "pytest"],
            complexity_level="medium",
            key_requirements=[]
        )
        
        dependencies = developer_agent._identify_dependencies(analysis)
        
        assert "fastapi" in dependencies
        assert "uvicorn" in dependencies
        assert "pytest" in dependencies
    
    def test_identify_dependencies_javascript(self, developer_agent):
        """Test JavaScript dependency identification."""
        analysis = CodeAnalysis(
            problem_understanding="Express API development",
            programming_language="javascript",
            framework_requirements=["express", "jest"],
            complexity_level="medium",
            key_requirements=[]
        )
        
        dependencies = developer_agent._identify_dependencies(analysis)
        
        assert "express" in dependencies
        assert "cors" in dependencies
        assert "jest" in dependencies
    
    def test_generate_setup_instructions_python(self, developer_agent):
        """Test Python setup instructions generation."""
        analysis = CodeAnalysis(
            problem_understanding="Python development",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_requirements=[]
        )
        
        instructions = developer_agent._generate_setup_instructions(analysis)
        
        assert any("virtual environment" in instruction for instruction in instructions)
        assert any("pip install" in instruction for instruction in instructions)
        assert any("uvicorn" in instruction for instruction in instructions)
    
    def test_public_generate_code_method(self, developer_agent):
        """Test public generate_code method."""
        requirements = "Create a simple calculator function"
        
        implementation = developer_agent.generate_code(requirements, "python")
        
        assert implementation is not None
        assert isinstance(implementation, CodeImplementation)
        assert len(implementation.code_blocks) > 0
    
    def test_public_analyze_requirements_method(self, developer_agent):
        """Test public analyze_requirements method."""
        requirements = "Create a REST API with Python and FastAPI"
        
        analysis = developer_agent.analyze_requirements(requirements)
        
        assert analysis is not None
        assert isinstance(analysis, CodeAnalysis)
        assert analysis.programming_language == "python"
        assert "fastapi" in analysis.framework_requirements


class TestCodeAnalysis:
    """Test cases for CodeAnalysis class."""
    
    def test_code_analysis_creation(self):
        """Test CodeAnalysis object creation."""
        analysis = CodeAnalysis(
            problem_understanding="Test problem",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_requirements=["API endpoints"]
        )
        
        assert analysis.problem_understanding == "Test problem"
        assert analysis.programming_language == "python"
        assert analysis.framework_requirements == ["fastapi"]
        assert analysis.complexity_level == "medium"
        assert analysis.key_requirements == ["API endpoints"]
    
    def test_code_analysis_to_dict(self):
        """Test CodeAnalysis to_dict method."""
        analysis = CodeAnalysis(
            problem_understanding="Test problem",
            programming_language="python",
            framework_requirements=["fastapi"],
            complexity_level="medium",
            key_requirements=["API endpoints"]
        )
        
        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["programming_language"] == "python"
        assert result["framework_requirements"] == ["fastapi"]


class TestCodeImplementation:
    """Test cases for CodeImplementation class."""
    
    def test_code_implementation_creation(self):
        """Test CodeImplementation object creation."""
        implementation = CodeImplementation(
            strategy="Test strategy",
            code_blocks=[{"filename": "test.py", "code": "print('test')"}],
            dependencies=["pytest"],
            setup_instructions=["pip install pytest"],
            usage_examples=["python test.py"],
            test_cases=[{"name": "test_basic", "code": "def test_basic(): pass"}]
        )
        
        assert implementation.strategy == "Test strategy"
        assert len(implementation.code_blocks) == 1
        assert implementation.dependencies == ["pytest"]
    
    def test_code_implementation_to_dict(self):
        """Test CodeImplementation to_dict method."""
        implementation = CodeImplementation(
            strategy="Test strategy",
            code_blocks=[{"filename": "test.py", "code": "print('test')"}],
            dependencies=["pytest"],
            setup_instructions=["pip install pytest"],
            usage_examples=["python test.py"],
            test_cases=[{"name": "test_basic", "code": "def test_basic(): pass"}]
        )
        
        result = implementation.to_dict()
        
        assert isinstance(result, dict)
        assert result["strategy"] == "Test strategy"
        assert len(result["code_blocks"]) == 1
        assert result["dependencies"] == ["pytest"]