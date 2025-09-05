"""Tests for TesterAgent."""

import pytest
from unittest.mock import Mock

from deep_agent_system.agents.tester import (
    TesterAgent, TestCase, TestSuite, TestStrategy, TestingResult
)
from deep_agent_system.config.models import SystemConfig
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
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
        agent_id="tester_test",
        agent_type=AgentType.TESTER,
        llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
        capabilities=[Capability.TEST_CREATION],
        rag_enabled=True,
        graph_rag_enabled=False
    )


@pytest.fixture
def tester_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create TesterAgent instance for testing."""
    return TesterAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager
    )


class TestTesterAgent:
    """Test cases for TesterAgent."""
    
    def test_initialization(self, tester_agent):
        """Test TesterAgent initialization."""
        assert tester_agent.agent_id == "tester_test"
        assert tester_agent.agent_type == AgentType.TESTER.value
        assert tester_agent.has_capability(Capability.TEST_CREATION)
        assert tester_agent.has_capability(Capability.QUALITY_ASSURANCE)
    
    def test_detect_language_python(self, tester_agent):
        """Test Python language detection."""
        code = "def hello_world():\n    print('Hello, World!')"
        language = tester_agent._detect_language(code)
        assert language == "python"
    
    def test_detect_language_javascript(self, tester_agent):
        """Test JavaScript language detection."""
        code = "function helloWorld() {\n    console.log('Hello, World!');\n}"
        language = tester_agent._detect_language(code)
        assert language == "javascript"
    
    def test_detect_language_java(self, tester_agent):
        """Test Java language detection."""
        code = "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
        language = tester_agent._detect_language(code)
        assert language == "java"
    
    def test_extract_code_from_message_markdown(self, tester_agent):
        """Test extracting code from markdown code blocks."""
        message = "Here's some code:\n```python\ndef hello():\n    print('hello')\n```"
        code = tester_agent._extract_code_from_message(message)
        assert code == "def hello():\n    print('hello')"
    
    def test_extract_code_from_message_inline(self, tester_agent):
        """Test extracting code from inline code."""
        message = "def hello():\n    print('hello')\n    return True"
        code = tester_agent._extract_code_from_message(message)
        assert "def hello():" in code
        assert "print('hello')" in code
    
    def test_analyze_testing_requirements_unit_tests(self, tester_agent):
        """Test analyzing requirements for unit tests."""
        question = "Create unit tests for my Python function"
        code = "def add(a, b):\n    return a + b"
        
        analysis = tester_agent._analyze_testing_requirements(question, code, None)
        
        assert analysis["language"] == "python"
        assert "unit" in analysis["test_types_needed"]
        assert "pytest" in analysis["frameworks_recommended"]
    
    def test_analyze_testing_requirements_api_testing(self, tester_agent):
        """Test analyzing requirements for API testing."""
        question = "Create integration tests for my REST API"
        code = None
        
        analysis = tester_agent._analyze_testing_requirements(question, code, None)
        
        assert "integration" in analysis["test_types_needed"]
        assert "api_testing" in analysis["testing_focus"]
    
    def test_analyze_testing_requirements_performance_testing(self, tester_agent):
        """Test analyzing requirements for performance testing."""
        question = "Create performance tests and load testing for my application"
        code = None
        
        analysis = tester_agent._analyze_testing_requirements(question, code, None)
        
        assert "performance" in analysis["test_types_needed"]
    
    def test_analyze_testing_requirements_security_testing(self, tester_agent):
        """Test analyzing requirements for security testing."""
        question = "Create security tests to check for vulnerabilities"
        code = None
        
        analysis = tester_agent._analyze_testing_requirements(question, code, None)
        
        assert "security" in analysis["test_types_needed"]
        assert "security_testing" in analysis["testing_focus"]
    
    def test_generate_testing_strategy(self, tester_agent):
        """Test generating testing strategy."""
        analysis = {
            "language": "python",
            "test_types_needed": ["unit", "integration"],
            "complexity_level": "medium",
            "frameworks_recommended": ["pytest"]
        }
        
        strategy = tester_agent._generate_testing_strategy(analysis)
        
        assert isinstance(strategy, TestStrategy)
        assert "unit" in strategy.test_types
        assert "integration" in strategy.test_types
        assert "pytest" in strategy.tools_recommended
    
    def test_generate_python_unit_tests(self, tester_agent):
        """Test generating Python unit tests."""
        code = "def add(a, b):\n    return a + b"
        
        test_cases = tester_agent._generate_python_unit_tests(code)
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "unit" for tc in test_cases)
        assert any("pytest" in tc.code for tc in test_cases)
    
    def test_generate_javascript_unit_tests(self, tester_agent):
        """Test generating JavaScript unit tests."""
        code = "function add(a, b) {\n    return a + b;\n}"
        
        test_cases = tester_agent._generate_javascript_unit_tests(code)
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "unit" for tc in test_cases)
        assert any("jest" in tc.code.lower() for tc in test_cases)
    
    def test_generate_java_unit_tests(self, tester_agent):
        """Test generating Java unit tests."""
        code = "public int add(int a, int b) {\n    return a + b;\n}"
        
        test_cases = tester_agent._generate_java_unit_tests(code)
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "unit" for tc in test_cases)
        assert any("junit" in tc.code.lower() for tc in test_cases)
    
    def test_generate_api_integration_tests_python(self, tester_agent):
        """Test generating Python API integration tests."""
        test_cases = tester_agent._generate_api_integration_tests("python")
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "integration" for tc in test_cases)
        assert any("fastapi" in tc.code.lower() or "client" in tc.code.lower() for tc in test_cases)
    
    def test_generate_api_integration_tests_javascript(self, tester_agent):
        """Test generating JavaScript API integration tests."""
        test_cases = tester_agent._generate_api_integration_tests("javascript")
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "integration" for tc in test_cases)
        assert any("supertest" in tc.code.lower() for tc in test_cases)
    
    def test_generate_database_integration_tests(self, tester_agent):
        """Test generating database integration tests."""
        test_cases = tester_agent._generate_database_integration_tests("python")
        
        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert any(tc.test_type == "integration" for tc in test_cases)
        assert any("database" in tc.description.lower() for tc in test_cases)
    
    def test_generate_automation_config_python(self, tester_agent):
        """Test generating automation config for Python."""
        analysis = {
            "language": "python",
            "test_types_needed": ["unit", "integration"],
            "frameworks_recommended": ["pytest"]
        }
        
        config = tester_agent._generate_automation_config(analysis)
        
        assert "ci_cd_integration" in config
        assert "test_execution" in config
        assert "coverage" in config
        assert "pytest" in str(config)
    
    def test_generate_automation_config_javascript(self, tester_agent):
        """Test generating automation config for JavaScript."""
        analysis = {
            "language": "javascript",
            "test_types_needed": ["unit", "integration"],
            "frameworks_recommended": ["jest"]
        }
        
        config = tester_agent._generate_automation_config(analysis)
        
        assert "ci_cd_integration" in config
        assert "test_execution" in config
        assert "coverage" in config
        assert "npm" in str(config)
    
    def test_generate_execution_instructions_python(self, tester_agent):
        """Test generating execution instructions for Python."""
        analysis = {"language": "python"}
        
        instructions = tester_agent._generate_execution_instructions(analysis)
        
        assert len(instructions) > 0
        assert any("pytest" in instruction for instruction in instructions)
        assert any("pip install" in instruction for instruction in instructions)
    
    def test_generate_execution_instructions_javascript(self, tester_agent):
        """Test generating execution instructions for JavaScript."""
        analysis = {"language": "javascript"}
        
        instructions = tester_agent._generate_execution_instructions(analysis)
        
        assert len(instructions) > 0
        assert any("npm" in instruction for instruction in instructions)
        assert any("jest" in instruction for instruction in instructions)
    
    def test_generate_performance_benchmarks(self, tester_agent):
        """Test generating performance benchmarks."""
        analysis = {
            "language": "python",
            "test_types_needed": ["performance"]
        }
        
        benchmarks = tester_agent._generate_performance_benchmarks(analysis)
        
        assert len(benchmarks) > 0
        assert all("name" in benchmark for benchmark in benchmarks)
        assert all("target" in benchmark for benchmark in benchmarks)
    
    def test_generate_security_tests(self, tester_agent):
        """Test generating security tests."""
        code = "def process_input(user_input):\n    return user_input"
        analysis = {
            "language": "python",
            "test_types_needed": ["security"]
        }
        
        security_tests = tester_agent._generate_security_tests(code, analysis)
        
        assert len(security_tests) > 0
        assert all(isinstance(tc, TestCase) for tc in security_tests)
        assert any(tc.test_type == "security" for tc in security_tests)
    
    def test_generate_testing_implementation_complete(self, tester_agent):
        """Test complete testing implementation generation."""
        question = "Create comprehensive tests for my Python API"
        code = "def get_user(user_id):\n    return {'id': user_id, 'name': 'Test User'}"
        analysis = {
            "language": "python",
            "test_types_needed": ["unit", "integration", "security"],
            "complexity_level": "medium",
            "testing_focus": ["api_testing"],
            "frameworks_recommended": ["pytest", "fastapi"]
        }
        
        result = tester_agent._generate_testing_implementation(question, code, analysis, None)
        
        assert result is not None
        assert isinstance(result, TestingResult)
        assert len(result.test_suites) > 0
        assert result.automation_config is not None
        assert len(result.execution_instructions) > 0
    
    def test_process_testing_request(self, tester_agent, mock_rag_manager):
        """Test processing testing request message."""
        message = Message(
            sender_id="user",
            recipient_id="tester_test",
            content="Create unit tests for this function:\n```python\ndef add(a, b):\n    return a + b\n```",
            message_type=MessageType.USER_QUESTION
        )
        
        # Mock RAG manager response
        mock_rag_manager.retrieve_vector_context.return_value = []
        
        response = tester_agent._handle_testing_request(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "tester_test"
        assert response.confidence_score > 0
        assert "Testing Strategy" in response.content
    
    def test_process_message_no_code(self, tester_agent, mock_rag_manager):
        """Test processing message without code."""
        message = Message(
            sender_id="user",
            recipient_id="tester_test",
            content="Create tests for my application",
            message_type=MessageType.USER_QUESTION
        )
        
        # Mock RAG manager response
        mock_rag_manager.retrieve_vector_context.return_value = []
        
        response = tester_agent._handle_testing_request(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "tester_test"
        assert response.confidence_score > 0
        # Should still generate tests even without specific code
    
    def test_public_create_test_suite_method(self, tester_agent):
        """Test public create_test_suite method."""
        code = "def multiply(a, b):\n    return a * b"
        
        result = tester_agent.create_test_suite(code, ["unit"], "python")
        
        assert result is not None
        assert isinstance(result, TestingResult)
        assert len(result.test_suites) > 0
    
    def test_public_generate_security_tests_method(self, tester_agent):
        """Test public generate_security_tests method."""
        code = "def process_user_input(input_data):\n    return input_data"
        
        security_tests = tester_agent.generate_security_tests(code)
        
        assert isinstance(security_tests, list)
        assert len(security_tests) > 0
        assert all(isinstance(test, TestCase) for test in security_tests)


class TestTestCase:
    """Test cases for TestCase class."""
    
    def test_test_case_creation(self):
        """Test TestCase object creation."""
        test_case = TestCase(
            name="test_addition",
            test_type="unit",
            description="Test addition function",
            code="def test_addition():\n    assert add(2, 3) == 5",
            setup_code="from calculator import add",
            teardown_code="# cleanup"
        )
        
        assert test_case.name == "test_addition"
        assert test_case.test_type == "unit"
        assert test_case.description == "Test addition function"
        assert "assert add(2, 3) == 5" in test_case.code
        assert test_case.setup_code == "from calculator import add"
        assert test_case.teardown_code == "# cleanup"
    
    def test_test_case_to_dict(self):
        """Test TestCase to_dict method."""
        test_case = TestCase(
            name="test_basic",
            test_type="unit",
            description="Basic test",
            code="assert True"
        )
        
        result = test_case.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test_basic"
        assert result["test_type"] == "unit"
        assert result["code"] == "assert True"


class TestTestSuite:
    """Test cases for TestSuite class."""
    
    def test_test_suite_creation(self):
        """Test TestSuite object creation."""
        test_case = TestCase("test_basic", "unit", "Basic test", "assert True")
        test_suite = TestSuite(
            name="Unit Tests",
            description="Collection of unit tests",
            test_cases=[test_case],
            setup_code="# setup",
            teardown_code="# teardown"
        )
        
        assert test_suite.name == "Unit Tests"
        assert test_suite.description == "Collection of unit tests"
        assert len(test_suite.test_cases) == 1
        assert test_suite.setup_code == "# setup"
        assert test_suite.teardown_code == "# teardown"
    
    def test_test_suite_to_dict(self):
        """Test TestSuite to_dict method."""
        test_case = TestCase("test_basic", "unit", "Basic test", "assert True")
        test_suite = TestSuite(
            name="Unit Tests",
            description="Collection of unit tests",
            test_cases=[test_case]
        )
        
        result = test_suite.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "Unit Tests"
        assert len(result["test_cases"]) == 1


class TestTestStrategy:
    """Test cases for TestStrategy class."""
    
    def test_test_strategy_creation(self):
        """Test TestStrategy object creation."""
        strategy = TestStrategy(
            approach="TDD approach",
            test_types=["unit", "integration"],
            coverage_goals={"unit": "90%", "integration": "80%"},
            priorities=["Unit tests first", "Integration tests second"],
            tools_recommended=["pytest", "coverage"]
        )
        
        assert strategy.approach == "TDD approach"
        assert strategy.test_types == ["unit", "integration"]
        assert strategy.coverage_goals["unit"] == "90%"
        assert len(strategy.priorities) == 2
        assert "pytest" in strategy.tools_recommended
    
    def test_test_strategy_to_dict(self):
        """Test TestStrategy to_dict method."""
        strategy = TestStrategy(
            approach="Basic testing",
            test_types=["unit"],
            coverage_goals={"unit": "80%"},
            priorities=["Test core functionality"],
            tools_recommended=["pytest"]
        )
        
        result = strategy.to_dict()
        
        assert isinstance(result, dict)
        assert result["approach"] == "Basic testing"
        assert result["test_types"] == ["unit"]