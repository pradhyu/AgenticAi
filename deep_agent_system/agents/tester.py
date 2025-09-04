"""Tester Agent for test creation and quality assurance."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.models.agents import AgentType, Capability
from deep_agent_system.models.messages import (
    Context,
    Message,
    MessageType,
    Response,
    RetrievalType,
)


logger = logging.getLogger(__name__)


class TestCase:
    """Represents a single test case."""
    
    def __init__(
        self,
        name: str,
        test_type: str,
        description: str,
        code: str,
        setup_code: Optional[str] = None,
        teardown_code: Optional[str] = None
    ):
        self.name = name
        self.test_type = test_type  # "unit", "integration", "performance", "security"
        self.description = description
        self.code = code
        self.setup_code = setup_code
        self.teardown_code = teardown_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "test_type": self.test_type,
            "description": self.description,
            "code": self.code,
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code,
        }


class TestSuite:
    """Represents a collection of related test cases."""
    
    def __init__(
        self,
        name: str,
        description: str,
        test_cases: List[TestCase],
        setup_code: Optional[str] = None,
        teardown_code: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.test_cases = test_cases
        self.setup_code = setup_code
        self.teardown_code = teardown_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code,
        }


class TestStrategy:
    """Represents a comprehensive testing strategy."""
    
    def __init__(
        self,
        approach: str,
        test_types: List[str],
        coverage_goals: Dict[str, str],
        priorities: List[str],
        tools_recommended: List[str]
    ):
        self.approach = approach
        self.test_types = test_types
        self.coverage_goals = coverage_goals
        self.priorities = priorities
        self.tools_recommended = tools_recommended
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "approach": self.approach,
            "test_types": self.test_types,
            "coverage_goals": self.coverage_goals,
            "priorities": self.priorities,
            "tools_recommended": self.tools_recommended,
        }


class TestingResult:
    """Complete testing analysis and implementation result."""
    
    def __init__(
        self,
        strategy: TestStrategy,
        test_suites: List[TestSuite],
        automation_config: Dict[str, Any],
        execution_instructions: List[str],
        performance_benchmarks: List[Dict[str, Any]],
        security_tests: List[TestCase]
    ):
        self.strategy = strategy
        self.test_suites = test_suites
        self.automation_config = automation_config
        self.execution_instructions = execution_instructions
        self.performance_benchmarks = performance_benchmarks
        self.security_tests = security_tests    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "strategy": self.strategy.to_dict(),
            "test_suites": [ts.to_dict() for ts in self.test_suites],
            "automation_config": self.automation_config,
            "execution_instructions": self.execution_instructions,
            "performance_benchmarks": self.performance_benchmarks,
            "security_tests": [st.to_dict() for st in self.security_tests],
        }


class TesterAgent(BaseAgent):
    """Tester agent responsible for test creation and quality assurance."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Tester agent."""
        super().__init__(*args, **kwargs)
        
        # Validate that this agent has the required capabilities
        required_capabilities = [Capability.TEST_CREATION]
        for capability in required_capabilities:
            if not self.has_capability(capability):
                logger.warning(
                    f"TesterAgent {self.agent_id} missing required capability: {capability}"
                )
        
        # Testing frameworks by language
        self._testing_frameworks = {
            "python": {
                "unit": ["pytest", "unittest"],
                "integration": ["pytest", "requests"],
                "performance": ["pytest-benchmark", "locust"],
                "security": ["bandit", "safety"],
                "mocking": ["unittest.mock", "pytest-mock"]
            },
            "javascript": {
                "unit": ["jest", "mocha", "vitest"],
                "integration": ["supertest", "cypress"],
                "performance": ["artillery", "k6"],
                "security": ["eslint-plugin-security", "audit"],
                "mocking": ["jest", "sinon"]
            },
            "java": {
                "unit": ["junit", "testng"],
                "integration": ["spring-boot-test", "testcontainers"],
                "performance": ["jmeter", "gatling"],
                "security": ["spotbugs", "owasp-dependency-check"],
                "mocking": ["mockito", "powermock"]
            }
        }
        
        # Test patterns and templates
        self._test_patterns = {
            "unit_test_structure": [
                "Arrange (setup test data)",
                "Act (execute the code under test)",
                "Assert (verify the results)"
            ],
            "integration_test_structure": [
                "Setup test environment",
                "Execute integration scenario",
                "Verify end-to-end behavior",
                "Cleanup resources"
            ]
        }
        
        # Common test scenarios
        self._common_scenarios = {
            "api_testing": ["GET requests", "POST requests", "PUT requests", "DELETE requests", "Error handling"],
            "database_testing": ["CRUD operations", "Transactions", "Constraints", "Performance"],
            "security_testing": ["Input validation", "Authentication", "Authorization", "SQL injection", "XSS"],
            "performance_testing": ["Load testing", "Stress testing", "Spike testing", "Volume testing"]
        }
        
        logger.info(f"TesterAgent {self.agent_id} initialized successfully")
    
    def _process_message_impl(self, message: Message) -> Response:
        """Process incoming messages for test creation."""
        try:
            if message.message_type in [MessageType.USER_QUESTION, MessageType.AGENT_REQUEST]:
                return self._handle_testing_request(message)
            else:
                return self._create_error_response(
                    message.id,
                    f"Unsupported message type: {message.message_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message in TesterAgent: {e}")
            return self._create_error_response(message.id, str(e))
    
    def _handle_testing_request(self, message: Message) -> Response:
        """Handle testing and quality assurance requests."""
        question = message.content
        
        # Extract code from the message if present
        code_content = self._extract_code_from_message(question)
        
        # Step 1: Retrieve relevant testing patterns and best practices using RAG
        context = self._retrieve_testing_context(question, code_content)
        
        # Step 2: Analyze testing requirements
        testing_analysis = self._analyze_testing_requirements(question, code_content, context)
        
        # Step 3: Generate comprehensive testing implementation
        testing_result = self._generate_testing_implementation(question, code_content, testing_analysis, context)
        
        # Step 4: Create comprehensive response
        response_content = self._format_testing_response(
            question, code_content, testing_result, context
        )
        
        # Calculate confidence based on code availability and context
        confidence = self._calculate_confidence(question, code_content, context, testing_result)
        
        context_used = []
        if context and context.documents:
            context_used.extend([f"testing_pattern_{i}" for i in range(len(context.documents))])
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            confidence_score=confidence,
            context_used=context_used,
            metadata={
                "testing_result": testing_result.to_dict() if testing_result else None,
                "testing_domain": True,
                "has_code": code_content is not None,
            }
        )
    
    def _extract_code_from_message(self, message: str) -> Optional[str]:
        """Extract code blocks from the message."""
        # Look for code blocks in markdown format
        code_block_pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(code_block_pattern, message, re.DOTALL)
        
        if matches:
            return matches[0]  # Return the first code block
        
        # Look for inline code or code-like content
        lines = message.split('\n')
        code_lines = []
        
        for line in lines:
            # Check if line looks like code
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'import ', 'from ',
                                                  'if ', 'for ', 'while ', 'try:', 'except:',
                                                  'public ', 'private ', 'const ', 'let ', 'var ']):
                code_lines.append(line)
            elif code_lines and (line.startswith('    ') or line.startswith('\t')):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else None
    
    def _retrieve_testing_context(self, question: str, code_content: Optional[str]) -> Optional[Context]:
        """Retrieve context with emphasis on testing patterns and best practices."""
        if not self.rag_manager:
            return None
        
        try:
            # Create a query combining the question and testing focus
            language = self._detect_language(code_content) if code_content else "generic"
            query = f"testing best practices {question} {language} unit tests integration tests"
            
            # Use vector search for testing patterns and documentation
            return self.retrieve_context(query, RetrievalType.VECTOR, k=5)
            
        except Exception as e:
            logger.error(f"Error retrieving testing context: {e}")
            return None
    
    def _detect_language(self, code_content: str) -> str:
        """Detect programming language from code content."""
        if not code_content:
            return "generic"
        
        code_lower = code_content.lower()
        
        # Python indicators
        if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'class ', 'self.', 'elif ']):
            return "python"
        
        # JavaScript indicators
        elif any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
            return "javascript"
        
        # Java indicators
        elif any(keyword in code_lower for keyword in ['public class', 'private ', 'protected ', 'package ', 'import java']):
            return "java"
        
        return "generic"    

    def _analyze_testing_requirements(
        self,
        question: str,
        code_content: Optional[str],
        context: Optional[Context]
    ) -> Dict[str, Any]:
        """Analyze testing requirements from the question and code."""
        analysis = {
            "language": self._detect_language(code_content) if code_content else "generic",
            "test_types_needed": [],
            "complexity_level": "medium",
            "testing_focus": [],
            "frameworks_recommended": []
        }
        
        question_lower = question.lower()
        
        # Determine test types needed
        if "unit test" in question_lower or "unit" in question_lower:
            analysis["test_types_needed"].append("unit")
        
        if "integration" in question_lower or "api" in question_lower:
            analysis["test_types_needed"].append("integration")
        
        if "performance" in question_lower or "load" in question_lower:
            analysis["test_types_needed"].append("performance")
        
        if "security" in question_lower or "vulnerability" in question_lower:
            analysis["test_types_needed"].append("security")
        
        # Default to unit tests if none specified
        if not analysis["test_types_needed"]:
            analysis["test_types_needed"] = ["unit", "integration"]
        
        # Determine testing focus
        if "api" in question_lower or "rest" in question_lower:
            analysis["testing_focus"].append("api_testing")
        
        if "database" in question_lower or "db" in question_lower:
            analysis["testing_focus"].append("database_testing")
        
        if "auth" in question_lower or "login" in question_lower:
            analysis["testing_focus"].append("security_testing")
        
        # Assess complexity
        if code_content:
            lines = len(code_content.split('\n'))
            if lines > 50:
                analysis["complexity_level"] = "high"
            elif lines < 20:
                analysis["complexity_level"] = "simple"
        
        # Recommend frameworks
        language = analysis["language"]
        if language in self._testing_frameworks:
            frameworks = self._testing_frameworks[language]
            for test_type in analysis["test_types_needed"]:
                if test_type in frameworks:
                    analysis["frameworks_recommended"].extend(frameworks[test_type])
        
        # Remove duplicates
        analysis["frameworks_recommended"] = list(set(analysis["frameworks_recommended"]))
        
        return analysis
    
    def _generate_testing_implementation(
        self,
        question: str,
        code_content: Optional[str],
        analysis: Dict[str, Any],
        context: Optional[Context]
    ) -> Optional[TestingResult]:
        """Generate comprehensive testing implementation."""
        try:
            # Generate testing strategy
            strategy = self._generate_testing_strategy(analysis)
            
            # Generate test suites
            test_suites = self._generate_test_suites(code_content, analysis)
            
            # Generate automation configuration
            automation_config = self._generate_automation_config(analysis)
            
            # Generate execution instructions
            execution_instructions = self._generate_execution_instructions(analysis)
            
            # Generate performance benchmarks
            performance_benchmarks = self._generate_performance_benchmarks(analysis)
            
            # Generate security tests
            security_tests = self._generate_security_tests(code_content, analysis)
            
            return TestingResult(
                strategy=strategy,
                test_suites=test_suites,
                automation_config=automation_config,
                execution_instructions=execution_instructions,
                performance_benchmarks=performance_benchmarks,
                security_tests=security_tests
            )
            
        except Exception as e:
            logger.error(f"Error generating testing implementation: {e}")
            return None
    
    def _generate_testing_strategy(self, analysis: Dict[str, Any]) -> TestStrategy:
        """Generate testing strategy based on analysis."""
        approach = "Test-Driven Development (TDD) with comprehensive coverage"
        
        if analysis["complexity_level"] == "simple":
            approach = "Basic unit testing with essential integration tests"
        elif analysis["complexity_level"] == "high":
            approach = "Comprehensive testing strategy with multiple test layers and automation"
        
        test_types = analysis["test_types_needed"]
        
        coverage_goals = {
            "unit_tests": "90% code coverage for core business logic",
            "integration_tests": "100% coverage of API endpoints and critical workflows",
            "performance_tests": "Response time < 200ms for 95% of requests",
            "security_tests": "Zero critical vulnerabilities"
        }
        
        priorities = [
            "Unit tests for core functionality",
            "Integration tests for critical workflows",
            "Error handling and edge cases",
            "Performance and security validation"
        ]
        
        return TestStrategy(
            approach=approach,
            test_types=test_types,
            coverage_goals=coverage_goals,
            priorities=priorities,
            tools_recommended=analysis["frameworks_recommended"]
        )
    
    def _generate_test_suites(self, code_content: Optional[str], analysis: Dict[str, Any]) -> List[TestSuite]:
        """Generate test suites based on code and analysis."""
        test_suites = []
        language = analysis["language"]
        
        # Generate unit test suite
        if "unit" in analysis["test_types_needed"]:
            unit_tests = self._generate_unit_tests(code_content, language)
            if unit_tests:
                test_suites.append(TestSuite(
                    name="Unit Tests",
                    description="Unit tests for individual functions and classes",
                    test_cases=unit_tests
                ))
        
        # Generate integration test suite
        if "integration" in analysis["test_types_needed"]:
            integration_tests = self._generate_integration_tests(code_content, language, analysis)
            if integration_tests:
                test_suites.append(TestSuite(
                    name="Integration Tests",
                    description="Integration tests for API endpoints and workflows",
                    test_cases=integration_tests
                ))
        
        return test_suites
    
    def _generate_unit_tests(self, code_content: Optional[str], language: str) -> List[TestCase]:
        """Generate unit test cases."""
        test_cases = []
        
        if language == "python":
            test_cases.extend(self._generate_python_unit_tests(code_content))
        elif language == "javascript":
            test_cases.extend(self._generate_javascript_unit_tests(code_content))
        elif language == "java":
            test_cases.extend(self._generate_java_unit_tests(code_content))
        else:
            # Generic unit test template
            test_cases.append(TestCase(
                name="test_basic_functionality",
                test_type="unit",
                description="Basic functionality test",
                code=self._generate_generic_unit_test()
            ))
        
        return test_cases
    
    def _generate_python_unit_tests(self, code_content: Optional[str]) -> List[TestCase]:
        """Generate Python-specific unit tests."""
        test_cases = []
        
        # Basic unit test structure
        test_cases.append(TestCase(
            name="test_function_basic",
            test_type="unit",
            description="Test basic function behavior",
            code='''import pytest
from unittest.mock import Mock, patch
from your_module import YourClass, your_function

class TestYourClass:
    """Test cases for YourClass."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.instance = YourClass()
    
    def test_initialization(self):
        """Test class initialization."""
        assert self.instance is not None
        assert hasattr(self.instance, 'expected_attribute')
    
    def test_method_with_valid_input(self):
        """Test method with valid input."""
        # Arrange
        input_data = "test_input"
        expected_result = "expected_output"
        
        # Act
        result = self.instance.process(input_data)
        
        # Assert
        assert result == expected_result
    
    def test_method_with_invalid_input(self):
        """Test method with invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            self.instance.process(None)
    
    def test_method_with_edge_case(self):
        """Test method with edge case."""
        # Test empty string
        result = self.instance.process("")
        assert result == ""
        
        # Test very long string
        long_input = "a" * 1000
        result = self.instance.process(long_input)
        assert len(result) <= 1000

class TestYourFunction:
    """Test cases for standalone functions."""
    
    def test_function_normal_case(self):
        """Test function with normal input."""
        result = your_function("input")
        assert result is not None
    
    def test_function_with_mock(self):
        """Test function with mocked dependencies."""
        with patch('your_module.external_dependency') as mock_dep:
            mock_dep.return_value = "mocked_result"
            
            result = your_function("input")
            
            mock_dep.assert_called_once_with("input")
            assert result == "mocked_result"
    
    @pytest.mark.parametrize("input_val,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
        ("test3", "result3"),
    ])
    def test_function_parametrized(self, input_val, expected):
        """Test function with multiple parameter sets."""
        result = your_function(input_val)
        assert result == expected'''
        ))
        
        # Error handling test
        test_cases.append(TestCase(
            name="test_error_handling",
            test_type="unit",
            description="Test error handling scenarios",
            code='''def test_error_handling():
    """Test various error scenarios."""
    instance = YourClass()
    
    # Test with None input
    with pytest.raises(TypeError):
        instance.process(None)
    
    # Test with wrong type
    with pytest.raises(TypeError):
        instance.process(123)  # Expecting string
    
    # Test with empty input
    result = instance.process("")
    assert result == "" or result is None
    
    # Test with malformed input
    with pytest.raises(ValueError):
        instance.process("malformed_input")'''
        ))
        
        return test_cases    

    def _generate_javascript_unit_tests(self, code_content: Optional[str]) -> List[TestCase]:
        """Generate JavaScript-specific unit tests."""
        test_cases = []
        
        test_cases.append(TestCase(
            name="test_basic_functionality",
            test_type="unit",
            description="Basic JavaScript unit tests with Jest",
            code='''const { YourClass, yourFunction } = require('./your-module');

describe('YourClass', () => {
    let instance;
    
    beforeEach(() => {
        instance = new YourClass();
    });
    
    afterEach(() => {
        // Cleanup after each test
        jest.clearAllMocks();
    });
    
    test('should initialize correctly', () => {
        expect(instance).toBeDefined();
        expect(instance.property).toBeDefined();
    });
    
    test('should process valid input', () => {
        // Arrange
        const input = 'test input';
        const expected = 'expected output';
        
        // Act
        const result = instance.process(input);
        
        // Assert
        expect(result).toBe(expected);
    });
    
    test('should handle invalid input', () => {
        expect(() => {
            instance.process(null);
        }).toThrow('Invalid input');
    });
    
    test('should handle edge cases', () => {
        // Empty string
        expect(instance.process('')).toBe('');
        
        // Very long string
        const longInput = 'a'.repeat(1000);
        const result = instance.process(longInput);
        expect(result.length).toBeLessThanOrEqual(1000);
    });
});

describe('yourFunction', () => {
    test('should return correct result for normal input', () => {
        const result = yourFunction('input');
        expect(result).toBeDefined();
    });
    
    test('should work with mocked dependencies', () => {
        // Mock external dependency
        const mockDependency = jest.fn().mockReturnValue('mocked result');
        
        // Replace the dependency
        const originalDep = require('./your-module').externalDependency;
        require('./your-module').externalDependency = mockDependency;
        
        const result = yourFunction('input');
        
        expect(mockDependency).toHaveBeenCalledWith('input');
        expect(result).toBe('mocked result');
        
        // Restore original dependency
        require('./your-module').externalDependency = originalDep;
    });
    
    test.each([
        ['input1', 'output1'],
        ['input2', 'output2'],
        ['input3', 'output3'],
    ])('should handle %s and return %s', (input, expected) => {
        const result = yourFunction(input);
        expect(result).toBe(expected);
    });
});'''
        ))
        
        return test_cases
    
    def _generate_java_unit_tests(self, code_content: Optional[str]) -> List[TestCase]:
        """Generate Java-specific unit tests."""
        test_cases = []
        
        test_cases.append(TestCase(
            name="test_basic_functionality",
            test_type="unit",
            description="Basic Java unit tests with JUnit 5",
            code='''import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class YourClassTest {
    
    private YourClass instance;
    
    @Mock
    private ExternalDependency mockDependency;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        instance = new YourClass();
    }
    
    @AfterEach
    void tearDown() {
        // Cleanup after each test
    }
    
    @Test
    @DisplayName("Should initialize correctly")
    void testInitialization() {
        assertNotNull(instance);
        assertNotNull(instance.getProperty());
    }
    
    @Test
    @DisplayName("Should process valid input correctly")
    void testProcessValidInput() {
        // Arrange
        String input = "test input";
        String expected = "expected output";
        
        // Act
        String result = instance.process(input);
        
        // Assert
        assertEquals(expected, result);
    }
    
    @Test
    @DisplayName("Should throw exception for invalid input")
    void testProcessInvalidInput() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> instance.process(null)
        );
        
        assertEquals("Invalid input", exception.getMessage());
    }
    
    @ParameterizedTest
    @ValueSource(strings = {"", "test", "longer test string"})
    @DisplayName("Should handle various string inputs")
    void testProcessVariousInputs(String input) {
        String result = instance.process(input);
        assertNotNull(result);
    }
    
    @ParameterizedTest
    @CsvSource({
        "input1, output1",
        "input2, output2",
        "input3, output3"
    })
    @DisplayName("Should map inputs to expected outputs")
    void testInputOutputMapping(String input, String expected) {
        String result = instance.process(input);
        assertEquals(expected, result);
    }
    
    @Test
    @DisplayName("Should work with mocked dependencies")
    void testWithMockedDependencies() {
        // Arrange
        when(mockDependency.getData()).thenReturn("mocked data");
        instance.setDependency(mockDependency);
        
        // Act
        String result = instance.processWithDependency();
        
        // Assert
        verify(mockDependency).getData();
        assertEquals("processed: mocked data", result);
    }
}'''
        ))
        
        return test_cases
    
    def _generate_generic_unit_test(self) -> str:
        """Generate a generic unit test template."""
        return '''// Generic unit test template
function testBasicFunctionality() {
    // Arrange
    const input = "test input";
    const expected = "expected output";
    
    // Act
    const result = functionUnderTest(input);
    
    // Assert
    assert(result === expected, "Function should return expected output");
}

function testErrorHandling() {
    // Test with invalid input
    try {
        functionUnderTest(null);
        assert(false, "Should have thrown an error");
    } catch (error) {
        assert(error.message.includes("Invalid"), "Should throw meaningful error");
    }
}

function testEdgeCases() {
    // Test with empty input
    const emptyResult = functionUnderTest("");
    assert(emptyResult !== undefined, "Should handle empty input");
    
    // Test with boundary values
    const boundaryResult = functionUnderTest("boundary_value");
    assert(boundaryResult !== null, "Should handle boundary values");
}'''
    
    def _generate_integration_tests(
        self,
        code_content: Optional[str],
        language: str,
        analysis: Dict[str, Any]
    ) -> List[TestCase]:
        """Generate integration test cases."""
        test_cases = []
        
        if "api_testing" in analysis.get("testing_focus", []):
            test_cases.extend(self._generate_api_integration_tests(language))
        
        if "database_testing" in analysis.get("testing_focus", []):
            test_cases.extend(self._generate_database_integration_tests(language))
        
        # Default integration tests if no specific focus
        if not test_cases:
            test_cases.extend(self._generate_default_integration_tests(language))
        
        return test_cases
    
    def _generate_api_integration_tests(self, language: str) -> List[TestCase]:
        """Generate API integration tests."""
        test_cases = []
        
        if language == "python":
            test_cases.append(TestCase(
                name="test_api_endpoints",
                test_type="integration",
                description="Integration tests for API endpoints",
                code='''import pytest
import requests
from fastapi.testclient import TestClient
from your_app import app

client = TestClient(app)

class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    def test_get_items_endpoint(self):
        """Test GET /items endpoint."""
        response = client.get("/items")
        
        assert response.status_code == 200
        assert "data" in response.json()
        assert isinstance(response.json()["data"], list)
    
    def test_create_item_endpoint(self):
        """Test POST /items endpoint."""
        item_data = {
            "name": "Test Item",
            "description": "Test Description"
        }
        
        response = client.post("/items", json=item_data)
        
        assert response.status_code == 201
        assert response.json()["success"] is True
        assert response.json()["data"]["name"] == item_data["name"]
    
    def test_get_item_by_id_endpoint(self):
        """Test GET /items/{id} endpoint."""
        # First create an item
        item_data = {"name": "Test Item", "description": "Test"}
        create_response = client.post("/items", json=item_data)
        item_id = create_response.json()["data"]["id"]
        
        # Then retrieve it
        response = client.get(f"/items/{item_id}")
        
        assert response.status_code == 200
        assert response.json()["data"]["id"] == item_id
    
    def test_update_item_endpoint(self):
        """Test PUT /items/{id} endpoint."""
        # Create item
        item_data = {"name": "Original", "description": "Original"}
        create_response = client.post("/items", json=item_data)
        item_id = create_response.json()["data"]["id"]
        
        # Update item
        update_data = {"name": "Updated", "description": "Updated"}
        response = client.put(f"/items/{item_id}", json=update_data)
        
        assert response.status_code == 200
        assert response.json()["data"]["name"] == "Updated"
    
    def test_delete_item_endpoint(self):
        """Test DELETE /items/{id} endpoint."""
        # Create item
        item_data = {"name": "To Delete", "description": "Will be deleted"}
        create_response = client.post("/items", json=item_data)
        item_id = create_response.json()["data"]["id"]
        
        # Delete item
        response = client.delete(f"/items/{item_id}")
        
        assert response.status_code == 200
        
        # Verify deletion
        get_response = client.get(f"/items/{item_id}")
        assert get_response.status_code == 404
    
    def test_error_handling(self):
        """Test API error handling."""
        # Test 404 for non-existent item
        response = client.get("/items/99999")
        assert response.status_code == 404
        
        # Test 400 for invalid data
        invalid_data = {"invalid_field": "value"}
        response = client.post("/items", json=invalid_data)
        assert response.status_code == 400'''
            ))
        
        elif language == "javascript":
            test_cases.append(TestCase(
                name="test_api_endpoints",
                test_type="integration",
                description="Integration tests for Express API",
                code='''const request = require('supertest');
const app = require('../server');

describe('API Integration Tests', () => {
    let itemId;
    
    beforeEach(async () => {
        // Setup test data before each test
    });
    
    afterEach(async () => {
        // Cleanup after each test
    });
    
    describe('GET /items', () => {
        it('should return all items', async () => {
            const response = await request(app)
                .get('/items')
                .expect(200);
            
            expect(response.body.success).toBe(true);
            expect(Array.isArray(response.body.data)).toBe(true);
        });
    });
    
    describe('POST /items', () => {
        it('should create a new item', async () => {
            const itemData = {
                name: 'Test Item',
                description: 'Test Description'
            };
            
            const response = await request(app)
                .post('/items')
                .send(itemData)
                .expect(201);
            
            expect(response.body.success).toBe(true);
            expect(response.body.data.name).toBe(itemData.name);
            
            itemId = response.body.data.id;
        });
        
        it('should return error for invalid data', async () => {
            const invalidData = {};
            
            const response = await request(app)
                .post('/items')
                .send(invalidData)
                .expect(400);
            
            expect(response.body.success).toBe(false);
        });
    });
    
    describe('GET /items/:id', () => {
        beforeEach(async () => {
            // Create test item
            const itemData = { name: 'Test Item', description: 'Test' };
            const response = await request(app)
                .post('/items')
                .send(itemData);
            itemId = response.body.data.id;
        });
        
        it('should return specific item', async () => {
            const response = await request(app)
                .get(`/items/${itemId}`)
                .expect(200);
            
            expect(response.body.success).toBe(true);
            expect(response.body.data.id).toBe(itemId);
        });
        
        it('should return 404 for non-existent item', async () => {
            const response = await request(app)
                .get('/items/99999')
                .expect(404);
            
            expect(response.body.success).toBe(false);
        });
    });
    
    describe('PUT /items/:id', () => {
        beforeEach(async () => {
            const itemData = { name: 'Original', description: 'Original' };
            const response = await request(app)
                .post('/items')
                .send(itemData);
            itemId = response.body.data.id;
        });
        
        it('should update existing item', async () => {
            const updateData = { name: 'Updated', description: 'Updated' };
            
            const response = await request(app)
                .put(`/items/${itemId}`)
                .send(updateData)
                .expect(200);
            
            expect(response.body.success).toBe(true);
            expect(response.body.data.name).toBe('Updated');
        });
    });
    
    describe('DELETE /items/:id', () => {
        beforeEach(async () => {
            const itemData = { name: 'To Delete', description: 'Will be deleted' };
            const response = await request(app)
                .post('/items')
                .send(itemData);
            itemId = response.body.data.id;
        });
        
        it('should delete existing item', async () => {
            await request(app)
                .delete(`/items/${itemId}`)
                .expect(200);
            
            // Verify deletion
            await request(app)
                .get(`/items/${itemId}`)
                .expect(404);
        });
    });
});'''
            ))
        
        return test_cases
    
    def _generate_database_integration_tests(self, language: str) -> List[TestCase]:
        """Generate database integration tests."""
        test_cases = []
        
        if language == "python":
            test_cases.append(TestCase(
                name="test_database_operations",
                test_type="integration",
                description="Database integration tests",
                code='''import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from your_app.models import Base, Item
from your_app.database import get_db

# Test database URL (use in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="function")
def test_db():
    """Create test database session."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=engine)

class TestDatabaseOperations:
    """Test database CRUD operations."""
    
    def test_create_item(self, test_db):
        """Test creating an item in database."""
        item_data = {
            "name": "Test Item",
            "description": "Test Description"
        }
        
        item = Item(**item_data)
        test_db.add(item)
        test_db.commit()
        test_db.refresh(item)
        
        assert item.id is not None
        assert item.name == item_data["name"]
        assert item.created_at is not None
    
    def test_read_item(self, test_db):
        """Test reading an item from database."""
        # Create item
        item = Item(name="Test Item", description="Test")
        test_db.add(item)
        test_db.commit()
        test_db.refresh(item)
        
        # Read item
        retrieved_item = test_db.query(Item).filter(Item.id == item.id).first()
        
        assert retrieved_item is not None
        assert retrieved_item.name == "Test Item"
    
    def test_update_item(self, test_db):
        """Test updating an item in database."""
        # Create item
        item = Item(name="Original", description="Original")
        test_db.add(item)
        test_db.commit()
        test_db.refresh(item)
        
        # Update item
        item.name = "Updated"
        test_db.commit()
        test_db.refresh(item)
        
        assert item.name == "Updated"
        assert item.updated_at is not None
    
    def test_delete_item(self, test_db):
        """Test deleting an item from database."""
        # Create item
        item = Item(name="To Delete", description="Will be deleted")
        test_db.add(item)
        test_db.commit()
        item_id = item.id
        
        # Delete item
        test_db.delete(item)
        test_db.commit()
        
        # Verify deletion
        deleted_item = test_db.query(Item).filter(Item.id == item_id).first()
        assert deleted_item is None
    
    def test_query_items(self, test_db):
        """Test querying multiple items."""
        # Create multiple items
        items = [
            Item(name="Item 1", description="First"),
            Item(name="Item 2", description="Second"),
            Item(name="Item 3", description="Third")
        ]
        
        for item in items:
            test_db.add(item)
        test_db.commit()
        
        # Query all items
        all_items = test_db.query(Item).all()
        assert len(all_items) == 3
        
        # Query with filter
        filtered_items = test_db.query(Item).filter(Item.name.like("Item%")).all()
        assert len(filtered_items) == 3
    
    def test_database_constraints(self, test_db):
        """Test database constraints and validation."""
        # Test unique constraint (if applicable)
        item1 = Item(name="Unique Item", description="First")
        test_db.add(item1)
        test_db.commit()
        
        # Test foreign key constraint (if applicable)
        # This would depend on your specific model relationships
        
        # Test not null constraint
        with pytest.raises(Exception):  # Specific exception depends on your setup
            invalid_item = Item(name=None, description="Invalid")
            test_db.add(invalid_item)
            test_db.commit()'''
            ))
        
        return test_cases
    
    def _generate_default_integration_tests(self, language: str) -> List[TestCase]:
        """Generate default integration tests."""
        test_cases = []
        
        test_cases.append(TestCase(
            name="test_workflow_integration",
            test_type="integration",
            description="End-to-end workflow integration test",
            code=f'''// Default integration test for {language}
// This test verifies that different components work together correctly

function testCompleteWorkflow() {{
    // Setup test environment
    const testData = setupTestData();
    
    try {{
        // Step 1: Initialize system
        const system = initializeSystem();
        
        // Step 2: Process test data
        const result = system.processData(testData);
        
        // Step 3: Verify results
        assert(result.success === true, "Workflow should complete successfully");
        assert(result.data !== null, "Result should contain data");
        
        // Step 4: Verify side effects
        const sideEffects = system.getSideEffects();
        assert(sideEffects.length > 0, "Should have recorded side effects");
        
    }} finally {{
        // Cleanup
        cleanupTestData();
    }}
}}

function setupTestData() {{
    return {{
        input: "test input",
        config: {{ option: "test" }}
    }};
}}

function cleanupTestData() {{
    // Clean up any test data or resources
}}'''
        ))
        
        return test_cases
    
    def _generate_automation_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test automation configuration."""
        language = analysis["language"]
        
        config = {
            "ci_cd_integration": {},
            "test_execution": {},
            "reporting": {},
            "coverage": {}
        }
        
        if language == "python":
            config.update({
                "ci_cd_integration": {
                    "github_actions": {
                        "workflow_file": ".github/workflows/test.yml",
                        "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                        "commands": [
                            "pip install -r requirements.txt",
                            "pip install pytest pytest-cov",
                            "pytest --cov=your_module --cov-report=xml"
                        ]
                    }
                },
                "test_execution": {
                    "command": "pytest",
                    "config_file": "pytest.ini",
                    "parallel_execution": "pytest -n auto"
                },
                "coverage": {
                    "tool": "pytest-cov",
                    "minimum_coverage": 80,
                    "exclude_files": ["tests/*", "*/migrations/*"]
                }
            })
        
        elif language == "javascript":
            config.update({
                "ci_cd_integration": {
                    "github_actions": {
                        "workflow_file": ".github/workflows/test.yml",
                        "node_versions": ["16", "18", "20"],
                        "commands": [
                            "npm ci",
                            "npm run test:coverage"
                        ]
                    }
                },
                "test_execution": {
                    "command": "npm test",
                    "config_file": "jest.config.js",
                    "watch_mode": "npm run test:watch"
                },
                "coverage": {
                    "tool": "jest",
                    "minimum_coverage": 80,
                    "exclude_files": ["node_modules/*", "dist/*"]
                }
            })
        
        return config
    
    def _generate_execution_instructions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate test execution instructions."""
        language = analysis["language"]
        instructions = []
        
        if language == "python":
            instructions.extend([
                "Install test dependencies: pip install pytest pytest-cov pytest-mock",
                "Run all tests: pytest",
                "Run with coverage: pytest --cov=your_module",
                "Run specific test file: pytest tests/test_specific.py",
                "Run tests in parallel: pytest -n auto",
                "Generate HTML coverage report: pytest --cov=your_module --cov-report=html"
            ])
        
        elif language == "javascript":
            instructions.extend([
                "Install test dependencies: npm install --save-dev jest supertest",
                "Run all tests: npm test",
                "Run with coverage: npm run test:coverage",
                "Run in watch mode: npm run test:watch",
                "Run specific test: npm test -- test_file.test.js"
            ])
        
        else:
            instructions.extend([
                "Set up test environment according to your language/framework",
                "Run unit tests first to verify individual components",
                "Run integration tests to verify component interactions",
                "Check test coverage and aim for >80% coverage",
                "Run performance tests under expected load conditions"
            ])
        
        return instructions
    
    def _generate_performance_benchmarks(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance benchmarks and tests."""
        benchmarks = []
        
        if "performance" in analysis["test_types_needed"]:
            benchmarks.extend([
                {
                    "name": "Response Time Benchmark",
                    "description": "Measure API response times under normal load",
                    "target": "< 200ms for 95% of requests",
                    "test_type": "load_test",
                    "tool": "locust" if analysis["language"] == "python" else "artillery"
                },
                {
                    "name": "Throughput Benchmark",
                    "description": "Measure maximum requests per second",
                    "target": "> 1000 RPS",
                    "test_type": "stress_test",
                    "tool": "k6" if analysis["language"] == "javascript" else "jmeter"
                },
                {
                    "name": "Memory Usage Benchmark",
                    "description": "Monitor memory consumption under load",
                    "target": "< 512MB peak memory usage",
                    "test_type": "resource_test",
                    "tool": "memory_profiler"
                }
            ])
        
        return benchmarks
    
    def _generate_security_tests(self, code_content: Optional[str], analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate security test cases."""
        security_tests = []
        
        if "security" in analysis["test_types_needed"]:
            security_tests.append(TestCase(
                name="test_input_validation",
                test_type="security",
                description="Test input validation and sanitization",
                code='''def test_input_validation_security():
    """Test various input validation scenarios for security."""
    
    # Test SQL injection attempts
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'/*",
        "' UNION SELECT * FROM users --"
    ]
    
    for malicious_input in malicious_inputs:
        with pytest.raises((ValueError, SecurityError)):
            process_user_input(malicious_input)
    
    # Test XSS attempts
    xss_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "';alert('xss');//"
    ]
    
    for xss_input in xss_inputs:
        result = sanitize_input(xss_input)
        assert "<script>" not in result
        assert "javascript:" not in result
        assert "onerror=" not in result
    
    # Test path traversal attempts
    path_inputs = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    ]
    
    for path_input in path_inputs:
        with pytest.raises((ValueError, SecurityError)):
            process_file_path(path_input)'''
            ))
            
            security_tests.append(TestCase(
                name="test_authentication_security",
                test_type="security",
                description="Test authentication and authorization security",
                code='''def test_authentication_security():
    """Test authentication and authorization mechanisms."""
    
    # Test password strength requirements
    weak_passwords = ["123", "password", "admin", ""]
    for weak_password in weak_passwords:
        with pytest.raises(WeakPasswordError):
            validate_password(weak_password)
    
    # Test rate limiting
    for i in range(10):  # Assuming rate limit is 5 attempts
        try:
            failed_login_attempt("user", "wrong_password")
        except RateLimitExceeded:
            assert i >= 5, "Rate limiting should kick in after 5 attempts"
            break
    
    # Test session security
    session_token = create_session("user")
    assert len(session_token) >= 32, "Session token should be sufficiently long"
    assert session_token.isalnum() or set(session_token) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"), "Session token should be safe"
    
    # Test token expiration
    expired_token = create_expired_token()
    with pytest.raises(TokenExpiredError):
        validate_token(expired_token)'''
            ))
        
        return security_tests    

    def _format_testing_response(
        self,
        question: str,
        code_content: Optional[str],
        testing_result: Optional[TestingResult],
        context: Optional[Context]
    ) -> str:
        """Format the comprehensive testing response."""
        response = "# Comprehensive Testing Strategy & Implementation\n\n"
        
        if not testing_result:
            return response + "Unable to generate testing implementation."
        
        # Testing Strategy Section
        response += "## 1. Testing Strategy\n\n"
        strategy = testing_result.strategy
        response += f"**Approach:** {strategy.approach}\n\n"
        
        response += "**Test Types Recommended:**\n"
        for test_type in strategy.test_types:
            response += f"- {test_type.title()} Testing\n"
        response += "\n"
        
        response += "**Coverage Goals:**\n"
        for goal_type, goal_desc in strategy.coverage_goals.items():
            response += f"- **{goal_type.replace('_', ' ').title()}**: {goal_desc}\n"
        response += "\n"
        
        response += "**Testing Priorities:**\n"
        for i, priority in enumerate(strategy.priorities, 1):
            response += f"{i}. {priority}\n"
        response += "\n"
        
        if strategy.tools_recommended:
            response += "**Recommended Tools:**\n"
            for tool in strategy.tools_recommended:
                response += f"- {tool}\n"
            response += "\n"
        
        # Test Suites Section
        if testing_result.test_suites:
            response += "## 2. Test Implementation\n\n"
            
            for suite in testing_result.test_suites:
                response += f"### {suite.name}\n\n"
                response += f"*{suite.description}*\n\n"
                
                if suite.setup_code:
                    response += "**Setup Code:**\n"
                    response += f"```python\n{suite.setup_code}\n```\n\n"
                
                for test_case in suite.test_cases:
                    response += f"#### {test_case.name}\n\n"
                    response += f"*{test_case.description}*\n\n"
                    
                    # Determine language for syntax highlighting
                    lang = "python"  # Default
                    if "javascript" in test_case.code.lower() or "jest" in test_case.code:
                        lang = "javascript"
                    elif "java" in test_case.code.lower() or "junit" in test_case.code:
                        lang = "java"
                    
                    response += f"```{lang}\n{test_case.code}\n```\n\n"
                
                if suite.teardown_code:
                    response += "**Teardown Code:**\n"
                    response += f"```python\n{suite.teardown_code}\n```\n\n"
        
        # Performance Benchmarks Section
        if testing_result.performance_benchmarks:
            response += "## 3. Performance Testing\n\n"
            
            for benchmark in testing_result.performance_benchmarks:
                response += f"### {benchmark['name']}\n\n"
                response += f"**Description:** {benchmark['description']}\n\n"
                response += f"**Target:** {benchmark['target']}\n\n"
                response += f"**Tool:** {benchmark['tool']}\n\n"
                
                if benchmark['test_type'] == 'load_test':
                    response += "**Load Test Configuration:**\n"
                    response += "```yaml\n"
                    response += "config:\n"
                    response += "  target: 'http://localhost:8000'\n"
                    response += "  phases:\n"
                    response += "    - duration: 60\n"
                    response += "      arrivalRate: 10\n"
                    response += "    - duration: 120\n"
                    response += "      arrivalRate: 50\n"
                    response += "scenarios:\n"
                    response += "  - name: 'API Load Test'\n"
                    response += "    requests:\n"
                    response += "      - get:\n"
                    response += "          url: '/items'\n"
                    response += "      - post:\n"
                    response += "          url: '/items'\n"
                    response += "          json:\n"
                    response += "            name: 'Test Item'\n"
                    response += "```\n\n"
        
        # Security Tests Section
        if testing_result.security_tests:
            response += "## 4. Security Testing\n\n"
            
            for security_test in testing_result.security_tests:
                response += f"### {security_test.name}\n\n"
                response += f"*{security_test.description}*\n\n"
                response += f"```python\n{security_test.code}\n```\n\n"
        
        # Test Automation Section
        if testing_result.automation_config:
            response += "## 5. Test Automation & CI/CD Integration\n\n"
            
            config = testing_result.automation_config
            
            if "ci_cd_integration" in config:
                ci_config = config["ci_cd_integration"]
                if "github_actions" in ci_config:
                    gh_config = ci_config["github_actions"]
                    response += "### GitHub Actions Workflow\n\n"
                    response += f"Create file: `{gh_config['workflow_file']}`\n\n"
                    response += "```yaml\n"
                    response += "name: Tests\n\n"
                    response += "on: [push, pull_request]\n\n"
                    response += "jobs:\n"
                    response += "  test:\n"
                    response += "    runs-on: ubuntu-latest\n"
                    response += "    strategy:\n"
                    response += "      matrix:\n"
                    
                    if "python_versions" in gh_config:
                        response += f"        python-version: {gh_config['python_versions']}\n"
                    elif "node_versions" in gh_config:
                        response += f"        node-version: {gh_config['node_versions']}\n"
                    
                    response += "    steps:\n"
                    response += "      - uses: actions/checkout@v3\n"
                    
                    if "python_versions" in gh_config:
                        response += "      - name: Set up Python\n"
                        response += "        uses: actions/setup-python@v4\n"
                        response += "        with:\n"
                        response += "          python-version: ${{ matrix.python-version }}\n"
                    elif "node_versions" in gh_config:
                        response += "      - name: Set up Node.js\n"
                        response += "        uses: actions/setup-node@v3\n"
                        response += "        with:\n"
                        response += "          node-version: ${{ matrix.node-version }}\n"
                    
                    response += "      - name: Install dependencies and run tests\n"
                    response += "        run: |\n"
                    for command in gh_config["commands"]:
                        response += f"          {command}\n"
                    response += "```\n\n"
            
            if "coverage" in config:
                coverage_config = config["coverage"]
                response += "### Coverage Configuration\n\n"
                response += f"**Tool:** {coverage_config['tool']}\n\n"
                response += f"**Minimum Coverage:** {coverage_config['minimum_coverage']}%\n\n"
                response += "**Excluded Files:**\n"
                for excluded in coverage_config['exclude_files']:
                    response += f"- {excluded}\n"
                response += "\n"
        
        # Execution Instructions Section
        if testing_result.execution_instructions:
            response += "## 6. Test Execution Instructions\n\n"
            
            response += "### Quick Start\n\n"
            for i, instruction in enumerate(testing_result.execution_instructions, 1):
                response += f"{i}. {instruction}\n"
            response += "\n"
            
            response += "### Test Organization\n\n"
            response += "```\n"
            response += "tests/\n"
            response += "├── unit/\n"
            response += "│   ├── test_models.py\n"
            response += "│   ├── test_services.py\n"
            response += "│   └── test_utils.py\n"
            response += "├── integration/\n"
            response += "│   ├── test_api.py\n"
            response += "│   └── test_database.py\n"
            response += "├── performance/\n"
            response += "│   └── load_tests.py\n"
            response += "├── security/\n"
            response += "│   └── security_tests.py\n"
            response += "└── conftest.py\n"
            response += "```\n\n"
        
        # Best Practices Section
        response += "## 7. Testing Best Practices\n\n"
        
        response += "### Test Design Principles\n\n"
        response += "- **FIRST Principles**: Fast, Independent, Repeatable, Self-validating, Timely\n"
        response += "- **AAA Pattern**: Arrange, Act, Assert for clear test structure\n"
        response += "- **Test Isolation**: Each test should be independent and not rely on others\n"
        response += "- **Meaningful Names**: Test names should clearly describe what is being tested\n"
        response += "- **Single Responsibility**: Each test should verify one specific behavior\n\n"
        
        response += "### Test Data Management\n\n"
        response += "- Use factories or builders for creating test data\n"
        response += "- Keep test data minimal and focused on the test scenario\n"
        response += "- Clean up test data after each test to prevent interference\n"
        response += "- Use fixtures for common setup and teardown operations\n"
        response += "- Avoid hardcoded values; use constants or configuration\n\n"
        
        response += "### Mocking and Test Doubles\n\n"
        response += "- Mock external dependencies to isolate units under test\n"
        response += "- Use stubs for simple return values\n"
        response += "- Use spies to verify interactions\n"
        response += "- Avoid over-mocking; test real integrations when valuable\n"
        response += "- Keep mocks simple and focused on the test scenario\n\n"
        
        # Maintenance and Monitoring Section
        response += "## 8. Test Maintenance & Monitoring\n\n"
        
        response += "### Continuous Improvement\n\n"
        response += "1. **Regular Review**: Review and update tests as code evolves\n"
        response += "2. **Coverage Monitoring**: Track test coverage trends over time\n"
        response += "3. **Performance Tracking**: Monitor test execution times\n"
        response += "4. **Flaky Test Detection**: Identify and fix unreliable tests\n"
        response += "5. **Test Metrics**: Track test success rates and failure patterns\n\n"
        
        response += "### Quality Gates\n\n"
        response += "- All tests must pass before merging code\n"
        response += "- Maintain minimum code coverage thresholds\n"
        response += "- Performance tests must meet defined benchmarks\n"
        response += "- Security tests must pass without critical issues\n"
        response += "- New features must include corresponding tests\n\n"
        
        return response
    
    def _calculate_confidence(
        self,
        question: str,
        code_content: Optional[str],
        context: Optional[Context],
        testing_result: Optional[TestingResult]
    ) -> float:
        """Calculate confidence score for the testing response."""
        confidence = 0.8  # Base confidence for testing analysis
        
        # Adjust based on code availability
        if code_content:
            confidence += 0.1
            # More confidence for longer, more complex code
            lines = len(code_content.split('\n'))
            if lines > 20:
                confidence += 0.05
        
        # Adjust based on context availability
        if context and context.documents:
            confidence += 0.05
        
        # Adjust based on testing result completeness
        if testing_result:
            if testing_result.test_suites:
                confidence += 0.05
            if testing_result.automation_config:
                confidence += 0.03
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _create_error_response(self, message_id: str, error_message: str) -> Response:
        """Create an error response."""
        return Response(
            message_id=message_id,
            agent_id=self.agent_id,
            content=f"I encountered an error while creating your testing strategy: {error_message}",
            confidence_score=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def create_test_suite(
        self,
        code_content: str,
        test_types: List[str] = None,
        language: str = "python"
    ) -> Optional[TestingResult]:
        """Public method to create test suite for given code."""
        if test_types is None:
            test_types = ["unit", "integration"]
        
        analysis = {
            "language": language,
            "test_types_needed": test_types,
            "complexity_level": "medium",
            "testing_focus": [],
            "frameworks_recommended": []
        }
        
        return self._generate_testing_implementation("Create tests", code_content, analysis, None)
    
    def generate_security_tests(self, code_content: str) -> List[TestCase]:
        """Public method to generate security tests only."""
        analysis = {
            "language": self._detect_language(code_content),
            "test_types_needed": ["security"]
        }
        return self._generate_security_tests(code_content, analysis)