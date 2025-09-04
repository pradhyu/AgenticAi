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
        required_capabilities = [Capability.TEST_CREATION, Capability.QUALITY_ASSURANCE]
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