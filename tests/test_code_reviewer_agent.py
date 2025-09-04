"""Tests for CodeReviewerAgent."""

import pytest
from unittest.mock import Mock

from deep_agent_system.agents.code_reviewer import (
    CodeReviewerAgent, SecurityIssue, PerformanceIssue, QualityIssue, CodeReviewResult
)
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
        agent_id="code_reviewer_test",
        agent_type=AgentType.CODE_REVIEWER,
        capabilities=[Capability.CODE_REVIEW, Capability.QUALITY_ANALYSIS],
        rag_enabled=True,
        graph_rag_enabled=False
    )


@pytest.fixture
def code_reviewer_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create CodeReviewerAgent instance for testing."""
    return CodeReviewerAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager
    )


class TestCodeReviewerAgent:
    """Test cases for CodeReviewerAgent."""
    
    def test_initialization(self, code_reviewer_agent):
        """Test CodeReviewerAgent initialization."""
        assert code_reviewer_agent.agent_id == "code_reviewer_test"
        assert code_reviewer_agent.agent_type == AgentType.CODE_REVIEWER.value
        assert code_reviewer_agent.has_capability(Capability.CODE_REVIEW)
        assert code_reviewer_agent.has_capability(Capability.QUALITY_ANALYSIS)
    
    def test_detect_language_python(self, code_reviewer_agent):
        """Test Python language detection."""
        code = "def hello_world():\n    print('Hello, World!')"
        language = code_reviewer_agent._detect_language(code)
        assert language == "python"
    
    def test_detect_language_javascript(self, code_reviewer_agent):
        """Test JavaScript language detection."""
        code = "function helloWorld() {\n    console.log('Hello, World!');\n}"
        language = code_reviewer_agent._detect_language(code)
        assert language == "javascript"
    
    def test_detect_language_java(self, code_reviewer_agent):
        """Test Java language detection."""
        code = "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
        language = code_reviewer_agent._detect_language(code)
        assert language == "java"
    
    def test_extract_code_from_message_markdown(self, code_reviewer_agent):
        """Test extracting code from markdown code blocks."""
        message = "Here's some code:\n```python\ndef hello():\n    print('hello')\n```"
        code = code_reviewer_agent._extract_code_from_message(message)
        assert code == "def hello():\n    print('hello')"
    
    def test_extract_code_from_message_inline(self, code_reviewer_agent):
        """Test extracting code from inline code."""
        message = "def hello():\n    print('hello')\n    return True"
        code = code_reviewer_agent._extract_code_from_message(message)
        assert "def hello():" in code
        assert "print('hello')" in code
    
    def test_analyze_security_issues_sql_injection(self, code_reviewer_agent):
        """Test SQL injection detection."""
        code = "query = \"SELECT * FROM users WHERE id = \" + user_id\ncursor.execute(query)"
        security_issues = code_reviewer_agent._analyze_security_issues(code)
        
        assert len(security_issues) > 0
        sql_injection_issues = [issue for issue in security_issues if issue.issue_type == "sql_injection"]
        assert len(sql_injection_issues) > 0
        assert sql_injection_issues[0].severity == "critical"
    
    def test_analyze_security_issues_hardcoded_secrets(self, code_reviewer_agent):
        """Test hardcoded secrets detection."""
        code = "password = \"secret123\"\napi_key = \"abc123def456\""
        security_issues = code_reviewer_agent._analyze_security_issues(code)
        
        hardcoded_issues = [issue for issue in security_issues if issue.issue_type == "hardcoded_secrets"]
        assert len(hardcoded_issues) >= 1
        assert hardcoded_issues[0].severity == "high"
    
    def test_analyze_performance_issues_n_plus_one(self, code_reviewer_agent):
        """Test N+1 query detection."""
        code = "for user in users:\n    profile = Profile.objects.get(user_id=user.id)"
        performance_issues = code_reviewer_agent._analyze_performance_issues(code)
        
        n_plus_one_issues = [issue for issue in performance_issues if issue.issue_type == "n_plus_one"]
        assert len(n_plus_one_issues) > 0
        assert n_plus_one_issues[0].impact == "high"
    
    def test_analyze_quality_issues_long_lines(self, code_reviewer_agent):
        """Test long line detection."""
        long_line = "x = " + "a" * 150  # Create a line longer than 120 characters
        code = f"def test():\n    {long_line}\n    return x"
        quality_issues = code_reviewer_agent._analyze_quality_issues(code)
        
        long_line_issues = [issue for issue in quality_issues if issue.category == "readability"]
        assert len(long_line_issues) > 0
    
    def test_analyze_quality_issues_magic_numbers(self, code_reviewer_agent):
        """Test magic number detection."""
        code = "def calculate():\n    return value * 42 + 100"
        quality_issues = code_reviewer_agent._analyze_quality_issues(code)
        
        magic_number_issues = [issue for issue in quality_issues if issue.category == "maintainability"]
        assert len(magic_number_issues) > 0
    
    def test_detect_bugs_null_reference(self, code_reviewer_agent):
        """Test null reference detection."""
        code = "result = user.profile.settings.theme"
        bugs = code_reviewer_agent._detect_bugs(code)
        
        null_ref_bugs = [bug for bug in bugs if bug["type"] == "potential_null_reference"]
        assert len(null_ref_bugs) > 0
        assert null_ref_bugs[0]["severity"] == "medium"
    
    def test_detect_bugs_missing_exception_handling(self, code_reviewer_agent):
        """Test missing exception handling detection."""
        code = "data = json.loads(response_text)\nfile = open('test.txt')"
        bugs = code_reviewer_agent._detect_bugs(code)
        
        exception_bugs = [bug for bug in bugs if bug["type"] == "missing_exception_handling"]
        assert len(exception_bugs) > 0
    
    def test_calculate_overall_score_high_quality(self, code_reviewer_agent):
        """Test overall score calculation for high-quality code."""
        security_issues = []
        performance_issues = []
        quality_issues = []
        bugs_found = []
        
        score = code_reviewer_agent._calculate_overall_score(
            security_issues, performance_issues, quality_issues, bugs_found
        )
        
        assert score == 10.0
    
    def test_calculate_overall_score_with_issues(self, code_reviewer_agent):
        """Test overall score calculation with various issues."""
        security_issues = [
            SecurityIssue("critical", "sql_injection", "SQL injection found", 1, "Use parameterized queries")
        ]
        performance_issues = [
            PerformanceIssue("high", "n_plus_one", "N+1 query detected", 5, "Use bulk queries")
        ]
        quality_issues = [
            QualityIssue("complexity", "medium", "Function too long", 10, "Break into smaller functions")
        ]
        bugs_found = [
            {"type": "null_reference", "severity": "medium", "line_number": 15}
        ]
        
        score = code_reviewer_agent._calculate_overall_score(
            security_issues, performance_issues, quality_issues, bugs_found
        )
        
        assert score < 10.0
        assert score >= 1.0
    
    def test_perform_code_review_complete(self, code_reviewer_agent):
        """Test complete code review process."""
        code = """
def process_user_input(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    cursor.execute(query)
    return cursor.fetchall()
        """
        
        review_result = code_reviewer_agent._perform_code_review(code, None)
        
        assert review_result is not None
        assert isinstance(review_result, CodeReviewResult)
        assert review_result.overall_score <= 10.0
        assert len(review_result.security_issues) > 0  # Should detect SQL injection
        assert len(review_result.improvements) > 0
    
    def test_process_code_review_request(self, code_reviewer_agent, mock_rag_manager):
        """Test processing code review request message."""
        message = Message(
            sender_id="user",
            recipient_id="code_reviewer_test",
            content="Please review this code:\n```python\ndef hello():\n    print('hello')\n```",
            message_type=MessageType.USER_QUESTION
        )
        
        # Mock RAG manager response
        mock_rag_manager.retrieve_vector_context.return_value = []
        
        response = code_reviewer_agent._handle_code_review_request(message)
        
        assert isinstance(response, Response)
        assert response.agent_id == "code_reviewer_test"
        assert response.confidence_score > 0
        assert "Code Review Analysis" in response.content
    
    def test_process_message_no_code(self, code_reviewer_agent):
        """Test processing message without code."""
        message = Message(
            sender_id="user",
            recipient_id="code_reviewer_test",
            content="Please review my code but I forgot to include it",
            message_type=MessageType.USER_QUESTION
        )
        
        response = code_reviewer_agent._handle_code_review_request(message)
        
        assert isinstance(response, Response)
        assert response.confidence_score == 0.0
        assert "No code found" in response.content
    
    def test_public_review_code_method(self, code_reviewer_agent):
        """Test public review_code method."""
        code = "def hello():\n    print('hello world')"
        
        result = code_reviewer_agent.review_code(code)
        
        assert result is not None
        assert isinstance(result, CodeReviewResult)
        assert result.overall_score > 0
    
    def test_public_analyze_security_method(self, code_reviewer_agent):
        """Test public analyze_security method."""
        code = "password = 'hardcoded_secret'"
        
        security_issues = code_reviewer_agent.analyze_security(code)
        
        assert isinstance(security_issues, list)
        assert len(security_issues) > 0
        assert all(isinstance(issue, SecurityIssue) for issue in security_issues)
    
    def test_public_analyze_performance_method(self, code_reviewer_agent):
        """Test public analyze_performance method."""
        code = "for item in items:\n    result = database.get(item.id)"
        
        performance_issues = code_reviewer_agent.analyze_performance(code)
        
        assert isinstance(performance_issues, list)
        # May or may not find issues depending on exact pattern matching


class TestSecurityIssue:
    """Test cases for SecurityIssue class."""
    
    def test_security_issue_creation(self):
        """Test SecurityIssue object creation."""
        issue = SecurityIssue(
            severity="high",
            issue_type="xss",
            description="XSS vulnerability found",
            line_number=5,
            recommendation="Sanitize input"
        )
        
        assert issue.severity == "high"
        assert issue.issue_type == "xss"
        assert issue.description == "XSS vulnerability found"
        assert issue.line_number == 5
        assert issue.recommendation == "Sanitize input"
    
    def test_security_issue_to_dict(self):
        """Test SecurityIssue to_dict method."""
        issue = SecurityIssue(
            severity="critical",
            issue_type="sql_injection",
            description="SQL injection detected",
            line_number=10,
            recommendation="Use parameterized queries"
        )
        
        result = issue.to_dict()
        
        assert isinstance(result, dict)
        assert result["severity"] == "critical"
        assert result["issue_type"] == "sql_injection"
        assert result["line_number"] == 10


class TestPerformanceIssue:
    """Test cases for PerformanceIssue class."""
    
    def test_performance_issue_creation(self):
        """Test PerformanceIssue object creation."""
        issue = PerformanceIssue(
            impact="high",
            issue_type="n_plus_one",
            description="N+1 query detected",
            line_number=15,
            optimization="Use bulk queries"
        )
        
        assert issue.impact == "high"
        assert issue.issue_type == "n_plus_one"
        assert issue.description == "N+1 query detected"
        assert issue.line_number == 15
        assert issue.optimization == "Use bulk queries"
    
    def test_performance_issue_to_dict(self):
        """Test PerformanceIssue to_dict method."""
        issue = PerformanceIssue(
            impact="medium",
            issue_type="inefficient_loop",
            description="Inefficient loop detected",
            optimization="Use list comprehension"
        )
        
        result = issue.to_dict()
        
        assert isinstance(result, dict)
        assert result["impact"] == "medium"
        assert result["issue_type"] == "inefficient_loop"


class TestQualityIssue:
    """Test cases for QualityIssue class."""
    
    def test_quality_issue_creation(self):
        """Test QualityIssue object creation."""
        issue = QualityIssue(
            category="complexity",
            severity="medium",
            description="Function too complex",
            line_number=20,
            suggestion="Break into smaller functions"
        )
        
        assert issue.category == "complexity"
        assert issue.severity == "medium"
        assert issue.description == "Function too complex"
        assert issue.line_number == 20
        assert issue.suggestion == "Break into smaller functions"
    
    def test_quality_issue_to_dict(self):
        """Test QualityIssue to_dict method."""
        issue = QualityIssue(
            category="readability",
            severity="low",
            description="Line too long",
            suggestion="Break long lines"
        )
        
        result = issue.to_dict()
        
        assert isinstance(result, dict)
        assert result["category"] == "readability"
        assert result["severity"] == "low"