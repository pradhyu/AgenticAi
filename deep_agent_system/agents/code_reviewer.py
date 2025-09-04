"""Code Reviewer Agent for code analysis and improvement recommendations."""

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


class SecurityIssue:
    """Represents a security issue found in code."""
    
    def __init__(
        self,
        severity: str,
        issue_type: str,
        description: str,
        line_number: Optional[int] = None,
        recommendation: str = ""
    ):
        self.severity = severity  # "critical", "high", "medium", "low"
        self.issue_type = issue_type
        self.description = description
        self.line_number = line_number
        self.recommendation = recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "severity": self.severity,
            "issue_type": self.issue_type,
            "description": self.description,
            "line_number": self.line_number,
            "recommendation": self.recommendation,
        }


class PerformanceIssue:
    """Represents a performance issue found in code."""
    
    def __init__(
        self,
        impact: str,
        issue_type: str,
        description: str,
        line_number: Optional[int] = None,
        optimization: str = ""
    ):
        self.impact = impact  # "high", "medium", "low"
        self.issue_type = issue_type
        self.description = description
        self.line_number = line_number
        self.optimization = optimization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "impact": self.impact,
            "issue_type": self.issue_type,
            "description": self.description,
            "line_number": self.line_number,
            "optimization": self.optimization,
        }


class QualityIssue:
    """Represents a code quality issue."""
    
    def __init__(
        self,
        category: str,
        severity: str,
        description: str,
        line_number: Optional[int] = None,
        suggestion: str = ""
    ):
        self.category = category  # "maintainability", "readability", "complexity", "style"
        self.severity = severity  # "high", "medium", "low"
        self.description = description
        self.line_number = line_number
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "line_number": self.line_number,
            "suggestion": self.suggestion,
        }


class CodeReviewResult:
    """Complete code review analysis result."""
    
    def __init__(
        self,
        overall_score: float,
        summary: str,
        security_issues: List[SecurityIssue],
        performance_issues: List[PerformanceIssue],
        quality_issues: List[QualityIssue],
        bugs_found: List[Dict[str, Any]],
        improvements: List[Dict[str, Any]],
        positive_aspects: List[str]
    ):
        self.overall_score = overall_score
        self.summary = summary
        self.security_issues = security_issues
        self.performance_issues = performance_issues
        self.quality_issues = quality_issues
        self.bugs_found = bugs_found
        self.improvements = improvements
        self.positive_aspects = positive_aspects
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "summary": self.summary,
            "security_issues": [issue.to_dict() for issue in self.security_issues],
            "performance_issues": [issue.to_dict() for issue in self.performance_issues],
            "quality_issues": [issue.to_dict() for issue in self.quality_issues],
            "bugs_found": self.bugs_found,
            "improvements": self.improvements,
            "positive_aspects": self.positive_aspects,
        }


class CodeReviewerAgent(BaseAgent):
    """Code Reviewer agent responsible for code analysis and improvement recommendations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Code Reviewer agent."""
        super().__init__(*args, **kwargs)
        
        # Validate that this agent has the required capabilities
        required_capabilities = [Capability.CODE_REVIEW]
        for capability in required_capabilities:
            if not self.has_capability(capability):
                logger.warning(
                    f"CodeReviewerAgent {self.agent_id} missing required capability: {capability}"
                )
        
        # Security vulnerability patterns
        self._security_patterns = {
            "sql_injection": {
                "patterns": [r"SELECT.*\+.*", r"INSERT.*\+.*", r"UPDATE.*\+.*", r"DELETE.*\+.*"],
                "description": "Potential SQL injection vulnerability",
                "severity": "critical"
            },
            "xss": {
                "patterns": [r"innerHTML\s*=", r"document\.write\(", r"eval\("],
                "description": "Potential XSS vulnerability",
                "severity": "high"
            },
            "hardcoded_secrets": {
                "patterns": [r"password\s*=\s*['\"][^'\"]+['\"]", r"api_key\s*=\s*['\"][^'\"]+['\"]", 
                           r"secret\s*=\s*['\"][^'\"]+['\"]"],
                "description": "Hardcoded secrets or credentials",
                "severity": "high"
            },
            "unsafe_deserialization": {
                "patterns": [r"pickle\.loads?", r"yaml\.load\(", r"eval\("],
                "description": "Unsafe deserialization",
                "severity": "high"
            }
        }
        
        # Performance anti-patterns
        self._performance_patterns = {
            "n_plus_one": {
                "patterns": [r"for.*in.*:\s*.*\.get\(", r"for.*in.*:\s*.*\.filter\("],
                "description": "Potential N+1 query problem",
                "impact": "high"
            },
            "inefficient_loops": {
                "patterns": [r"for.*in.*:\s*.*\.append\(.*for.*in"],
                "description": "Nested loops that could be optimized",
                "impact": "medium"
            },
            "memory_leaks": {
                "patterns": [r"global\s+\w+\s*=\s*\[\]", r"cache\s*=\s*\{\}"],
                "description": "Potential memory leak with global variables",
                "impact": "medium"
            }
        }
        
        # Code quality indicators
        self._quality_indicators = {
            "long_functions": {"threshold": 50, "category": "complexity"},
            "deep_nesting": {"threshold": 4, "category": "complexity"},
            "long_lines": {"threshold": 120, "category": "readability"},
            "magic_numbers": {"pattern": r"\b\d{2,}\b", "category": "maintainability"},
            "todo_comments": {"pattern": r"#\s*TODO|#\s*FIXME|#\s*HACK", "category": "maintainability"}
        }
        
        logger.info(f"CodeReviewerAgent {self.agent_id} initialized successfully")
    
    def _process_message_impl(self, message: Message) -> Response:
        """Process incoming messages for code review."""
        try:
            if message.message_type in [MessageType.USER_QUESTION, MessageType.AGENT_REQUEST]:
                return self._handle_code_review_request(message)
            else:
                return self._create_error_response(
                    message.id,
                    f"Unsupported message type: {message.message_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message in CodeReviewerAgent: {e}")
            return self._create_error_response(message.id, str(e))
    
    def _handle_code_review_request(self, message: Message) -> Response:
        """Handle code review requests."""
        question = message.content
        
        # Extract code from the message
        code_content = self._extract_code_from_message(question)
        
        if not code_content:
            return self._create_error_response(
                message.id,
                "No code found in the message. Please provide code to review."
            )
        
        # Step 1: Retrieve relevant best practices and guidelines using RAG
        context = self._retrieve_review_context(question, code_content)
        
        # Step 2: Perform comprehensive code analysis
        review_result = self._perform_code_review(code_content, context)
        
        # Step 3: Create comprehensive response
        response_content = self._format_review_response(
            question, code_content, review_result, context
        )
        
        # Calculate confidence based on code complexity and context
        confidence = self._calculate_confidence(code_content, context, review_result)
        
        context_used = []
        if context and context.documents:
            context_used.extend([f"best_practice_{i}" for i in range(len(context.documents))])
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            confidence_score=confidence,
            context_used=context_used,
            metadata={
                "review_result": review_result.to_dict() if review_result else None,
                "code_review_domain": True,
                "code_length": len(code_content.split('\n')) if code_content else 0,
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
            # Check if line looks like code (contains common programming constructs)
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'import ', 'from ', 
                                                  'if ', 'for ', 'while ', 'try:', 'except:', 
                                                  'public ', 'private ', 'protected ', 'const ', 'let ', 'var ']):
                code_lines.append(line)
            elif code_lines and (line.startswith('    ') or line.startswith('\t')):
                # Indented line following code
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else None 
   
    def _retrieve_review_context(self, question: str, code_content: str) -> Optional[Context]:
        """Retrieve context with emphasis on best practices and guidelines."""
        if not self.rag_manager:
            return None
        
        try:
            # Create a query combining the question and code analysis
            query = f"code review best practices {question} {self._detect_language(code_content)}"
            
            # Use vector search for best practices and guidelines
            return self.retrieve_context(query, RetrievalType.VECTOR, k=5)
            
        except Exception as e:
            logger.error(f"Error retrieving review context: {e}")
            return None
    
    def _detect_language(self, code_content: str) -> str:
        """Detect programming language from code content."""
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
        
        # Go indicators
        elif any(keyword in code_lower for keyword in ['func ', 'package ', 'import ', 'go ']):
            return "go"
        
        # Default to generic
        return "generic"
    
    def _perform_code_review(self, code_content: str, context: Optional[Context]) -> Optional[CodeReviewResult]:
        """Perform comprehensive code analysis and review."""
        try:
            # Analyze security issues
            security_issues = self._analyze_security_issues(code_content)
            
            # Analyze performance issues
            performance_issues = self._analyze_performance_issues(code_content)
            
            # Analyze code quality issues
            quality_issues = self._analyze_quality_issues(code_content)
            
            # Detect potential bugs
            bugs_found = self._detect_bugs(code_content)
            
            # Generate improvement recommendations
            improvements = self._generate_improvements(code_content, security_issues, performance_issues, quality_issues)
            
            # Identify positive aspects
            positive_aspects = self._identify_positive_aspects(code_content)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                security_issues, performance_issues, quality_issues, bugs_found
            )
            
            # Generate summary
            summary = self._generate_summary(
                overall_score, security_issues, performance_issues, quality_issues, bugs_found
            )
            
            return CodeReviewResult(
                overall_score=overall_score,
                summary=summary,
                security_issues=security_issues,
                performance_issues=performance_issues,
                quality_issues=quality_issues,
                bugs_found=bugs_found,
                improvements=improvements,
                positive_aspects=positive_aspects
            )
            
        except Exception as e:
            logger.error(f"Error performing code review: {e}")
            return None
    
    def _analyze_security_issues(self, code_content: str) -> List[SecurityIssue]:
        """Analyze code for security vulnerabilities."""
        security_issues = []
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern_info in self._security_patterns.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, line, re.IGNORECASE):
                        security_issues.append(SecurityIssue(
                            severity=pattern_info["severity"],
                            issue_type=issue_type,
                            description=pattern_info["description"],
                            line_number=line_num,
                            recommendation=self._get_security_recommendation(issue_type)
                        ))
        
        return security_issues
    
    def _get_security_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for specific issue type."""
        recommendations = {
            "sql_injection": "Use parameterized queries or ORM methods instead of string concatenation",
            "xss": "Sanitize user input and use safe DOM manipulation methods",
            "hardcoded_secrets": "Use environment variables or secure configuration management",
            "unsafe_deserialization": "Validate input and use safe deserialization methods"
        }
        return recommendations.get(issue_type, "Review and apply security best practices")
    
    def _analyze_performance_issues(self, code_content: str) -> List[PerformanceIssue]:
        """Analyze code for performance issues."""
        performance_issues = []
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern_info in self._performance_patterns.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, line, re.IGNORECASE):
                        performance_issues.append(PerformanceIssue(
                            impact=pattern_info["impact"],
                            issue_type=issue_type,
                            description=pattern_info["description"],
                            line_number=line_num,
                            optimization=self._get_performance_optimization(issue_type)
                        ))
        
        return performance_issues
    
    def _get_performance_optimization(self, issue_type: str) -> str:
        """Get performance optimization recommendation for specific issue type."""
        optimizations = {
            "n_plus_one": "Use bulk queries, select_related, or prefetch_related to reduce database calls",
            "inefficient_loops": "Consider using list comprehensions, map(), or vectorized operations",
            "memory_leaks": "Use local variables or implement proper cleanup mechanisms"
        }
        return optimizations.get(issue_type, "Review algorithm complexity and optimize data structures")
    
    def _analyze_quality_issues(self, code_content: str) -> List[QualityIssue]:
        """Analyze code for quality issues."""
        quality_issues = []
        lines = code_content.split('\n')
        
        # Check for long functions
        current_function_length = 0
        function_start_line = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Function detection (simplified)
            if any(keyword in stripped_line for keyword in ['def ', 'function ', 'public ', 'private ']):
                if current_function_length > self._quality_indicators["long_functions"]["threshold"]:
                    quality_issues.append(QualityIssue(
                        category="complexity",
                        severity="medium",
                        description=f"Function is too long ({current_function_length} lines)",
                        line_number=function_start_line,
                        suggestion="Consider breaking this function into smaller, more focused functions"
                    ))
                current_function_length = 1
                function_start_line = line_num
            elif stripped_line and not stripped_line.startswith('#'):
                current_function_length += 1
            
            # Check line length
            if len(line) > self._quality_indicators["long_lines"]["threshold"]:
                quality_issues.append(QualityIssue(
                    category="readability",
                    severity="low",
                    description=f"Line too long ({len(line)} characters)",
                    line_number=line_num,
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for magic numbers
            magic_pattern = self._quality_indicators["magic_numbers"]["pattern"]
            if re.search(magic_pattern, line):
                quality_issues.append(QualityIssue(
                    category="maintainability",
                    severity="low",
                    description="Magic number found",
                    line_number=line_num,
                    suggestion="Replace magic numbers with named constants"
                ))
            
            # Check for TODO comments
            todo_pattern = self._quality_indicators["todo_comments"]["pattern"]
            if re.search(todo_pattern, line, re.IGNORECASE):
                quality_issues.append(QualityIssue(
                    category="maintainability",
                    severity="low",
                    description="TODO/FIXME comment found",
                    line_number=line_num,
                    suggestion="Address TODO items or create proper issue tracking"
                ))
            
            # Check nesting depth
            indent_level = len(line) - len(line.lstrip())
            if indent_level > self._quality_indicators["deep_nesting"]["threshold"] * 4:  # Assuming 4 spaces per level
                quality_issues.append(QualityIssue(
                    category="complexity",
                    severity="medium",
                    description="Deep nesting detected",
                    line_number=line_num,
                    suggestion="Reduce nesting by using early returns or extracting methods"
                ))
        
        return quality_issues    

    def _detect_bugs(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect potential bugs in the code."""
        bugs = []
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Potential null pointer/attribute errors
            if re.search(r'\w+\.\w+\.\w+', stripped_line) and 'if' not in stripped_line:
                bugs.append({
                    "type": "potential_null_reference",
                    "severity": "medium",
                    "description": "Potential null reference without null check",
                    "line_number": line_num,
                    "suggestion": "Add null checks before accessing nested properties"
                })
            
            # Unreachable code after return
            if 'return' in stripped_line and line_num < len(lines):
                next_line = lines[line_num].strip() if line_num < len(lines) else ""
                if next_line and not next_line.startswith('#') and not any(keyword in next_line for keyword in ['def ', 'class ', 'if ', 'else:', 'elif ', 'except:', 'finally:']):
                    bugs.append({
                        "type": "unreachable_code",
                        "severity": "low",
                        "description": "Potential unreachable code after return statement",
                        "line_number": line_num + 1,
                        "suggestion": "Remove unreachable code or restructure logic"
                    })
            
            # Missing exception handling
            if any(risky in stripped_line for risky in ['open(', 'requests.get', 'json.loads', 'int(', 'float(']):
                # Check if there's a try block nearby
                context_lines = lines[max(0, line_num-3):min(len(lines), line_num+3)]
                if not any('try:' in context_line for context_line in context_lines):
                    bugs.append({
                        "type": "missing_exception_handling",
                        "severity": "medium",
                        "description": "Risky operation without exception handling",
                        "line_number": line_num,
                        "suggestion": "Wrap risky operations in try-except blocks"
                    })
            
            # Potential division by zero
            if re.search(r'/\s*\w+', stripped_line) and 'if' not in stripped_line:
                bugs.append({
                    "type": "potential_division_by_zero",
                    "severity": "medium",
                    "description": "Potential division by zero",
                    "line_number": line_num,
                    "suggestion": "Add check to ensure divisor is not zero"
                })
        
        return bugs
    
    def _generate_improvements(
        self,
        code_content: str,
        security_issues: List[SecurityIssue],
        performance_issues: List[PerformanceIssue],
        quality_issues: List[QualityIssue]
    ) -> List[Dict[str, Any]]:
        """Generate improvement recommendations."""
        improvements = []
        
        # Security improvements
        if security_issues:
            critical_security = [issue for issue in security_issues if issue.severity == "critical"]
            if critical_security:
                improvements.append({
                    "category": "security",
                    "priority": "critical",
                    "title": "Address Critical Security Vulnerabilities",
                    "description": "Fix critical security issues that could lead to data breaches",
                    "actions": [issue.recommendation for issue in critical_security]
                })
        
        # Performance improvements
        if performance_issues:
            high_impact = [issue for issue in performance_issues if issue.impact == "high"]
            if high_impact:
                improvements.append({
                    "category": "performance",
                    "priority": "high",
                    "title": "Optimize High-Impact Performance Issues",
                    "description": "Address performance bottlenecks that significantly impact user experience",
                    "actions": [issue.optimization for issue in high_impact]
                })
        
        # Code quality improvements
        complexity_issues = [issue for issue in quality_issues if issue.category == "complexity"]
        if len(complexity_issues) > 3:
            improvements.append({
                "category": "quality",
                "priority": "medium",
                "title": "Reduce Code Complexity",
                "description": "Simplify complex code structures for better maintainability",
                "actions": [
                    "Break down large functions into smaller, focused methods",
                    "Reduce nesting levels using early returns",
                    "Extract complex logic into separate functions"
                ]
            })
        
        # General improvements
        improvements.append({
            "category": "general",
            "priority": "low",
            "title": "Add Comprehensive Documentation",
            "description": "Improve code documentation and comments",
            "actions": [
                "Add docstrings to all functions and classes",
                "Include type hints for better code clarity",
                "Add inline comments for complex logic"
            ]
        })
        
        improvements.append({
            "category": "testing",
            "priority": "medium",
            "title": "Improve Test Coverage",
            "description": "Add comprehensive tests for better code reliability",
            "actions": [
                "Write unit tests for all public methods",
                "Add integration tests for critical workflows",
                "Include edge case testing"
            ]
        })
        
        return improvements
    
    def _identify_positive_aspects(self, code_content: str) -> List[str]:
        """Identify positive aspects of the code."""
        positive_aspects = []
        lines = code_content.split('\n')
        
        # Check for good practices
        has_docstrings = any('"""' in line or "'''" in line for line in lines)
        if has_docstrings:
            positive_aspects.append("Good use of docstrings for documentation")
        
        has_type_hints = any(':' in line and '->' in line for line in lines)
        if has_type_hints:
            positive_aspects.append("Proper use of type hints for better code clarity")
        
        has_error_handling = any('try:' in line or 'except' in line for line in lines)
        if has_error_handling:
            positive_aspects.append("Includes error handling with try-except blocks")
        
        has_logging = any('log' in line.lower() for line in lines)
        if has_logging:
            positive_aspects.append("Implements logging for better debugging and monitoring")
        
        # Check for good naming conventions
        good_naming = True
        for line in lines:
            # Check for single letter variables (except common ones like i, j, x, y)
            if re.search(r'\b[a-z]\s*=', line) and not re.search(r'\b[ijxy]\s*=', line):
                good_naming = False
                break
        
        if good_naming:
            positive_aspects.append("Uses descriptive variable and function names")
        
        # Check for modular structure
        has_functions = any('def ' in line for line in lines)
        has_classes = any('class ' in line for line in lines)
        if has_functions or has_classes:
            positive_aspects.append("Well-structured code with proper function/class organization")
        
        # Default positive aspects if none found
        if not positive_aspects:
            positive_aspects.append("Code follows basic syntax and structure correctly")
        
        return positive_aspects
    
    def _calculate_overall_score(
        self,
        security_issues: List[SecurityIssue],
        performance_issues: List[PerformanceIssue],
        quality_issues: List[QualityIssue],
        bugs_found: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall code quality score (1-10)."""
        base_score = 10.0
        
        # Deduct for security issues
        for issue in security_issues:
            if issue.severity == "critical":
                base_score -= 2.0
            elif issue.severity == "high":
                base_score -= 1.5
            elif issue.severity == "medium":
                base_score -= 1.0
            else:
                base_score -= 0.5
        
        # Deduct for performance issues
        for issue in performance_issues:
            if issue.impact == "high":
                base_score -= 1.0
            elif issue.impact == "medium":
                base_score -= 0.7
            else:
                base_score -= 0.3
        
        # Deduct for quality issues
        for issue in quality_issues:
            if issue.severity == "high":
                base_score -= 0.8
            elif issue.severity == "medium":
                base_score -= 0.5
            else:
                base_score -= 0.2
        
        # Deduct for bugs
        for bug in bugs_found:
            if bug["severity"] == "high":
                base_score -= 1.2
            elif bug["severity"] == "medium":
                base_score -= 0.8
            else:
                base_score -= 0.4
        
        return max(1.0, min(10.0, base_score))
    
    def _generate_summary(
        self,
        overall_score: float,
        security_issues: List[SecurityIssue],
        performance_issues: List[PerformanceIssue],
        quality_issues: List[QualityIssue],
        bugs_found: List[Dict[str, Any]]
    ) -> str:
        """Generate a summary of the code review."""
        summary = f"Code Quality Score: {overall_score:.1f}/10\n\n"
        
        if overall_score >= 8.0:
            summary += "Excellent code quality with minimal issues. "
        elif overall_score >= 6.0:
            summary += "Good code quality with some areas for improvement. "
        elif overall_score >= 4.0:
            summary += "Moderate code quality with several issues to address. "
        else:
            summary += "Poor code quality requiring significant improvements. "
        
        # Issue counts
        total_issues = len(security_issues) + len(performance_issues) + len(quality_issues) + len(bugs_found)
        summary += f"Found {total_issues} total issues: "
        
        issue_counts = []
        if security_issues:
            critical_security = len([i for i in security_issues if i.severity == "critical"])
            if critical_security:
                issue_counts.append(f"{critical_security} critical security")
            else:
                issue_counts.append(f"{len(security_issues)} security")
        
        if performance_issues:
            issue_counts.append(f"{len(performance_issues)} performance")
        
        if quality_issues:
            issue_counts.append(f"{len(quality_issues)} quality")
        
        if bugs_found:
            issue_counts.append(f"{len(bugs_found)} potential bugs")
        
        summary += ", ".join(issue_counts) + "."
        
        return summary  
  
    def _format_review_response(
        self,
        question: str,
        code_content: str,
        review_result: Optional[CodeReviewResult],
        context: Optional[Context]
    ) -> str:
        """Format the comprehensive code review response."""
        response = "# Code Review Analysis\n\n"
        
        if not review_result:
            return response + "Unable to perform code review analysis."
        
        # Overall Assessment Section
        response += "## 1. Overall Assessment\n\n"
        response += f"**Quality Score:** {review_result.overall_score:.1f}/10\n\n"
        response += f"**Summary:** {review_result.summary}\n\n"
        
        # Security Analysis Section
        response += "## 2. Security Analysis\n\n"
        if review_result.security_issues:
            response += "**Security Issues Found:**\n\n"
            for issue in review_result.security_issues:
                response += f"- **{issue.severity.upper()}**: {issue.description}"
                if issue.line_number:
                    response += f" (Line {issue.line_number})"
                response += f"\n  - *Recommendation*: {issue.recommendation}\n\n"
        else:
            response += "✅ No obvious security vulnerabilities detected.\n\n"
        
        # Performance Review Section
        response += "## 3. Performance Review\n\n"
        if review_result.performance_issues:
            response += "**Performance Issues Found:**\n\n"
            for issue in review_result.performance_issues:
                response += f"- **{issue.impact.upper()} IMPACT**: {issue.description}"
                if issue.line_number:
                    response += f" (Line {issue.line_number})"
                response += f"\n  - *Optimization*: {issue.optimization}\n\n"
        else:
            response += "✅ No significant performance issues detected.\n\n"
        
        # Code Quality Issues Section
        response += "## 4. Code Quality Issues\n\n"
        if review_result.quality_issues:
            # Group by category
            quality_by_category = {}
            for issue in review_result.quality_issues:
                if issue.category not in quality_by_category:
                    quality_by_category[issue.category] = []
                quality_by_category[issue.category].append(issue)
            
            for category, issues in quality_by_category.items():
                response += f"**{category.title()} Issues:**\n"
                for issue in issues:
                    response += f"- {issue.description}"
                    if issue.line_number:
                        response += f" (Line {issue.line_number})"
                    if issue.suggestion:
                        response += f"\n  - *Suggestion*: {issue.suggestion}"
                    response += "\n\n"
        else:
            response += "✅ Code quality meets basic standards.\n\n"
        
        # Bug Detection Section
        response += "## 5. Bug Detection\n\n"
        if review_result.bugs_found:
            response += "**Potential Bugs Found:**\n\n"
            for bug in review_result.bugs_found:
                response += f"- **{bug['severity'].upper()}**: {bug['description']}"
                if bug.get('line_number'):
                    response += f" (Line {bug['line_number']})"
                response += f"\n  - *Suggestion*: {bug['suggestion']}\n\n"
        else:
            response += "✅ No obvious bugs detected.\n\n"
        
        # Improvement Recommendations Section
        response += "## 6. Improvement Recommendations\n\n"
        if review_result.improvements:
            for improvement in review_result.improvements:
                response += f"### {improvement['title']} ({improvement['priority'].upper()} Priority)\n\n"
                response += f"{improvement['description']}\n\n"
                response += "**Actions:**\n"
                for action in improvement['actions']:
                    response += f"- {action}\n"
                response += "\n"
        
        # Positive Aspects Section
        response += "## 7. Positive Aspects\n\n"
        if review_result.positive_aspects:
            response += "**Strengths Identified:**\n\n"
            for aspect in review_result.positive_aspects:
                response += f"✅ {aspect}\n"
            response += "\n"
        
        # Code Examples Section (if improvements needed)
        if review_result.security_issues or review_result.performance_issues:
            response += "## 8. Code Improvement Examples\n\n"
            
            # Security improvement example
            if review_result.security_issues:
                response += "### Security Improvement Example\n\n"
                response += "**Before (Vulnerable):**\n"
                response += "```python\n"
                response += "# SQL injection vulnerability\n"
                response += "query = \"SELECT * FROM users WHERE id = \" + user_id\n"
                response += "cursor.execute(query)\n"
                response += "```\n\n"
                response += "**After (Secure):**\n"
                response += "```python\n"
                response += "# Parameterized query\n"
                response += "query = \"SELECT * FROM users WHERE id = %s\"\n"
                response += "cursor.execute(query, (user_id,))\n"
                response += "```\n\n"
            
            # Performance improvement example
            if review_result.performance_issues:
                response += "### Performance Improvement Example\n\n"
                response += "**Before (Inefficient):**\n"
                response += "```python\n"
                response += "# N+1 query problem\n"
                response += "for user in users:\n"
                response += "    profile = Profile.objects.get(user_id=user.id)\n"
                response += "```\n\n"
                response += "**After (Optimized):**\n"
                response += "```python\n"
                response += "# Bulk query with join\n"
                response += "users_with_profiles = User.objects.select_related('profile').all()\n"
                response += "```\n\n"
        
        # Next Steps Section
        response += "## 9. Next Steps\n\n"
        response += "**Immediate Actions (High Priority):**\n"
        
        critical_security = [i for i in review_result.security_issues if i.severity == "critical"]
        if critical_security:
            response += "1. **URGENT**: Fix critical security vulnerabilities immediately\n"
        
        high_impact_perf = [i for i in review_result.performance_issues if i.impact == "high"]
        if high_impact_perf:
            response += "2. Address high-impact performance issues\n"
        
        high_severity_bugs = [b for b in review_result.bugs_found if b["severity"] == "high"]
        if high_severity_bugs:
            response += "3. Fix high-severity potential bugs\n"
        
        response += "\n**Medium-Term Improvements:**\n"
        response += "1. Refactor complex functions and reduce code complexity\n"
        response += "2. Add comprehensive unit and integration tests\n"
        response += "3. Improve code documentation and comments\n"
        response += "4. Implement consistent error handling patterns\n\n"
        
        response += "**Long-Term Maintenance:**\n"
        response += "1. Set up automated code quality checks (linting, static analysis)\n"
        response += "2. Establish code review processes for future changes\n"
        response += "3. Monitor performance metrics in production\n"
        response += "4. Regular security audits and dependency updates\n\n"
        
        return response
    
    def _calculate_confidence(
        self,
        code_content: str,
        context: Optional[Context],
        review_result: Optional[CodeReviewResult]
    ) -> float:
        """Calculate confidence score for the code review."""
        confidence = 0.8  # Base confidence for code analysis
        
        # Adjust based on code length and complexity
        lines = len(code_content.split('\n'))
        if lines > 10:
            confidence += 0.1
        if lines > 50:
            confidence += 0.05
        
        # Adjust based on context availability
        if context and context.documents:
            confidence += 0.05
        
        # Adjust based on detected language
        language = self._detect_language(code_content)
        if language != "generic":
            confidence += 0.05
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _create_error_response(self, message_id: str, error_message: str) -> Response:
        """Create an error response."""
        return Response(
            message_id=message_id,
            agent_id=self.agent_id,
            content=f"I encountered an error while reviewing your code: {error_message}",
            confidence_score=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def review_code(
        self,
        code_content: str,
        language: Optional[str] = None
    ) -> Optional[CodeReviewResult]:
        """Public method to review code directly."""
        return self._perform_code_review(code_content, None)
    
    def analyze_security(self, code_content: str) -> List[SecurityIssue]:
        """Public method to analyze security issues only."""
        return self._analyze_security_issues(code_content)
    
    def analyze_performance(self, code_content: str) -> List[PerformanceIssue]:
        """Public method to analyze performance issues only."""
        return self._analyze_performance_issues(code_content)