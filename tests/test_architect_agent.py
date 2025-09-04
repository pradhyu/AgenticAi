"""Tests for the ArchitectAgent."""

import pytest
from unittest.mock import Mock, patch

from deep_agent_system.agents.architect import (
    ArchitectAgent,
    ArchitecturalAnalysis,
    ArchitecturalSolution,
)
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
from deep_agent_system.models.messages import (
    Context,
    Document,
    GraphData,
    Message,
    MessageType,
    RetrievalType,
)
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = Mock(spec=PromptManager)
    manager.get_prompt.return_value = """
You are an expert Architect Agent.
User Question: {user_question}
Context: {context}
Provide architectural guidance.
"""
    return manager


@pytest.fixture
def mock_rag_manager():
    """Create a mock RAG manager."""
    manager = Mock(spec=RAGManager)
    
    # Mock vector retrieval
    mock_docs = [
        Document(
            content="Microservices architecture patterns and best practices",
            metadata={"source": "architecture_guide.md"}
        ),
        Document(
            content="API design principles and REST conventions",
            metadata={"source": "api_design.md"}
        )
    ]
    
    mock_context = Context(
        query="test query",
        documents=mock_docs,
        retrieval_method=RetrievalType.VECTOR
    )
    
    manager.retrieve_vector_context.return_value = mock_docs
    manager.retrieve_graph_context.return_value = GraphData(
        nodes=[{"id": "service1", "type": "microservice"}],
        relationships=[{"from": "service1", "to": "database", "type": "uses"}]
    )
    manager.hybrid_retrieve.return_value = mock_context
    
    return manager


@pytest.fixture
def agent_config():
    """Create agent configuration for testing."""
    return AgentConfig(
        agent_id="architect_001",
        agent_type=AgentType.ARCHITECT,
        llm_config=LLMConfig(model_name="gpt-4"),
        capabilities=[
            Capability.ARCHITECTURE_DESIGN,
            Capability.SOLUTION_EXPLANATION,
            Capability.RAG_RETRIEVAL
        ],
        rag_enabled=True,
        graph_rag_enabled=True,
        prompt_templates={}
    )


@pytest.fixture
def architect_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create an ArchitectAgent instance for testing."""
    return ArchitectAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager
    )


class TestArchitectAgent:
    """Test cases for ArchitectAgent."""
    
    def test_initialization(self, architect_agent):
        """Test agent initialization."""
        assert architect_agent.agent_id == "architect_001"
        assert architect_agent.agent_type == AgentType.ARCHITECT.value
        assert architect_agent.has_capability(Capability.ARCHITECTURE_DESIGN)
        assert architect_agent.has_capability(Capability.SOLUTION_EXPLANATION)
    
    def test_process_user_question(self, architect_agent):
        """Test processing user questions."""
        message = Message(
            sender_id="user_001",
            recipient_id="architect_001",
            content="How should I design a microservices architecture for an e-commerce platform?",
            message_type=MessageType.USER_QUESTION
        )
        
        response = architect_agent.process_message(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "architect_001"
        assert response.confidence_score > 0.5
        assert "microservices" in response.content.lower()
        assert "architecture" in response.content.lower()
        assert response.metadata["architecture_domain"] is True
    
    def test_process_agent_request(self, architect_agent):
        """Test processing requests from other agents."""
        message = Message(
            sender_id="analyst_001",
            recipient_id="architect_001",
            content="Design a scalable API architecture for a social media platform",
            message_type=MessageType.AGENT_REQUEST,
            metadata={"routing_context": "Classified as architecture question"}
        )
        
        response = architect_agent.process_message(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "architect_001"
        assert response.confidence_score > 0.5
        assert "api" in response.content.lower()
        assert "scalable" in response.content.lower()
    
    def test_architectural_analysis(self, architect_agent):
        """Test architectural problem analysis."""
        question = "Design a microservices architecture with high availability"
        
        analysis = architect_agent.analyze_architecture(question)
        
        assert isinstance(analysis, ArchitecturalAnalysis)
        assert "microservices" in analysis.problem_understanding.lower()
        assert len(analysis.key_challenges) > 0
        assert len(analysis.constraints) > 0
        assert len(analysis.assumptions) > 0
        
        # Check for relevant challenges
        challenges_text = " ".join(analysis.key_challenges).lower()
        assert "distributed" in challenges_text or "scalability" in challenges_text
    
    def test_solution_generation(self, architect_agent):
        """Test architectural solution generation."""
        question = "Create an API-first architecture for a mobile application"
        
        solution = architect_agent.generate_solution(question)
        
        assert isinstance(solution, ArchitecturalSolution)
        assert "api" in solution.overview.lower()
        assert len(solution.components) > 0
        assert len(solution.technology_stack) > 0
        assert len(solution.patterns) > 0
        
        # Check for API-related components
        component_names = [comp["name"].lower() for comp in solution.components]
        assert any("api" in name or "gateway" in name for name in component_names)
    
    def test_technology_stack_recommendations(self, architect_agent):
        """Test technology stack recommendations."""
        # Test Python-specific question
        question = "Design a Python-based microservices architecture"
        solution = architect_agent.generate_solution(question)
        
        assert "python" in solution.technology_stack["backend"].lower()
        
        # Test Node.js-specific question
        question = "Create a Node.js API architecture"
        solution = architect_agent.generate_solution(question)
        
        assert "node" in solution.technology_stack["backend"].lower()
    
    def test_architectural_patterns_suggestion(self, architect_agent):
        """Test architectural patterns suggestion."""
        question = "Design an event-driven microservices system"
        solution = architect_agent.generate_solution(question)
        
        pattern_names = [pattern["pattern"].lower() for pattern in solution.patterns]
        assert any("microservices" in name or "event" in name for name in pattern_names)
    
    def test_context_retrieval_integration(self, architect_agent, mock_rag_manager):
        """Test integration with RAG context retrieval."""
        question = "How to design a scalable database architecture?"
        
        # Test with graph RAG enabled
        architect_agent.config.graph_rag_enabled = True
        response = architect_agent._retrieve_architectural_context(question)
        
        # Should prefer graph RAG for architectural relationships
        mock_rag_manager.retrieve_graph_context.assert_called()
        
        # Test with only vector RAG
        architect_agent.config.graph_rag_enabled = False
        mock_rag_manager.reset_mock()
        
        response = architect_agent._retrieve_architectural_context(question)
        mock_rag_manager.retrieve_vector_context.assert_called()
    
    def test_confidence_calculation(self, architect_agent):
        """Test confidence score calculation."""
        # Test with specific architectural terms
        specific_question = "Design microservices architecture with API gateway and database"
        message = Message(
            sender_id="user_001",
            recipient_id="architect_001",
            content=specific_question,
            message_type=MessageType.USER_QUESTION
        )
        
        response = architect_agent.process_message(message)
        high_confidence = response.confidence_score
        
        # Test with vague question
        vague_question = "Help me with something"
        message.content = vague_question
        
        response = architect_agent.process_message(message)
        low_confidence = response.confidence_score
        
        assert high_confidence > low_confidence
        assert high_confidence > 0.7
        assert low_confidence >= 0.5
    
    def test_component_identification(self, architect_agent):
        """Test system component identification."""
        # Test web API question
        question = "Design a web API with authentication"
        solution = architect_agent.generate_solution(question)
        
        component_names = [comp["name"] for comp in solution.components]
        assert any("API" in name for name in component_names)
        assert any("Authentication" in name for name in component_names)
        
        # Test messaging system question
        question = "Design a system with real-time notifications"
        solution = architect_agent.generate_solution(question)
        
        component_names = [comp["name"] for comp in solution.components]
        assert any("Messaging" in name or "Notification" in name for name in component_names)
    
    def test_diagram_generation(self, architect_agent):
        """Test architecture diagram generation."""
        question = "Design a microservices API architecture"
        solution = architect_agent.generate_solution(question)
        
        assert len(solution.diagrams) > 0
        
        # Check for Mermaid syntax
        for diagram in solution.diagrams:
            assert "```mermaid" in diagram
            assert "```" in diagram
    
    def test_response_formatting(self, architect_agent):
        """Test comprehensive response formatting."""
        message = Message(
            sender_id="user_001",
            recipient_id="architect_001",
            content="Design a scalable e-commerce platform architecture",
            message_type=MessageType.USER_QUESTION
        )
        
        response = architect_agent.process_message(message)
        
        # Check for required sections
        content = response.content
        assert "# Architectural Design Response" in content
        assert "## 1. Problem Analysis" in content
        assert "## 2. Proposed Solution" in content
        assert "## 3. Design Patterns" in content
        assert "## 5. Implementation Roadmap" in content
        
        # Check for structured content
        assert "**Understanding:**" in content
        assert "**Key Components:**" in content
        assert "**Technology Stack:**" in content
    
    def test_error_handling(self, architect_agent, mock_rag_manager):
        """Test error handling in message processing."""
        # Mock RAG manager to raise exception
        mock_rag_manager.retrieve_vector_context.side_effect = Exception("RAG error")
        
        message = Message(
            sender_id="user_001",
            recipient_id="architect_001",
            content="Design an architecture",
            message_type=MessageType.USER_QUESTION
        )
        
        response = architect_agent.process_message(message)
        
        # Should still return a response, not raise exception
        assert response.message_id == message.id
        assert response.confidence_score >= 0.0
        # Should still provide architectural guidance even without RAG
        assert len(response.content) > 0
    
    def test_unsupported_message_type(self, architect_agent):
        """Test handling of unsupported message types."""
        message = Message(
            sender_id="user_001",
            recipient_id="architect_001",
            content="Test message",
            message_type=MessageType.SYSTEM_MESSAGE
        )
        
        response = architect_agent.process_message(message)
        
        assert response.confidence_score == 0.0
        assert response.metadata["error"] is True
        assert "Unsupported message type" in response.content
    
    def test_rag_disabled_handling(self, architect_agent):
        """Test behavior when RAG is disabled."""
        architect_agent.config.rag_enabled = False
        
        question = "Design a microservices architecture"
        context = architect_agent._retrieve_architectural_context(question)
        
        assert context is None
        
        # Should still be able to generate solutions without RAG
        solution = architect_agent.generate_solution(question)
        assert isinstance(solution, ArchitecturalSolution)
        assert len(solution.components) > 0
    
    def test_public_methods(self, architect_agent):
        """Test public interface methods."""
        question = "Design a scalable web application"
        
        # Test analyze_architecture method
        analysis = architect_agent.analyze_architecture(question)
        assert isinstance(analysis, ArchitecturalAnalysis)
        
        # Test generate_solution method
        solution = architect_agent.generate_solution(question, analysis)
        assert isinstance(solution, ArchitecturalSolution)
        
        # Test generate_solution without analysis
        solution2 = architect_agent.generate_solution(question)
        assert isinstance(solution2, ArchitecturalSolution)


class TestArchitecturalAnalysis:
    """Test cases for ArchitecturalAnalysis class."""
    
    def test_initialization(self):
        """Test ArchitecturalAnalysis initialization."""
        analysis = ArchitecturalAnalysis(
            problem_understanding="Test understanding",
            key_challenges=["Challenge 1", "Challenge 2"],
            constraints=["Constraint 1"],
            assumptions=["Assumption 1"]
        )
        
        assert analysis.problem_understanding == "Test understanding"
        assert len(analysis.key_challenges) == 2
        assert len(analysis.constraints) == 1
        assert len(analysis.assumptions) == 1
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = ArchitecturalAnalysis(
            problem_understanding="Test understanding",
            key_challenges=["Challenge 1"],
            constraints=["Constraint 1"],
            assumptions=["Assumption 1"]
        )
        
        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["problem_understanding"] == "Test understanding"
        assert result["key_challenges"] == ["Challenge 1"]
        assert result["constraints"] == ["Constraint 1"]
        assert result["assumptions"] == ["Assumption 1"]


class TestArchitecturalSolution:
    """Test cases for ArchitecturalSolution class."""
    
    def test_initialization(self):
        """Test ArchitecturalSolution initialization."""
        components = [{"name": "API Gateway", "responsibility": "Routing"}]
        tech_stack = {"backend": "Python", "database": "PostgreSQL"}
        patterns = [{"pattern": "Microservices", "description": "Distributed services"}]
        
        solution = ArchitecturalSolution(
            overview="Test solution",
            components=components,
            technology_stack=tech_stack,
            patterns=patterns,
            diagrams=["diagram1"]
        )
        
        assert solution.overview == "Test solution"
        assert len(solution.components) == 1
        assert len(solution.technology_stack) == 2
        assert len(solution.patterns) == 1
        assert len(solution.diagrams) == 1
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        solution = ArchitecturalSolution(
            overview="Test solution",
            components=[{"name": "Service", "responsibility": "Logic"}],
            technology_stack={"backend": "Python"},
            patterns=[{"pattern": "MVC", "description": "Model-View-Controller"}]
        )
        
        result = solution.to_dict()
        
        assert isinstance(result, dict)
        assert result["overview"] == "Test solution"
        assert len(result["components"]) == 1
        assert result["technology_stack"]["backend"] == "Python"
        assert len(result["patterns"]) == 1
        assert result["diagrams"] == []