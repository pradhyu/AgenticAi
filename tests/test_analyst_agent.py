"""Tests for the AnalystAgent class."""

import pytest
from unittest.mock import Mock, patch

from deep_agent_system.agents.analyst import AnalystAgent, ClassificationResult, RoutingDecision
from deep_agent_system.config.models import SystemConfig
from deep_agent_system.models.agents import AgentConfig, AgentType, Capability, LLMConfig
from deep_agent_system.models.messages import (
    Context,
    Document,
    Message,
    MessageType,
    Response,
    RetrievalType,
)
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
    
    # Mock context retrieval
    mock_docs = [
        Document(content="Test document 1", metadata={"source": "test1"}),
        Document(content="Test document 2", metadata={"source": "test2"}),
    ]
    mock_context = Context(
        query="test query",
        documents=mock_docs,
        retrieval_method=RetrievalType.VECTOR,
        relevance_scores=[0.9, 0.8]
    )
    manager.retrieve_vector_context.return_value = mock_docs
    manager.hybrid_retrieve.return_value = mock_context
    
    return manager


@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        agent_id="test_analyst",
        agent_type=AgentType.ANALYST,
        llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
        capabilities=[Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
        rag_enabled=True,
        graph_rag_enabled=True,
    )


@pytest.fixture
def analyst_agent(agent_config, mock_prompt_manager, mock_rag_manager):
    """Create a test AnalystAgent instance."""
    return AnalystAgent(
        config=agent_config,
        prompt_manager=mock_prompt_manager,
        rag_manager=mock_rag_manager,
    )


class TestAnalystAgent:
    """Test cases for AnalystAgent."""
    
    def test_initialization(self, analyst_agent):
        """Test agent initialization."""
        assert analyst_agent.agent_id == "test_analyst"
        assert analyst_agent.agent_type == "analyst"
        assert analyst_agent.has_capability(Capability.QUESTION_ROUTING)
        assert analyst_agent.has_capability(Capability.CLASSIFICATION)
    
    def test_initialization_missing_capabilities(self, mock_prompt_manager, mock_rag_manager):
        """Test initialization with missing required capabilities."""
        # The AgentConfig validation will prevent creating an agent without required capabilities
        # So we test that the validation works correctly
        with pytest.raises(ValueError, match="Agent type AgentType.ANALYST requires capabilities"):
            AgentConfig(
                agent_id="test_analyst",
                agent_type=AgentType.ANALYST,
                llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
                capabilities=[],  # Missing required capabilities
                rag_enabled=True,
            )
    
    def test_rule_based_classification_architecture(self, analyst_agent):
        """Test rule-based classification for architecture questions."""
        question = "How should I design a scalable microservices architecture?"
        
        classification = analyst_agent._rule_based_classification(question)
        
        assert classification.category == "ARCHITECTURE"
        assert classification.confidence > 0.5
        assert classification.recommended_agent == "architect"
        assert "architecture" in classification.reasoning.lower()
    
    def test_rule_based_classification_development(self, analyst_agent):
        """Test rule-based classification for development questions."""
        question = "How do I implement a function to sort an array in Python?"
        
        classification = analyst_agent._rule_based_classification(question)
        
        assert classification.category == "DEVELOPMENT"
        assert classification.confidence > 0.5
        assert classification.recommended_agent == "developer"
        assert "development" in classification.reasoning.lower()
    
    def test_rule_based_classification_code_review(self, analyst_agent):
        """Test rule-based classification for code review questions."""
        question = "Can you review this code and suggest optimizations?"
        
        classification = analyst_agent._rule_based_classification(question)
        
        assert classification.category == "CODE_REVIEW"
        assert classification.confidence > 0.5
        assert classification.recommended_agent == "code_reviewer"
    
    def test_rule_based_classification_testing(self, analyst_agent):
        """Test rule-based classification for testing questions."""
        question = "How do I write unit tests for this function using pytest?"
        
        classification = analyst_agent._rule_based_classification(question)
        
        assert classification.category == "TESTING"
        assert classification.confidence > 0.5
        assert classification.recommended_agent == "tester"
    
    def test_rule_based_classification_general(self, analyst_agent):
        """Test rule-based classification for general questions."""
        question = "What is the weather like today?"
        
        classification = analyst_agent._rule_based_classification(question)
        
        assert classification.category == "GENERAL"
        assert classification.confidence >= 0.5
        assert classification.recommended_agent == "analyst"
    
    def test_simple_routing_logic_high_confidence(self, analyst_agent):
        """Test routing logic with high confidence classification."""
        classification = ClassificationResult(
            category="DEVELOPMENT",
            confidence=0.9,
            reasoning="High confidence development question",
            recommended_agent="developer"
        )
        
        routing = analyst_agent._simple_routing_logic(classification)
        
        assert routing.strategy == "SINGLE"
        assert routing.target_agents == ["developer"]
        assert "0.90 confidence" in routing.context_package
    
    def test_simple_routing_logic_low_confidence(self, analyst_agent):
        """Test routing logic with low confidence classification."""
        classification = ClassificationResult(
            category="GENERAL",
            confidence=0.4,
            reasoning="Low confidence classification",
            recommended_agent="analyst"
        )
        
        routing = analyst_agent._simple_routing_logic(classification)
        
        assert routing.strategy == "CLARIFICATION"
        assert routing.target_agents == []
        assert "Low confidence" in routing.context_package
    
    def test_simple_routing_logic_general_question(self, analyst_agent):
        """Test routing logic for general questions."""
        classification = ClassificationResult(
            category="GENERAL",
            confidence=0.8,
            reasoning="General question",
            recommended_agent="analyst"
        )
        
        routing = analyst_agent._simple_routing_logic(classification)
        
        assert routing.strategy == "SINGLE"
        assert routing.target_agents == ["test_analyst"]  # Should use actual agent ID
        assert "General question" in routing.context_package
    
    def test_handle_user_question_single_routing(self, analyst_agent):
        """Test handling user question with single agent routing."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="How do I implement a binary search algorithm?",
            message_type=MessageType.USER_QUESTION,
        )
        
        # Mock the agent registry to include a developer agent
        mock_developer = Mock()
        mock_developer.process_message.return_value = Response(
            message_id=message.id,
            agent_id="developer",
            content="Here's how to implement binary search...",
            confidence_score=0.9,
        )
        analyst_agent._agent_registry["developer"] = mock_developer
        
        response = analyst_agent._handle_user_question(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert "binary search" in response.content or "routed" in str(response.metadata)
    
    def test_handle_user_question_clarification(self, analyst_agent):
        """Test handling user question that requires clarification."""
        # Register the agent with itself so it can handle self-routing
        analyst_agent.register_agent(analyst_agent)
        
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Help me with something",  # Vague question
            message_type=MessageType.USER_QUESTION,
        )
        
        response = analyst_agent._handle_user_question(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        # The response should either be clarification or general handling
        assert ("clarification" in response.content.lower() or 
                "general" in response.content.lower() or
                "specialist agents" in response.content.lower())
        # Check if it's either clarification or general handling
        is_clarification = response.metadata.get("requires_clarification") is True
        is_general = response.metadata.get("handled_as_general") is True
        assert is_clarification or is_general
    
    def test_handle_general_question(self, analyst_agent):
        """Test handling general questions directly."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="What can you help me with?",
            message_type=MessageType.USER_QUESTION,
        )
        
        classification = ClassificationResult(
            category="GENERAL",
            confidence=0.8,
            reasoning="General inquiry",
            recommended_agent="analyst"
        )
        
        response = analyst_agent._handle_general_question(message, classification)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert "specialist agents" in response.content.lower()
        assert response.metadata.get("handled_as_general") is True
    
    def test_request_clarification(self, analyst_agent):
        """Test requesting clarification from user."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Unclear question",
            message_type=MessageType.USER_QUESTION,
        )
        
        classification = ClassificationResult(
            category="GENERAL",
            confidence=0.3,
            reasoning="Unclear intent",
            recommended_agent="analyst"
        )
        
        response = analyst_agent._request_clarification(message, classification)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert "clarification" in response.content.lower()
        assert response.metadata.get("requires_clarification") is True
        assert response.confidence_score == 0.3
    
    def test_aggregate_responses_single(self, analyst_agent):
        """Test aggregating a single response."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Test question",
            message_type=MessageType.USER_QUESTION,
        )
        
        responses = [
            Response(
                message_id=message.id,
                agent_id="developer",
                content="Single response content",
                confidence_score=0.9,
            )
        ]
        
        classification = ClassificationResult(
            category="DEVELOPMENT",
            confidence=0.8,
            reasoning="Test classification",
            recommended_agent="developer"
        )
        
        result = analyst_agent._aggregate_responses(message, responses, classification)
        
        assert result.content == "Single response content"
        assert "routing_classification" in result.metadata
    
    def test_aggregate_responses_multiple(self, analyst_agent):
        """Test aggregating multiple responses."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Test question",
            message_type=MessageType.USER_QUESTION,
        )
        
        responses = [
            Response(
                message_id=message.id,
                agent_id="developer",
                content="Developer response",
                confidence_score=0.9,
                context_used=["dev_context"]
            ),
            Response(
                message_id=message.id,
                agent_id="architect",
                content="Architect response",
                confidence_score=0.8,
                context_used=["arch_context"]
            )
        ]
        
        classification = ClassificationResult(
            category="DEVELOPMENT",
            confidence=0.8,
            reasoning="Test classification",
            recommended_agent="developer"
        )
        
        result = analyst_agent._aggregate_responses(message, responses, classification)
        
        assert "Developer Perspective" in result.content
        assert "Architect Perspective" in result.content
        assert abs(result.confidence_score - 0.85) < 0.001  # Average of 0.9 and 0.8 (allow for floating point precision)
        assert "dev_context" in result.context_used
        assert "arch_context" in result.context_used
        assert result.metadata["response_count"] == 2
    
    def test_aggregate_responses_empty(self, analyst_agent):
        """Test aggregating empty responses list."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Test question",
            message_type=MessageType.USER_QUESTION,
        )
        
        classification = ClassificationResult(
            category="DEVELOPMENT",
            confidence=0.8,
            reasoning="Test classification",
            recommended_agent="developer"
        )
        
        result = analyst_agent._aggregate_responses(message, [], classification)
        
        assert "error" in result.content.lower()
        assert result.confidence_score == 0.0
        assert result.metadata.get("error") is True
    
    def test_process_message_user_question(self, analyst_agent):
        """Test processing user question message."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="How do I design a REST API?",
            message_type=MessageType.USER_QUESTION,
        )
        
        response = analyst_agent.process_message(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert isinstance(response.content, str)
        assert len(response.content) > 0
    
    def test_process_message_agent_request(self, analyst_agent):
        """Test processing agent request message."""
        message = Message(
            sender_id="other_agent",
            recipient_id="test_analyst",
            content="Route this question please",
            message_type=MessageType.AGENT_REQUEST,
        )
        
        response = analyst_agent.process_message(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert isinstance(response.content, str)
    
    def test_process_message_unsupported_type(self, analyst_agent):
        """Test processing unsupported message type."""
        message = Message(
            sender_id="user",
            recipient_id="test_analyst",
            content="Test message",
            message_type=MessageType.SYSTEM_MESSAGE,
        )
        
        response = analyst_agent.process_message(message)
        
        assert response.message_id == message.id
        assert response.agent_id == "test_analyst"
        assert "unsupported message type" in response.content.lower()
        assert response.metadata.get("error") is True
    
    def test_classify_question_public_method(self, analyst_agent):
        """Test public classify_question method."""
        question = "How do I implement a sorting algorithm?"
        
        classification = analyst_agent.classify_question(question)
        
        assert isinstance(classification, ClassificationResult)
        assert classification.category in ["ARCHITECTURE", "DEVELOPMENT", "CODE_REVIEW", "TESTING", "GENERAL"]
        assert 0.0 <= classification.confidence <= 1.0
        assert classification.recommended_agent in ["analyst", "architect", "developer", "code_reviewer", "tester"]
    
    def test_get_routing_decision_public_method(self, analyst_agent):
        """Test public get_routing_decision method."""
        question = "How do I write unit tests?"
        
        routing = analyst_agent.get_routing_decision(question)
        
        assert isinstance(routing, RoutingDecision)
        assert routing.strategy in ["SINGLE", "MULTIPLE", "CLARIFICATION"]
        assert isinstance(routing.target_agents, list)
    
    def test_get_routing_decision_with_classification(self, analyst_agent):
        """Test get_routing_decision with provided classification."""
        question = "Test question"
        classification = ClassificationResult(
            category="TESTING",
            confidence=0.9,
            reasoning="Test classification",
            recommended_agent="tester"
        )
        
        routing = analyst_agent.get_routing_decision(question, classification)
        
        assert routing.strategy == "SINGLE"
        assert routing.target_agents == ["tester"]
    
    def test_classification_result_to_dict(self):
        """Test ClassificationResult to_dict method."""
        classification = ClassificationResult(
            category="DEVELOPMENT",
            confidence=0.8,
            reasoning="Test reasoning",
            recommended_agent="developer",
            additional_context="Test context"
        )
        
        result_dict = classification.to_dict()
        
        assert result_dict["category"] == "DEVELOPMENT"
        assert result_dict["confidence"] == 0.8
        assert result_dict["reasoning"] == "Test reasoning"
        assert result_dict["recommended_agent"] == "developer"
        assert result_dict["additional_context"] == "Test context"
    
    def test_routing_decision_to_dict(self):
        """Test RoutingDecision to_dict method."""
        routing = RoutingDecision(
            strategy="SINGLE",
            target_agents=["developer"],
            priority_order=["developer"],
            context_package="Test context",
            expected_workflow="Test workflow"
        )
        
        result_dict = routing.to_dict()
        
        assert result_dict["strategy"] == "SINGLE"
        assert result_dict["target_agents"] == ["developer"]
        assert result_dict["priority_order"] == ["developer"]
        assert result_dict["context_package"] == "Test context"
        assert result_dict["expected_workflow"] == "Test workflow"
    
    def test_context_retrieval_integration(self, analyst_agent, mock_rag_manager):
        """Test integration with RAG context retrieval."""
        question = "How do I implement authentication?"
        
        # The classify_question method should use RAG if enabled
        classification = analyst_agent.classify_question(question)
        
        # Verify RAG was called
        mock_rag_manager.retrieve_vector_context.assert_called()
        
        # Should still return a valid classification
        assert isinstance(classification, ClassificationResult)
        assert classification.category in ["ARCHITECTURE", "DEVELOPMENT", "CODE_REVIEW", "TESTING", "GENERAL"]
    
    def test_error_handling_in_classification(self, analyst_agent, mock_rag_manager):
        """Test error handling during classification."""
        # Make RAG manager raise an exception
        mock_rag_manager.retrieve_vector_context.side_effect = Exception("RAG error")
        
        question = "Test question"
        classification = analyst_agent.classify_question(question)
        
        # Should still return a classification despite the error
        assert isinstance(classification, ClassificationResult)
        # The classification should still work based on rule-based logic, even if RAG fails
        assert classification.category in ["ARCHITECTURE", "DEVELOPMENT", "CODE_REVIEW", "TESTING", "GENERAL"]
        # The reasoning should indicate either classification failed or be based on rule-based logic
        assert ("Classification failed" in classification.reasoning or 
                "keyword" in classification.reasoning.lower())