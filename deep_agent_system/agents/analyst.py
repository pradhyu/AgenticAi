"""Analyst Agent for question routing and classification."""

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


class ClassificationResult:
    """Result of question classification."""
    
    def __init__(
        self,
        category: str,
        confidence: float,
        reasoning: str,
        recommended_agent: str,
        additional_context: Optional[str] = None
    ):
        self.category = category
        self.confidence = confidence
        self.reasoning = reasoning
        self.recommended_agent = recommended_agent
        self.additional_context = additional_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "category": self.category,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommended_agent": self.recommended_agent,
            "additional_context": self.additional_context,
        }


class RoutingDecision:
    """Result of routing decision."""
    
    def __init__(
        self,
        strategy: str,
        target_agents: List[str],
        priority_order: Optional[List[str]] = None,
        context_package: Optional[str] = None,
        expected_workflow: Optional[str] = None
    ):
        self.strategy = strategy
        self.target_agents = target_agents
        self.priority_order = priority_order or target_agents
        self.context_package = context_package
        self.expected_workflow = expected_workflow
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "strategy": self.strategy,
            "target_agents": self.target_agents,
            "priority_order": self.priority_order,
            "context_package": self.context_package,
            "expected_workflow": self.expected_workflow,
        }


class AnalystAgent(BaseAgent):
    """Analyst agent responsible for question routing and classification."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Analyst agent."""
        super().__init__(*args, **kwargs)
        
        # Validate that this agent has the required capabilities
        required_capabilities = [Capability.QUESTION_ROUTING, Capability.CLASSIFICATION]
        for capability in required_capabilities:
            if not self.has_capability(capability):
                logger.warning(
                    f"AnalystAgent {self.agent_id} missing required capability: {capability}"
                )
        
        # Category to agent type mapping
        self._category_to_agent = {
            "ARCHITECTURE": AgentType.ARCHITECT.value,
            "DEVELOPMENT": AgentType.DEVELOPER.value,
            "CODE_REVIEW": AgentType.CODE_REVIEWER.value,
            "TESTING": AgentType.TESTER.value,
            "GENERAL": AgentType.ANALYST.value,  # Handle general questions ourselves
        }
        
        logger.info(f"AnalystAgent {self.agent_id} initialized successfully")
    
    def _process_message_impl(self, message: Message) -> Response:
        """Process incoming messages for routing and classification."""
        try:
            if message.message_type == MessageType.USER_QUESTION:
                return self._handle_user_question(message)
            elif message.message_type == MessageType.AGENT_REQUEST:
                return self._handle_agent_request(message)
            else:
                return self._create_error_response(
                    message.id,
                    f"Unsupported message type: {message.message_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message in AnalystAgent: {e}")
            return self._create_error_response(message.id, str(e))
    
    def _handle_user_question(self, message: Message) -> Response:
        """Handle user questions by classifying and routing them."""
        question = message.content
        
        # Step 1: Classify the question
        classification = self._classify_question(question)
        
        # Step 2: Make routing decision
        routing_decision = self._make_routing_decision(question, classification)
        
        # Step 3: Execute routing strategy
        if routing_decision.strategy == "SINGLE":
            return self._route_to_single_agent(message, classification, routing_decision)
        elif routing_decision.strategy == "MULTIPLE":
            return self._route_to_multiple_agents(message, classification, routing_decision)
        elif routing_decision.strategy == "CLARIFICATION":
            return self._request_clarification(message, classification)
        else:
            return self._create_error_response(
                message.id,
                f"Unknown routing strategy: {routing_decision.strategy}"
            )
    
    def _handle_agent_request(self, message: Message) -> Response:
        """Handle requests from other agents."""
        # For now, treat agent requests as general questions
        # In the future, this could handle more sophisticated inter-agent communication
        return self._handle_user_question(message)
    
    def _classify_question(self, question: str) -> ClassificationResult:
        """Classify a user question into appropriate categories."""
        try:
            # Retrieve context if RAG is enabled
            context = None
            if self.config.rag_enabled and self.rag_manager:
                context = self.retrieve_context(question, RetrievalType.VECTOR, k=3)
            
            # Get classification prompt
            prompt_template = self.get_prompt("classification")
            
            # Format the prompt with the question
            formatted_prompt = prompt_template.format(user_question=question)
            
            # Add context if available
            if context and context.documents:
                context_text = "\n".join([doc.content for doc in context.documents[:3]])
                formatted_prompt += f"\n\nRelevant Context:\n{context_text}"
            
            # For now, implement rule-based classification
            # In a real implementation, this would use an LLM
            classification = self._rule_based_classification(question)
            
            logger.debug(f"Classified question as: {classification.category}")
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            # Return a default classification
            return ClassificationResult(
                category="GENERAL",
                confidence=0.5,
                reasoning=f"Classification failed: {e}",
                recommended_agent=AgentType.ANALYST.value,
                additional_context="Please rephrase your question for better classification."
            )
    
    def _rule_based_classification(self, question: str) -> ClassificationResult:
        """Simple rule-based classification for demonstration."""
        question_lower = question.lower()
        
        # Architecture keywords
        architecture_keywords = [
            "architecture", "design", "system", "pattern", "structure",
            "scalability", "microservices", "api", "database design",
            "high-level", "overview", "approach", "strategy"
        ]
        
        # Development keywords
        development_keywords = [
            "implement", "code", "function", "class", "method", "algorithm",
            "programming", "write", "create", "build", "develop", "syntax",
            "library", "framework", "example"
        ]
        
        # Code review keywords
        review_keywords = [
            "review", "optimize", "improve", "debug", "fix", "error",
            "bug", "performance", "refactor", "best practice", "clean",
            "analyze", "check", "validate", "suggest", "optimization"
        ]
        
        # Testing keywords
        testing_keywords = [
            "test", "testing", "unit test", "integration test", "mock",
            "assert", "coverage", "qa", "quality", "validation",
            "pytest", "unittest", "tdd"
        ]
        
        # Count keyword matches
        arch_score = sum(1 for keyword in architecture_keywords if keyword in question_lower)
        dev_score = sum(1 for keyword in development_keywords if keyword in question_lower)
        review_score = sum(1 for keyword in review_keywords if keyword in question_lower)
        test_score = sum(1 for keyword in testing_keywords if keyword in question_lower)
        
        # Determine category based on highest score
        scores = {
            "ARCHITECTURE": arch_score,
            "DEVELOPMENT": dev_score,
            "CODE_REVIEW": review_score,
            "TESTING": test_score,
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            category = "GENERAL"
            confidence = 0.6
            reasoning = "No specific domain keywords detected"
        else:
            category = max(scores, key=scores.get)
            confidence = min(0.9, 0.5 + (max_score * 0.1))
            reasoning = f"Detected {max_score} relevant keywords for {category.lower()}"
        
        recommended_agent = self._category_to_agent.get(category, AgentType.ANALYST.value)
        
        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            recommended_agent=recommended_agent
        )
    
    def _make_routing_decision(
        self,
        question: str,
        classification: ClassificationResult
    ) -> RoutingDecision:
        """Make a routing decision based on classification results."""
        try:
            # Get routing prompt
            prompt_template = self.get_prompt("routing")
            
            # Format the prompt
            classification_text = (
                f"Classification: {classification.category}\n"
                f"Confidence: {classification.confidence}\n"
                f"Reasoning: {classification.reasoning}\n"
                f"Recommended Agent: {classification.recommended_agent}"
            )
            
            formatted_prompt = prompt_template.format(
                classification_results=classification_text,
                user_question=question
            )
            
            # For now, implement simple routing logic
            # In a real implementation, this would use an LLM
            return self._simple_routing_logic(classification)
            
        except Exception as e:
            logger.error(f"Error making routing decision: {e}")
            # Return a default routing decision
            return RoutingDecision(
                strategy="SINGLE",
                target_agents=[AgentType.ANALYST.value],
                context_package="Routing decision failed, handling as general question."
            )
    
    def _simple_routing_logic(self, classification: ClassificationResult) -> RoutingDecision:
        """Simple routing logic based on classification."""
        if classification.confidence < 0.6:
            return RoutingDecision(
                strategy="CLARIFICATION",
                target_agents=[],
                context_package="Low confidence in classification, requesting clarification."
            )
        
        if classification.category == "GENERAL":
            return RoutingDecision(
                strategy="SINGLE",
                target_agents=[self.agent_id],  # Use actual agent ID instead of type
                context_package="General question handled by analyst."
            )
        
        # For high-confidence classifications, route to the appropriate agent
        target_agent = classification.recommended_agent
        
        return RoutingDecision(
            strategy="SINGLE",
            target_agents=[target_agent],
            context_package=f"Classified as {classification.category} with {classification.confidence:.2f} confidence.",
            expected_workflow=f"Route to {target_agent} for specialized handling."
        )
    
    def _route_to_single_agent(
        self,
        message: Message,
        classification: ClassificationResult,
        routing_decision: RoutingDecision
    ) -> Response:
        """Route message to a single target agent."""
        target_agent = routing_decision.target_agents[0]
        
        if target_agent == self.agent_id:
            # Handle the question ourselves
            return self._handle_general_question(message, classification)
        
        # Forward to the target agent
        forwarded_message = Message(
            sender_id=self.agent_id,
            recipient_id=target_agent,
            content=message.content,
            message_type=MessageType.AGENT_REQUEST,
            metadata={
                "original_sender": message.sender_id,
                "classification": classification.to_dict(),
                "routing_context": routing_decision.context_package,
            }
        )
        
        # Communicate with the target agent
        response = self.communicate_with_agent(target_agent, forwarded_message)
        
        if response:
            # Return the response with routing metadata
            return Response(
                message_id=message.id,
                agent_id=self.agent_id,
                content=response.content,
                confidence_score=response.confidence_score,
                context_used=response.context_used + [f"routed_via_{self.agent_id}"],
                metadata={
                    "routing_classification": classification.to_dict(),
                    "target_agent": target_agent,
                    "original_response_agent": response.agent_id,
                }
            )
        else:
            return self._create_error_response(
                message.id,
                f"Failed to get response from target agent: {target_agent}"
            )
    
    def _route_to_multiple_agents(
        self,
        message: Message,
        classification: ClassificationResult,
        routing_decision: RoutingDecision
    ) -> Response:
        """Route message to multiple agents and aggregate responses."""
        responses = []
        
        for target_agent in routing_decision.target_agents:
            if target_agent == self.agent_id:
                # Handle ourselves
                response = self._handle_general_question(message, classification)
                responses.append(response)
            else:
                # Forward to target agent
                forwarded_message = Message(
                    sender_id=self.agent_id,
                    recipient_id=target_agent,
                    content=message.content,
                    message_type=MessageType.AGENT_REQUEST,
                    metadata={
                        "original_sender": message.sender_id,
                        "classification": classification.to_dict(),
                        "routing_context": routing_decision.context_package,
                    }
                )
                
                response = self.communicate_with_agent(target_agent, forwarded_message)
                if response:
                    responses.append(response)
        
        # Aggregate responses
        return self._aggregate_responses(message, responses, classification)
    
    def _request_clarification(
        self,
        message: Message,
        classification: ClassificationResult
    ) -> Response:
        """Request clarification from the user."""
        clarification_text = (
            f"I need some clarification to better understand your question. "
            f"Based on my analysis, your question seems to be about {classification.category.lower()}, "
            f"but I'm not confident enough (confidence: {classification.confidence:.2f}) to route it appropriately.\n\n"
            f"Could you please provide more details or rephrase your question? "
            f"For example:\n"
            f"- If it's about system design or architecture, please specify the scope\n"
            f"- If it's about coding, please mention the programming language or framework\n"
            f"- If it's about code review, please provide the code you'd like reviewed\n"
            f"- If it's about testing, please specify what you'd like to test"
        )
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=clarification_text,
            confidence_score=classification.confidence,
            metadata={
                "requires_clarification": True,
                "classification": classification.to_dict(),
            }
        )
    
    def _handle_general_question(
        self,
        message: Message,
        classification: ClassificationResult
    ) -> Response:
        """Handle general questions that don't require specialized agents."""
        # For general questions, provide a helpful response
        general_response = (
            f"I understand you have a question about {classification.category.lower()}. "
            f"While I can provide some general guidance, you might get more detailed help "
            f"by asking a more specific question that I can route to one of our specialist agents:\n\n"
            f"- **Architecture questions**: Ask about system design, patterns, or high-level approaches\n"
            f"- **Development questions**: Ask for code implementation or programming help\n"
            f"- **Code review questions**: Share code you'd like reviewed or optimized\n"
            f"- **Testing questions**: Ask about test creation or testing strategies\n\n"
            f"Feel free to rephrase your question with more specific details!"
        )
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=general_response,
            confidence_score=classification.confidence,
            metadata={
                "handled_as_general": True,
                "classification": classification.to_dict(),
            }
        )
    
    def _aggregate_responses(
        self,
        original_message: Message,
        responses: List[Response],
        classification: ClassificationResult
    ) -> Response:
        """Aggregate multiple agent responses into a unified response."""
        if not responses:
            return self._create_error_response(
                original_message.id,
                "No responses received from target agents"
            )
        
        if len(responses) == 1:
            # Single response, just return it with routing metadata
            response = responses[0]
            response.metadata["routing_classification"] = classification.to_dict()
            return response
        
        # Multiple responses - aggregate them
        try:
            # Get aggregation prompt
            prompt_template = self.get_prompt("aggregation")
            
            # Format agent responses
            agent_responses_text = ""
            for i, response in enumerate(responses, 1):
                agent_responses_text += f"Agent {response.agent_id}:\n{response.content}\n\n"
            
            # For now, create a simple aggregated response
            # In a real implementation, this would use an LLM
            aggregated_content = self._simple_aggregation(responses, original_message.content)
            
            # Calculate average confidence
            avg_confidence = sum(r.confidence_score for r in responses) / len(responses)
            
            # Combine context used
            all_context = []
            for response in responses:
                all_context.extend(response.context_used)
            
            return Response(
                message_id=original_message.id,
                agent_id=self.agent_id,
                content=aggregated_content,
                confidence_score=avg_confidence,
                context_used=list(set(all_context)),  # Remove duplicates
                metadata={
                    "aggregated_from": [r.agent_id for r in responses],
                    "classification": classification.to_dict(),
                    "response_count": len(responses),
                }
            )
            
        except Exception as e:
            logger.error(f"Error aggregating responses: {e}")
            # Fall back to the first response
            return responses[0]
    
    def _simple_aggregation(self, responses: List[Response], original_question: str) -> str:
        """Simple aggregation of multiple responses."""
        if len(responses) == 1:
            return responses[0].content
        
        aggregated = f"Based on analysis from multiple specialist agents, here's a comprehensive response to your question:\n\n"
        
        for i, response in enumerate(responses, 1):
            agent_name = response.agent_id.replace("_", " ").title()
            aggregated += f"**{agent_name} Perspective:**\n{response.content}\n\n"
        
        aggregated += "This response combines insights from multiple specialists to give you a well-rounded answer."
        
        return aggregated
    
    def _create_error_response(self, message_id: str, error_message: str) -> Response:
        """Create an error response."""
        return Response(
            message_id=message_id,
            agent_id=self.agent_id,
            content=f"I encountered an error while processing your request: {error_message}",
            confidence_score=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def classify_question(self, question: str) -> ClassificationResult:
        """Public method to classify a question without routing."""
        return self._classify_question(question)
    
    def get_routing_decision(
        self,
        question: str,
        classification: Optional[ClassificationResult] = None
    ) -> RoutingDecision:
        """Public method to get routing decision for a question."""
        if classification is None:
            classification = self._classify_question(question)
        return self._make_routing_decision(question, classification)