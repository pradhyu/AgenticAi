"""Workflow manager for creating and executing LangGraph workflows dynamically."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.models.agents import AgentType, Capability


logger = logging.getLogger(__name__)


class WorkflowComplexity(str, Enum):
    """Complexity levels for workflows."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class WorkflowPattern(str, Enum):
    """Common workflow patterns."""
    SINGLE_AGENT = "single_agent"
    SEQUENTIAL_CHAIN = "sequential_chain"
    PARALLEL_PROCESSING = "parallel_processing"
    HIERARCHICAL_REVIEW = "hierarchical_review"
    COLLABORATIVE_REFINEMENT = "collaborative_refinement"
    ITERATIVE_IMPROVEMENT = "iterative_improvement"


class WorkflowState(Dict[str, Any]):
    """Enhanced state object for LangGraph workflows."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure required fields exist
        if "messages" not in self:
            self["messages"] = []
        if "responses" not in self:
            self["responses"] = []
        if "current_step" not in self:
            self["current_step"] = 0
        if "workflow_id" not in self:
            self["workflow_id"] = str(uuid4())
        if "status" not in self:
            self["status"] = "pending"
        if "metadata" not in self:
            self["metadata"] = {}
        if "context" not in self:
            self["context"] = {}
        if "iteration_count" not in self:
            self["iteration_count"] = 0
        if "max_iterations" not in self:
            self["max_iterations"] = 5


class WorkflowTemplate:
    """Template for creating workflows based on common patterns."""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        pattern: WorkflowPattern,
        agent_types: List[AgentType],
        required_capabilities: List[Capability],
        complexity: WorkflowComplexity = WorkflowComplexity.SIMPLE,
        max_iterations: int = 5,
        timeout: int = 300,
    ):
        """Initialize workflow template.
        
        Args:
            template_id: Unique identifier for the template
            name: Human-readable name
            description: Description of the workflow
            pattern: Workflow pattern type
            agent_types: Required agent types
            required_capabilities: Required capabilities
            complexity: Complexity level
            max_iterations: Maximum iterations for iterative workflows
            timeout: Timeout in seconds
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.pattern = pattern
        self.agent_types = agent_types
        self.required_capabilities = required_capabilities
        self.complexity = complexity
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.created_at = datetime.utcnow()


class WorkflowSpec:
    """Specification for creating a workflow."""
    
    def __init__(
        self,
        workflow_id: str,
        template: Optional[WorkflowTemplate] = None,
        agent_ids: Optional[List[str]] = None,
        custom_nodes: Optional[Dict[str, Callable]] = None,
        custom_edges: Optional[List[tuple]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        conditions: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize workflow specification.
        
        Args:
            workflow_id: Unique identifier for the workflow
            template: Optional template to base workflow on
            agent_ids: List of specific agent IDs to use
            custom_nodes: Custom node functions
            custom_edges: Custom edge definitions
            initial_state: Initial state for the workflow
            conditions: Conditional routing functions
        """
        self.workflow_id = workflow_id
        self.template = template
        self.agent_ids = agent_ids or []
        self.custom_nodes = custom_nodes or {}
        self.custom_edges = custom_edges or []
        self.initial_state = initial_state or {}
        self.conditions = conditions or {}


class WorkflowManager:
    """Manager for creating and executing LangGraph workflows dynamically."""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        """Initialize the workflow manager.
        
        Args:
            agents: Dictionary of available agents by ID
        """
        self.agents = agents
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._compiled_workflows: Dict[str, CompiledStateGraph] = {}
        self._workflow_cache: Dict[str, StateGraph] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
        
        logger.info("WorkflowManager initialized")
    
    def _initialize_default_templates(self) -> None:
        """Initialize default workflow templates."""
        
        # Single agent template
        single_agent_template = WorkflowTemplate(
            template_id="single_agent",
            name="Single Agent Processing",
            description="Route question to single most appropriate agent",
            pattern=WorkflowPattern.SINGLE_AGENT,
            agent_types=[AgentType.ANALYST],
            required_capabilities=[Capability.QUESTION_ROUTING],
            complexity=WorkflowComplexity.SIMPLE,
        )
        
        # Sequential chain template
        sequential_template = WorkflowTemplate(
            template_id="sequential_chain",
            name="Sequential Agent Chain",
            description="Process question through multiple agents in sequence",
            pattern=WorkflowPattern.SEQUENTIAL_CHAIN,
            agent_types=[AgentType.ANALYST, AgentType.ARCHITECT, AgentType.DEVELOPER],
            required_capabilities=[
                Capability.QUESTION_ROUTING,
                Capability.ARCHITECTURE_DESIGN,
                Capability.CODE_GENERATION,
            ],
            complexity=WorkflowComplexity.MODERATE,
        )
        
        # Parallel processing template
        parallel_template = WorkflowTemplate(
            template_id="parallel_processing",
            name="Parallel Agent Processing",
            description="Process question with multiple agents in parallel",
            pattern=WorkflowPattern.PARALLEL_PROCESSING,
            agent_types=[AgentType.ARCHITECT, AgentType.DEVELOPER, AgentType.TESTER],
            required_capabilities=[
                Capability.ARCHITECTURE_DESIGN,
                Capability.CODE_GENERATION,
                Capability.TEST_CREATION,
            ],
            complexity=WorkflowComplexity.MODERATE,
        )
        
        # Hierarchical review template
        hierarchical_template = WorkflowTemplate(
            template_id="hierarchical_review",
            name="Hierarchical Review Process",
            description="Architect designs, developer implements, reviewer validates",
            pattern=WorkflowPattern.HIERARCHICAL_REVIEW,
            agent_types=[AgentType.ARCHITECT, AgentType.DEVELOPER, AgentType.CODE_REVIEWER],
            required_capabilities=[
                Capability.ARCHITECTURE_DESIGN,
                Capability.CODE_GENERATION,
                Capability.CODE_REVIEW,
            ],
            complexity=WorkflowComplexity.COMPLEX,
        )
        
        # Collaborative refinement template
        collaborative_template = WorkflowTemplate(
            template_id="collaborative_refinement",
            name="Collaborative Refinement",
            description="Multiple agents collaborate to refine solution iteratively",
            pattern=WorkflowPattern.COLLABORATIVE_REFINEMENT,
            agent_types=[AgentType.ARCHITECT, AgentType.DEVELOPER, AgentType.CODE_REVIEWER, AgentType.TESTER],
            required_capabilities=[
                Capability.ARCHITECTURE_DESIGN,
                Capability.CODE_GENERATION,
                Capability.CODE_REVIEW,
                Capability.TEST_CREATION,
            ],
            complexity=WorkflowComplexity.COMPLEX,
            max_iterations=3,
        )
        
        # Iterative improvement template
        iterative_template = WorkflowTemplate(
            template_id="iterative_improvement",
            name="Iterative Improvement",
            description="Iteratively improve solution through multiple rounds",
            pattern=WorkflowPattern.ITERATIVE_IMPROVEMENT,
            agent_types=[AgentType.DEVELOPER, AgentType.CODE_REVIEWER],
            required_capabilities=[
                Capability.CODE_GENERATION,
                Capability.CODE_REVIEW,
            ],
            complexity=WorkflowComplexity.MODERATE,
            max_iterations=3,
        )
        
        # Register all templates
        for template in [
            single_agent_template,
            sequential_template,
            parallel_template,
            hierarchical_template,
            collaborative_template,
            iterative_template,
        ]:
            self.register_template(template)
    
    def register_template(self, template: WorkflowTemplate) -> None:
        """Register a workflow template.
        
        Args:
            template: Template to register
        """
        self._templates[template.template_id] = template
        logger.info(f"Registered workflow template: {template.name}")
    
    def get_templates(self) -> List[WorkflowTemplate]:
        """Get all registered workflow templates.
        
        Returns:
            List of workflow templates
        """
        return list(self._templates.values())
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow template.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Workflow template or None if not found
        """
        return self._templates.get(template_id)
    
    def analyze_question_complexity(self, question: str, context: Optional[Dict[str, Any]] = None) -> WorkflowComplexity:
        """Analyze question complexity to determine appropriate workflow.
        
        Args:
            question: Question to analyze
            context: Optional context information
            
        Returns:
            Complexity level of the question
        """
        # Simple heuristics for complexity analysis
        question_lower = question.lower()
        
        # Keywords that indicate complexity
        complex_keywords = [
            "architecture", "design", "system", "integrate", "implement",
            "review", "test", "optimize", "refactor", "multiple", "various"
        ]
        
        moderate_keywords = [
            "code", "function", "method", "class", "algorithm", "solution"
        ]
        
        simple_keywords = [
            "what", "how", "why", "explain", "define", "describe"
        ]
        
        # Count keyword matches
        complex_count = sum(1 for keyword in complex_keywords if keyword in question_lower)
        moderate_count = sum(1 for keyword in moderate_keywords if keyword in question_lower)
        simple_count = sum(1 for keyword in simple_keywords if keyword in question_lower)
        
        # Determine complexity based on question length and keywords
        if len(question.split()) > 20 or complex_count >= 2:
            return WorkflowComplexity.COMPLEX
        elif len(question.split()) > 10 or moderate_count >= 1 or complex_count >= 1:
            return WorkflowComplexity.MODERATE
        else:
            return WorkflowComplexity.SIMPLE
    
    def suggest_workflow_template(
        self,
        question: str,
        available_agents: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowTemplate]:
        """Suggest the most appropriate workflow template for a question.
        
        Args:
            question: Question to process
            available_agents: List of available agent IDs
            context: Optional context information
            
        Returns:
            Suggested workflow template or None
        """
        complexity = self.analyze_question_complexity(question, context)
        available_agents = available_agents or list(self.agents.keys())
        
        # Get agent types for available agents
        available_types = set()
        for agent_id in available_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if hasattr(agent, 'config') and hasattr(agent.config, 'agent_type'):
                    available_types.add(agent.config.agent_type)
        
        # Find templates that match complexity and available agents
        suitable_templates = []
        for template in self._templates.values():
            if template.complexity == complexity:
                # Check if we have the required agent types
                required_types = set(template.agent_types)
                if required_types.issubset(available_types):
                    suitable_templates.append(template)
        
        # If no exact complexity match, try simpler templates
        if not suitable_templates and complexity != WorkflowComplexity.SIMPLE:
            for template in self._templates.values():
                if template.complexity == WorkflowComplexity.SIMPLE:
                    required_types = set(template.agent_types)
                    if required_types.issubset(available_types):
                        suitable_templates.append(template)
        
        # Return the first suitable template (could be enhanced with better selection logic)
        return suitable_templates[0] if suitable_templates else None
    
    def create_workflow(self, workflow_spec: WorkflowSpec) -> StateGraph:
        """Create a LangGraph workflow from a specification.
        
        Args:
            workflow_spec: Workflow specification
            
        Returns:
            LangGraph StateGraph
        """
        workflow_id = workflow_spec.workflow_id
        
        # Check cache first
        if workflow_id in self._workflow_cache:
            logger.debug(f"Using cached workflow: {workflow_id}")
            return self._workflow_cache[workflow_id]
        
        # Create new workflow
        workflow = StateGraph(WorkflowState)
        
        if workflow_spec.template:
            # Create workflow from template
            self._create_workflow_from_template(workflow, workflow_spec)
        else:
            # Create custom workflow
            self._create_custom_workflow(workflow, workflow_spec)
        
        # Cache the workflow
        self._workflow_cache[workflow_id] = workflow
        
        logger.info(f"Created workflow: {workflow_id}")
        return workflow
    
    def _create_workflow_from_template(self, workflow: StateGraph, spec: WorkflowSpec) -> None:
        """Create workflow based on a template.
        
        Args:
            workflow: StateGraph to populate
            spec: Workflow specification with template
        """
        template = spec.template
        agent_ids = spec.agent_ids or self._get_agents_for_template(template)
        
        if template.pattern == WorkflowPattern.SINGLE_AGENT:
            self._create_single_agent_workflow(workflow, agent_ids)
        elif template.pattern == WorkflowPattern.SEQUENTIAL_CHAIN:
            self._create_sequential_workflow(workflow, agent_ids)
        elif template.pattern == WorkflowPattern.PARALLEL_PROCESSING:
            self._create_parallel_workflow(workflow, agent_ids)
        elif template.pattern == WorkflowPattern.HIERARCHICAL_REVIEW:
            self._create_hierarchical_workflow(workflow, agent_ids)
        elif template.pattern == WorkflowPattern.COLLABORATIVE_REFINEMENT:
            self._create_collaborative_workflow(workflow, agent_ids, template.max_iterations)
        elif template.pattern == WorkflowPattern.ITERATIVE_IMPROVEMENT:
            self._create_iterative_workflow(workflow, agent_ids, template.max_iterations)
        else:
            raise ValueError(f"Unknown workflow pattern: {template.pattern}")
    
    def _create_custom_workflow(self, workflow: StateGraph, spec: WorkflowSpec) -> None:
        """Create custom workflow from specification.
        
        Args:
            workflow: StateGraph to populate
            spec: Workflow specification
        """
        # Add custom nodes
        for node_name, node_func in spec.custom_nodes.items():
            workflow.add_node(node_name, node_func)
        
        # Add agent nodes if specified
        for agent_id in spec.agent_ids:
            if agent_id in self.agents:
                workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Add custom edges
        for edge in spec.custom_edges:
            if len(edge) == 2:
                workflow.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                # Conditional edge
                workflow.add_conditional_edges(edge[0], edge[1], edge[2])
        
        # Set entry point if agents are specified
        if spec.agent_ids:
            workflow.set_entry_point(spec.agent_ids[0])
    
    def _get_agents_for_template(self, template: WorkflowTemplate) -> List[str]:
        """Get available agent IDs that match template requirements.
        
        Args:
            template: Workflow template
            
        Returns:
            List of matching agent IDs
        """
        matching_agents = []
        
        for agent_type in template.agent_types:
            # Find an agent of this type
            for agent_id, agent in self.agents.items():
                if (hasattr(agent, 'config') and 
                    hasattr(agent.config, 'agent_type') and
                    agent.config.agent_type == agent_type):
                    matching_agents.append(agent_id)
                    break
        
        return matching_agents
    
    def _create_agent_node(self, agent_id: str) -> Callable:
        """Create a node function for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Node function for LangGraph
        """
        def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute agent processing within the workflow."""
            try:
                logger.debug(f"Executing agent node: {agent_id}")
                
                # Update current step
                state["current_step"] += 1
                
                # Get the latest message to process
                messages = state.get("messages", [])
                if not messages:
                    logger.warning(f"No messages to process for agent {agent_id}")
                    return state
                
                latest_message = messages[-1]
                
                # Create a proper Message object for the agent
                if isinstance(latest_message, dict):
                    agent_message = Message(
                        sender_id="workflow",
                        recipient_id=agent_id,
                        content=latest_message.get("content", ""),
                        message_type=MessageType.AGENT_REQUEST,
                    )
                else:
                    agent_message = latest_message
                
                # Process message with the agent
                if agent_id not in self.agents:
                    logger.error(f"Agent {agent_id} not found")
                    return state
                
                agent = self.agents[agent_id]
                response = agent.process_message(agent_message)
                
                if response:
                    # Add response to state
                    responses = state.get("responses", [])
                    responses.append({
                        "agent_id": agent_id,
                        "content": response.content,
                        "confidence_score": response.confidence_score,
                        "timestamp": response.timestamp.isoformat(),
                        "step": state["current_step"],
                    })
                    state["responses"] = responses
                    
                    # Update context with response
                    context = state.get("context", {})
                    context[f"response_{agent_id}"] = response.content
                    state["context"] = context
                    
                    logger.debug(f"Agent {agent_id} completed processing")
                else:
                    logger.error(f"No response received from agent {agent_id}")
                
                return state
                
            except Exception as e:
                logger.error(f"Error in agent node {agent_id}: {e}")
                state["status"] = "failed"
                state["error"] = str(e)
                return state
        
        return agent_node
    
    def _create_single_agent_workflow(self, workflow: StateGraph, agent_ids: List[str]) -> None:
        """Create a single agent workflow.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs (only first will be used)
        """
        if not agent_ids:
            raise ValueError("At least one agent required for single agent workflow")
        
        agent_id = agent_ids[0]
        workflow.add_node(agent_id, self._create_agent_node(agent_id))
        workflow.set_entry_point(agent_id)
        workflow.add_edge(agent_id, END)
    
    def _create_sequential_workflow(self, workflow: StateGraph, agent_ids: List[str]) -> None:
        """Create a sequential workflow.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs in sequence
        """
        if not agent_ids:
            raise ValueError("At least one agent required for sequential workflow")
        
        # Add all agent nodes
        for agent_id in agent_ids:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Connect agents in sequence
        for i in range(len(agent_ids) - 1):
            workflow.add_edge(agent_ids[i], agent_ids[i + 1])
        
        # Set entry point and end
        workflow.set_entry_point(agent_ids[0])
        workflow.add_edge(agent_ids[-1], END)
    
    def _create_parallel_workflow(self, workflow: StateGraph, agent_ids: List[str]) -> None:
        """Create a parallel workflow.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs to run in parallel
        """
        if not agent_ids:
            raise ValueError("At least one agent required for parallel workflow")
        
        # Add all agent nodes
        for agent_id in agent_ids:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        if len(agent_ids) == 1:
            # Single agent case
            workflow.set_entry_point(agent_ids[0])
            workflow.add_edge(agent_ids[0], END)
        else:
            # Create a dispatcher node for parallel execution
            def parallel_dispatcher(state: WorkflowState) -> WorkflowState:
                """Dispatch to all parallel agents."""
                state["parallel_agents"] = agent_ids[1:]  # Skip first agent
                return state
            
            workflow.add_node("dispatcher", parallel_dispatcher)
            workflow.set_entry_point("dispatcher")
            
            # Connect dispatcher to first agent
            workflow.add_edge("dispatcher", agent_ids[0])
            
            # Connect all agents to END
            for agent_id in agent_ids:
                workflow.add_edge(agent_id, END)
    
    def _create_hierarchical_workflow(self, workflow: StateGraph, agent_ids: List[str]) -> None:
        """Create a hierarchical workflow.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs (first is root, others are children)
        """
        if len(agent_ids) < 2:
            raise ValueError("At least two agents required for hierarchical workflow")
        
        root_agent = agent_ids[0]
        child_agents = agent_ids[1:]
        
        # Add all agent nodes
        for agent_id in agent_ids:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Root agent processes first
        workflow.set_entry_point(root_agent)
        
        # Create aggregator node for child results
        def child_aggregator(state: WorkflowState) -> WorkflowState:
            """Aggregate results from child agents."""
            responses = state.get("responses", [])
            child_responses = [r for r in responses if r["agent_id"] in child_agents]
            
            # Combine child responses
            combined_content = "\n\n".join([r["content"] for r in child_responses])
            
            # Add aggregated response
            responses.append({
                "agent_id": "aggregator",
                "content": f"Combined results:\n{combined_content}",
                "confidence_score": 1.0,
                "timestamp": datetime.utcnow().isoformat(),
                "step": state.get("current_step", 0) + 1,
            })
            state["responses"] = responses
            
            return state
        
        workflow.add_node("aggregator", child_aggregator)
        
        # Connect root to all children
        for child_agent in child_agents:
            workflow.add_edge(root_agent, child_agent)
        
        # Connect all children to aggregator
        for child_agent in child_agents:
            workflow.add_edge(child_agent, "aggregator")
        
        # Connect aggregator to END
        workflow.add_edge("aggregator", END)
    
    def _create_collaborative_workflow(self, workflow: StateGraph, agent_ids: List[str], max_iterations: int = 3) -> None:
        """Create a collaborative workflow with iterative refinement.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs for collaboration
            max_iterations: Maximum number of iterations
        """
        if not agent_ids:
            raise ValueError("At least one agent required for collaborative workflow")
        
        # Add all agent nodes
        for agent_id in agent_ids:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Create iteration controller
        def iteration_controller(state: WorkflowState) -> str:
            """Control iteration flow."""
            iteration_count = state.get("iteration_count", 0)
            max_iter = state.get("max_iterations", max_iterations)
            
            if iteration_count >= max_iter:
                return "end"
            else:
                # Determine next agent based on iteration
                next_agent_idx = iteration_count % len(agent_ids)
                return agent_ids[next_agent_idx]
        
        def increment_iteration(state: WorkflowState) -> WorkflowState:
            """Increment iteration counter."""
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            state["max_iterations"] = max_iterations
            return state
        
        workflow.add_node("controller", increment_iteration)
        workflow.set_entry_point("controller")
        
        # Add conditional edges from controller to agents
        workflow.add_conditional_edges(
            "controller",
            iteration_controller,
            {agent_id: agent_id for agent_id in agent_ids} | {"end": END}
        )
        
        # Connect all agents back to controller for next iteration
        for agent_id in agent_ids:
            workflow.add_edge(agent_id, "controller")
    
    def _create_iterative_workflow(self, workflow: StateGraph, agent_ids: List[str], max_iterations: int = 3) -> None:
        """Create an iterative improvement workflow.
        
        Args:
            workflow: StateGraph to populate
            agent_ids: List of agent IDs for iteration
            max_iterations: Maximum number of iterations
        """
        if len(agent_ids) < 2:
            raise ValueError("At least two agents required for iterative workflow")
        
        # Add all agent nodes
        for agent_id in agent_ids:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Create iteration logic
        def should_continue(state: WorkflowState) -> str:
            """Determine if iteration should continue."""
            iteration_count = state.get("iteration_count", 0)
            max_iter = state.get("max_iterations", max_iterations)
            
            if iteration_count >= max_iter:
                return "end"
            
            # Alternate between agents
            if iteration_count % 2 == 0:
                return agent_ids[0]  # First agent (e.g., developer)
            else:
                return agent_ids[1]  # Second agent (e.g., reviewer)
        
        def iteration_manager(state: WorkflowState) -> WorkflowState:
            """Manage iteration state."""
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            state["max_iterations"] = max_iterations
            return state
        
        workflow.add_node("iteration_manager", iteration_manager)
        workflow.set_entry_point("iteration_manager")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "iteration_manager",
            should_continue,
            {agent_ids[0]: agent_ids[0], agent_ids[1]: agent_ids[1], "end": END}
        )
        
        # Connect agents back to iteration manager
        for agent_id in agent_ids:
            workflow.add_edge(agent_id, "iteration_manager")
    
    def compile_workflow(self, workflow: StateGraph) -> CompiledStateGraph:
        """Compile a workflow for execution.
        
        Args:
            workflow: StateGraph to compile
            
        Returns:
            Compiled workflow ready for execution
        """
        try:
            compiled = workflow.compile()
            logger.debug("Successfully compiled workflow")
            return compiled
        except Exception as e:
            logger.error(f"Failed to compile workflow: {e}")
            raise
    
    def create_workflow_from_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        available_agents: Optional[List[str]] = None,
    ) -> Optional[StateGraph]:
        """Create a workflow dynamically based on question analysis.
        
        Args:
            question: Question to process
            context: Optional context information
            available_agents: List of available agent IDs
            
        Returns:
            Created workflow or None if no suitable template found
        """
        # Suggest appropriate template
        template = self.suggest_workflow_template(question, available_agents, context)
        
        if not template:
            logger.warning("No suitable workflow template found for question")
            return None
        
        # Create workflow specification
        workflow_id = f"dynamic_{uuid4().hex[:8]}"
        spec = WorkflowSpec(
            workflow_id=workflow_id,
            template=template,
            agent_ids=available_agents,
        )
        
        # Create and return workflow
        return self.create_workflow(spec)
    
    def clear_cache(self) -> None:
        """Clear workflow cache."""
        self._workflow_cache.clear()
        self._compiled_workflows.clear()
        logger.info("Cleared workflow cache")
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "templates_registered": len(self._templates),
            "workflows_cached": len(self._workflow_cache),
            "compiled_workflows": len(self._compiled_workflows),
            "available_agents": len(self.agents),
        }