"""Agent coordinator for managing multi-agent interactions and workflows."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.communication.manager import AgentCommunicationManager
from deep_agent_system.models.messages import Message, MessageType, Response


logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CoordinationStrategy(str, Enum):
    """Strategies for coordinating multiple agents."""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"     # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Agents work in a tree structure
    COLLABORATIVE = "collaborative"  # Agents collaborate dynamically


class WorkflowState(Dict[str, Any]):
    """State object for LangGraph workflows."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure required fields exist
        if "messages" not in self:
            self["messages"] = []
        if "responses" not in self:
            self["responses"] = []
        if "current_agent" not in self:
            self["current_agent"] = None
        if "workflow_id" not in self:
            self["workflow_id"] = str(uuid4())
        if "status" not in self:
            self["status"] = WorkflowStatus.PENDING
        if "metadata" not in self:
            self["metadata"] = {}


class WorkflowDefinition:
    """Definition of a multi-agent workflow."""
    
    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str,
        agent_sequence: List[str],
        coordination_strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL,
        conditions: Optional[Dict[str, Callable]] = None,
        timeout: int = 300,
    ):
        """Initialize workflow definition.
        
        Args:
            workflow_id: Unique identifier for the workflow
            name: Human-readable name
            description: Description of the workflow
            agent_sequence: List of agent IDs in execution order
            coordination_strategy: How agents should be coordinated
            conditions: Optional conditions for workflow routing
            timeout: Timeout in seconds for workflow execution
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.agent_sequence = agent_sequence
        self.coordination_strategy = coordination_strategy
        self.conditions = conditions or {}
        self.timeout = timeout
        self.created_at = datetime.utcnow()


class WorkflowExecution:
    """Represents an active workflow execution."""
    
    def __init__(
        self,
        execution_id: str,
        workflow_definition: WorkflowDefinition,
        initial_state: WorkflowState,
    ):
        """Initialize workflow execution.
        
        Args:
            execution_id: Unique identifier for this execution
            workflow_definition: The workflow definition to execute
            initial_state: Initial state for the workflow
        """
        self.execution_id = execution_id
        self.workflow_definition = workflow_definition
        self.state = initial_state
        self.status = WorkflowStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[Any] = None


class AgentCoordinator:
    """Coordinates multi-agent interactions and workflow management."""
    
    def __init__(
        self,
        communication_manager: AgentCommunicationManager,
        max_concurrent_workflows: int = 10,
    ):
        """Initialize the agent coordinator.
        
        Args:
            communication_manager: Manager for agent communication
            max_concurrent_workflows: Maximum number of concurrent workflows
        """
        self.communication_manager = communication_manager
        self.max_concurrent_workflows = max_concurrent_workflows
        
        # Workflow management
        self._workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_history: List[WorkflowExecution] = []
        
        # LangGraph compiled workflows cache
        self._compiled_workflows: Dict[str, CompiledStateGraph] = {}
        
        # Coordination state
        self._coordination_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info("AgentCoordinator initialized")
    
    def register_workflow(self, workflow_definition: WorkflowDefinition) -> None:
        """Register a workflow definition.
        
        Args:
            workflow_definition: The workflow to register
        """
        workflow_id = workflow_definition.workflow_id
        
        if workflow_id in self._workflow_definitions:
            logger.warning(f"Workflow {workflow_id} already registered, updating")
        
        self._workflow_definitions[workflow_id] = workflow_definition
        
        # Clear compiled workflow cache to force recompilation
        if workflow_id in self._compiled_workflows:
            del self._compiled_workflows[workflow_id]
        
        logger.info(f"Registered workflow: {workflow_definition.name} ({workflow_id})")
    
    def unregister_workflow(self, workflow_id: str) -> None:
        """Unregister a workflow definition.
        
        Args:
            workflow_id: ID of the workflow to unregister
        """
        if workflow_id in self._workflow_definitions:
            del self._workflow_definitions[workflow_id]
            
        if workflow_id in self._compiled_workflows:
            del self._compiled_workflows[workflow_id]
            
        logger.info(f"Unregistered workflow: {workflow_id}")
    
    def get_workflow_definitions(self) -> List[WorkflowDefinition]:
        """Get all registered workflow definitions.
        
        Returns:
            List of workflow definitions
        """
        return list(self._workflow_definitions.values())
    
    def _create_langgraph_workflow(self, workflow_def: WorkflowDefinition) -> StateGraph:
        """Create a LangGraph workflow from a workflow definition.
        
        Args:
            workflow_def: The workflow definition
            
        Returns:
            LangGraph StateGraph
        """
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_id in workflow_def.agent_sequence:
            workflow.add_node(agent_id, self._create_agent_node(agent_id))
        
        # Add edges based on coordination strategy
        if workflow_def.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
            self._add_sequential_edges(workflow, workflow_def.agent_sequence)
        elif workflow_def.coordination_strategy == CoordinationStrategy.PARALLEL:
            self._add_parallel_edges(workflow, workflow_def.agent_sequence)
        elif workflow_def.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            self._add_hierarchical_edges(workflow, workflow_def.agent_sequence)
        elif workflow_def.coordination_strategy == CoordinationStrategy.COLLABORATIVE:
            self._add_collaborative_edges(workflow, workflow_def.agent_sequence)
        
        # Set entry point
        if workflow_def.agent_sequence:
            workflow.set_entry_point(workflow_def.agent_sequence[0])
        
        return workflow
    
    def _create_agent_node(self, agent_id: str) -> Callable:
        """Create a node function for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Node function for LangGraph
        """
        async def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute agent processing within the workflow."""
            try:
                logger.debug(f"Executing agent node: {agent_id}")
                logger.debug(f"Current state: {state}")
                
                # Update current agent in state
                state["current_agent"] = agent_id
                
                # Get the latest message to process
                messages = state.get("messages", [])
                logger.debug(f"Messages in state: {messages}")
                
                if not messages:
                    logger.warning(f"No messages to process for agent {agent_id}")
                    return state
                
                latest_message = messages[-1]
                logger.debug(f"Processing message: {latest_message}")
                
                # Create a proper Message object for the agent
                message_content = latest_message.get("content", "")
                if isinstance(latest_message, dict):
                    # Convert dict to Message if needed
                    agent_message = Message(
                        sender_id="coordinator",
                        recipient_id=agent_id,
                        content=message_content,
                        message_type=MessageType.AGENT_REQUEST,
                    )
                else:
                    # Use the message directly if it's already a Message object
                    agent_message = latest_message
                
                # Get the agent directly and process the message
                if not self.communication_manager.is_agent_registered(agent_id):
                    logger.error(f"Agent {agent_id} not registered with communication manager")
                    return state
                
                # Get the agent from the communication manager's registry
                agent = self.communication_manager._agents.get(agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found in registry")
                    return state
                
                # Process message directly with the agent
                response = agent.process_message(agent_message)
                
                if response:
                    # Add response to state
                    responses = state.get("responses", [])
                    responses.append({
                        "agent_id": agent_id,
                        "content": response.content,
                        "confidence_score": response.confidence_score,
                        "timestamp": response.timestamp.isoformat(),
                    })
                    state["responses"] = responses
                    
                    logger.debug(f"Agent {agent_id} completed processing")
                else:
                    logger.error(f"No response received from agent {agent_id}")
                
                return state
                
            except Exception as e:
                logger.error(f"Error in agent node {agent_id}: {e}")
                state["status"] = WorkflowStatus.FAILED
                state["error"] = str(e)
                return state
        
        return agent_node
    
    def _add_sequential_edges(self, workflow: StateGraph, agent_sequence: List[str]) -> None:
        """Add sequential edges to the workflow.
        
        Args:
            workflow: The StateGraph to modify
            agent_sequence: List of agent IDs in sequence
        """
        for i in range(len(agent_sequence) - 1):
            workflow.add_edge(agent_sequence[i], agent_sequence[i + 1])
        
        # Last agent goes to END
        if agent_sequence:
            workflow.add_edge(agent_sequence[-1], END)
    
    def _add_parallel_edges(self, workflow: StateGraph, agent_sequence: List[str]) -> None:
        """Add parallel edges to the workflow.
        
        Args:
            workflow: The StateGraph to modify
            agent_sequence: List of agent IDs to run in parallel
        """
        # For parallel execution, all agents start from the first one
        # and all end at END (simplified parallel model)
        if len(agent_sequence) > 1:
            first_agent = agent_sequence[0]
            for agent_id in agent_sequence[1:]:
                workflow.add_edge(first_agent, agent_id)
                workflow.add_edge(agent_id, END)
            workflow.add_edge(first_agent, END)
        elif agent_sequence:
            workflow.add_edge(agent_sequence[0], END)
    
    def _add_hierarchical_edges(self, workflow: StateGraph, agent_sequence: List[str]) -> None:
        """Add hierarchical edges to the workflow.
        
        Args:
            workflow: The StateGraph to modify
            agent_sequence: List of agent IDs in hierarchical order
        """
        # For hierarchical, assume first agent is root, others are children
        if len(agent_sequence) > 1:
            root_agent = agent_sequence[0]
            for child_agent in agent_sequence[1:]:
                workflow.add_edge(root_agent, child_agent)
                workflow.add_edge(child_agent, END)
            workflow.add_edge(root_agent, END)
        elif agent_sequence:
            workflow.add_edge(agent_sequence[0], END)
    
    def _add_collaborative_edges(self, workflow: StateGraph, agent_sequence: List[str]) -> None:
        """Add collaborative edges to the workflow.
        
        Args:
            workflow: The StateGraph to modify
            agent_sequence: List of agent IDs for collaboration
        """
        # For collaborative, create a mesh where agents can communicate
        # Start with sequential as base, but allow dynamic routing
        self._add_sequential_edges(workflow, agent_sequence)
    
    async def _execute_coordination_strategy(
        self,
        workflow_def: WorkflowDefinition,
        initial_message: Message,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute coordination strategy without LangGraph complexity.
        
        Args:
            workflow_def: Workflow definition
            initial_message: Initial message
            state: Workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            if workflow_def.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
                return await self._execute_sequential(workflow_def, initial_message, state)
            elif workflow_def.coordination_strategy == CoordinationStrategy.PARALLEL:
                return await self._execute_parallel(workflow_def, initial_message, state)
            elif workflow_def.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
                return await self._execute_hierarchical(workflow_def, initial_message, state)
            elif workflow_def.coordination_strategy == CoordinationStrategy.COLLABORATIVE:
                return await self._execute_collaborative(workflow_def, initial_message, state)
            else:
                raise ValueError(f"Unknown coordination strategy: {workflow_def.coordination_strategy}")
        
        except Exception as e:
            state["status"] = WorkflowStatus.FAILED
            state["error"] = str(e)
            return state
    
    async def _execute_sequential(
        self,
        workflow_def: WorkflowDefinition,
        initial_message: Message,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute agents sequentially."""
        current_message = initial_message
        
        for agent_id in workflow_def.agent_sequence:
            if not self.communication_manager.is_agent_registered(agent_id):
                logger.error(f"Agent {agent_id} not registered")
                continue
            
            # Get agent and process message
            agent = self.communication_manager._agents[agent_id]
            response = agent.process_message(current_message)
            
            # Add response to state
            responses = state.get("responses", [])
            responses.append({
                "agent_id": agent_id,
                "content": response.content,
                "confidence_score": response.confidence_score,
                "timestamp": response.timestamp.isoformat(),
            })
            state["responses"] = responses
            
            # Create next message from response for next agent
            if response.content:
                current_message = Message(
                    sender_id=agent_id,
                    recipient_id="next_agent",
                    content=response.content,
                    message_type=MessageType.AGENT_RESPONSE,
                )
        
        return state
    
    async def _execute_parallel(
        self,
        workflow_def: WorkflowDefinition,
        initial_message: Message,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute agents in parallel."""
        tasks = []
        
        for agent_id in workflow_def.agent_sequence:
            if not self.communication_manager.is_agent_registered(agent_id):
                logger.error(f"Agent {agent_id} not registered")
                continue
            
            # Create task for each agent
            task = asyncio.create_task(self._process_agent_message(agent_id, initial_message))
            tasks.append((agent_id, task))
        
        # Wait for all tasks to complete
        responses = state.get("responses", [])
        for agent_id, task in tasks:
            try:
                response = await task
                if response:
                    responses.append({
                        "agent_id": agent_id,
                        "content": response.content,
                        "confidence_score": response.confidence_score,
                        "timestamp": response.timestamp.isoformat(),
                    })
            except Exception as e:
                logger.error(f"Error processing agent {agent_id}: {e}")
        
        state["responses"] = responses
        return state
    
    async def _execute_hierarchical(
        self,
        workflow_def: WorkflowDefinition,
        initial_message: Message,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute agents hierarchically (root first, then children)."""
        if not workflow_def.agent_sequence:
            return state
        
        # First agent is the root
        root_agent_id = workflow_def.agent_sequence[0]
        child_agents = workflow_def.agent_sequence[1:]
        
        responses = state.get("responses", [])
        
        # Process root agent first
        if self.communication_manager.is_agent_registered(root_agent_id):
            root_response = await self._process_agent_message(root_agent_id, initial_message)
            if root_response:
                responses.append({
                    "agent_id": root_agent_id,
                    "content": root_response.content,
                    "confidence_score": root_response.confidence_score,
                    "timestamp": root_response.timestamp.isoformat(),
                })
                
                # Create message for child agents based on root response
                child_message = Message(
                    sender_id=root_agent_id,
                    recipient_id="child_agents",
                    content=root_response.content,
                    message_type=MessageType.AGENT_RESPONSE,
                )
                
                # Process child agents in parallel
                child_tasks = []
                for child_agent_id in child_agents:
                    if self.communication_manager.is_agent_registered(child_agent_id):
                        task = asyncio.create_task(self._process_agent_message(child_agent_id, child_message))
                        child_tasks.append((child_agent_id, task))
                
                # Wait for child agents
                for child_agent_id, task in child_tasks:
                    try:
                        child_response = await task
                        if child_response:
                            responses.append({
                                "agent_id": child_agent_id,
                                "content": child_response.content,
                                "confidence_score": child_response.confidence_score,
                                "timestamp": child_response.timestamp.isoformat(),
                            })
                    except Exception as e:
                        logger.error(f"Error processing child agent {child_agent_id}: {e}")
        
        state["responses"] = responses
        return state
    
    async def _execute_collaborative(
        self,
        workflow_def: WorkflowDefinition,
        initial_message: Message,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute agents collaboratively (simplified as sequential for now)."""
        # For now, implement collaborative as sequential
        # This can be enhanced later with more complex collaboration logic
        return await self._execute_sequential(workflow_def, initial_message, state)
    
    async def _process_agent_message(self, agent_id: str, message: Message) -> Optional[Response]:
        """Process a message with a specific agent.
        
        Args:
            agent_id: ID of the agent to process the message
            message: Message to process
            
        Returns:
            Response from the agent or None if error
        """
        try:
            if not self.communication_manager.is_agent_registered(agent_id):
                logger.error(f"Agent {agent_id} not registered")
                return None
            
            agent = self.communication_manager._agents[agent_id]
            return agent.process_message(message)
            
        except Exception as e:
            logger.error(f"Error processing message with agent {agent_id}: {e}")
            return None
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_message: Message,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Execute a workflow with the given initial message.
        
        Args:
            workflow_id: ID of the workflow to execute
            initial_message: Initial message to start the workflow
            context: Optional context for the workflow
            
        Returns:
            WorkflowExecution object
            
        Raises:
            ValueError: If workflow is not registered or max concurrent limit reached
        """
        if workflow_id not in self._workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not registered")
        
        if len(self._active_executions) >= self.max_concurrent_workflows:
            raise ValueError("Maximum concurrent workflows limit reached")
        
        workflow_def = self._workflow_definitions[workflow_id]
        execution_id = str(uuid4())
        
        # Create initial state
        initial_state = WorkflowState(
            workflow_id=workflow_id,
            execution_id=execution_id,
            messages=[{
                "id": initial_message.id,
                "content": initial_message.content,
                "sender_id": initial_message.sender_id,
                "message_type": initial_message.message_type.value,
                "timestamp": initial_message.timestamp.isoformat(),
            }],
            context=context or {},
            status=WorkflowStatus.PENDING,
        )
        
        # Create execution object
        execution = WorkflowExecution(execution_id, workflow_def, initial_state)
        self._active_executions[execution_id] = execution
        
        try:
            # Start execution
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            logger.info(f"Starting workflow execution: {execution_id}")
            
            # Execute workflow based on coordination strategy with timeout
            result = await asyncio.wait_for(
                self._execute_coordination_strategy(workflow_def, initial_message, initial_state),
                timeout=workflow_def.timeout
            )
            
            # Update execution with results
            execution.state = result
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.result = result
            
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except asyncio.TimeoutError:
            execution.status = WorkflowStatus.FAILED
            execution.error = f"Workflow timed out after {workflow_def.timeout} seconds"
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution timed out: {execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            
        finally:
            # Move to history and clean up
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            self._execution_history.append(execution)
            
            # Keep only last 100 executions in history
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]
        
        return execution
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel an active workflow execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            True if cancelled successfully, False if not found
        """
        if execution_id not in self._active_executions:
            return False
        
        execution = self._active_executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        
        # Move to history
        del self._active_executions[execution_id]
        self._execution_history.append(execution)
        
        logger.info(f"Cancelled workflow execution: {execution_id}")
        return True
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active workflow executions.
        
        Returns:
            List of active executions
        """
        return list(self._active_executions.values())
    
    def get_execution_history(self, limit: int = 50) -> List[WorkflowExecution]:
        """Get workflow execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of historical executions
        """
        return self._execution_history[-limit:] if limit > 0 else self._execution_history
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowStatus]:
        """Get the status of a workflow execution.
        
        Args:
            execution_id: ID of the execution
            
        Returns:
            Status of the execution or None if not found
        """
        # Check active executions
        if execution_id in self._active_executions:
            return self._active_executions[execution_id].status
        
        # Check history
        for execution in self._execution_history:
            if execution.execution_id == execution_id:
                return execution.status
        
        return None
    
    async def coordinate_agents(
        self,
        agent_ids: List[str],
        task_message: Message,
        strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL,
        timeout: int = 300,
    ) -> List[Response]:
        """Coordinate multiple agents for a specific task.
        
        Args:
            agent_ids: List of agent IDs to coordinate
            task_message: Task message to process
            strategy: Coordination strategy to use
            timeout: Timeout for coordination
            
        Returns:
            List of responses from agents
        """
        # Create a temporary workflow for this coordination
        workflow_id = f"temp_coordination_{uuid4().hex[:8]}"
        workflow_def = WorkflowDefinition(
            workflow_id=workflow_id,
            name="Temporary Coordination",
            description=f"Coordinate agents: {', '.join(agent_ids)}",
            agent_sequence=agent_ids,
            coordination_strategy=strategy,
            timeout=timeout,
        )
        
        try:
            # Register and execute the temporary workflow
            self.register_workflow(workflow_def)
            execution = await self.execute_workflow(workflow_id, task_message)
            
            # Extract responses from execution result
            responses = []
            if execution.result and "responses" in execution.result:
                for response_data in execution.result["responses"]:
                    response = Response(
                        message_id=task_message.id,
                        agent_id=response_data["agent_id"],
                        content=response_data["content"],
                        confidence_score=response_data.get("confidence_score", 1.0),
                    )
                    responses.append(response)
            
            return responses
            
        finally:
            # Clean up temporary workflow
            self.unregister_workflow(workflow_id)
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics.
        
        Returns:
            Dictionary containing coordination statistics
        """
        total_executions = len(self._execution_history) + len(self._active_executions)
        completed_executions = len([e for e in self._execution_history if e.status == WorkflowStatus.COMPLETED])
        failed_executions = len([e for e in self._execution_history if e.status == WorkflowStatus.FAILED])
        
        return {
            "total_workflows": len(self._workflow_definitions),
            "active_executions": len(self._active_executions),
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "success_rate": completed_executions / total_executions if total_executions > 0 else 0.0,
        }
    
    def shutdown(self) -> None:
        """Shutdown the coordinator and clean up resources."""
        logger.info("Shutting down AgentCoordinator")
        
        # Cancel all active executions
        for execution_id in list(self._active_executions.keys()):
            asyncio.create_task(self.cancel_workflow(execution_id))
        
        # Clear all data structures
        self._workflow_definitions.clear()
        self._active_executions.clear()
        self._compiled_workflows.clear()
        self._coordination_locks.clear()
        
        logger.info("AgentCoordinator shutdown complete")