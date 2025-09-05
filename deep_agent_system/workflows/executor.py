"""Workflow execution engine with state persistence and error handling."""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from deep_agent_system.models.messages import Message, MessageType
from deep_agent_system.workflows.manager import WorkflowState


logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CheckpointType(str, Enum):
    """Types of checkpoints."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"


class WorkflowCheckpoint:
    """Represents a workflow execution checkpoint."""
    
    def __init__(
        self,
        checkpoint_id: str,
        execution_id: str,
        state: WorkflowState,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize workflow checkpoint.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            execution_id: ID of the workflow execution
            state: Current workflow state
            checkpoint_type: Type of checkpoint
            metadata: Optional metadata
        """
        self.checkpoint_id = checkpoint_id
        self.execution_id = execution_id
        self.state = state.copy()
        self.checkpoint_type = checkpoint_type
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "state": dict(self.state),
            "checkpoint_type": self.checkpoint_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Deserialize checkpoint from dictionary."""
        checkpoint = cls(
            checkpoint_id=data["checkpoint_id"],
            execution_id=data["execution_id"],
            state=WorkflowState(**data["state"]),
            checkpoint_type=CheckpointType(data["checkpoint_type"]),
            metadata=data.get("metadata", {}),
        )
        if "created_at" in data:
            checkpoint.created_at = datetime.fromisoformat(data["created_at"])
        return checkpoint


class WorkflowExecution:
    """Represents a workflow execution instance."""
    
    def __init__(
        self,
        execution_id: str,
        workflow_id: str,
        compiled_workflow: CompiledStateGraph,
        initial_state: WorkflowState,
        timeout: int = 300,
    ):
        """Initialize workflow execution.
        
        Args:
            execution_id: Unique identifier for the execution
            workflow_id: ID of the workflow being executed
            compiled_workflow: Compiled LangGraph workflow
            initial_state: Initial state for execution
            timeout: Timeout in seconds
        """
        self.execution_id = execution_id
        self.workflow_id = workflow_id
        self.compiled_workflow = compiled_workflow
        self.current_state = initial_state
        self.timeout = timeout
        
        # Execution tracking
        self.status = ExecutionStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[Any] = None
        
        # Checkpoints
        self.checkpoints: List[WorkflowCheckpoint] = []
        self.auto_checkpoint_interval = 30  # seconds
        self.last_checkpoint_time: Optional[datetime] = None
        
        # Recovery
        self.retry_count = 0
        self.max_retries = 3
        self.recovery_strategies: List[str] = []
    
    def create_checkpoint(
        self,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowCheckpoint:
        """Create a checkpoint of the current state.
        
        Args:
            checkpoint_type: Type of checkpoint
            metadata: Optional metadata
            
        Returns:
            Created checkpoint
        """
        checkpoint_id = f"checkpoint_{len(self.checkpoints) + 1}_{uuid4().hex[:8]}"
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            execution_id=self.execution_id,
            state=self.current_state,
            checkpoint_type=checkpoint_type,
            metadata=metadata,
        )
        
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_time = datetime.utcnow()
        
        logger.debug(f"Created checkpoint {checkpoint_id} for execution {self.execution_id}")
        return checkpoint
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore execution state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore from
            
        Returns:
            True if restored successfully, False otherwise
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                self.current_state = checkpoint.state.copy()
                logger.info(f"Restored execution {self.execution_id} from checkpoint {checkpoint_id}")
                return True
        
        logger.error(f"Checkpoint {checkpoint_id} not found for execution {self.execution_id}")
        return False
    
    def get_latest_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        """Get the most recent checkpoint.
        
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        return self.checkpoints[-1] if self.checkpoints else None
    
    def should_create_checkpoint(self) -> bool:
        """Determine if a checkpoint should be created.
        
        Returns:
            True if checkpoint should be created
        """
        if not self.last_checkpoint_time:
            return True
        
        time_since_last = datetime.utcnow() - self.last_checkpoint_time
        return time_since_last.total_seconds() >= self.auto_checkpoint_interval
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution to dictionary."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "current_state": dict(self.current_state),
            "timeout": self.timeout,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }


class WorkflowExecutor:
    """Executes workflows with state persistence and error handling."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        max_concurrent_executions: int = 10,
        auto_checkpoint: bool = True,
    ):
        """Initialize the workflow executor.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_concurrent_executions: Maximum concurrent executions
            auto_checkpoint: Whether to automatically create checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_concurrent_executions = max_concurrent_executions
        self.auto_checkpoint = auto_checkpoint
        
        # Active executions
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Execution history
        self._execution_history: List[WorkflowExecution] = []
        
        # Statistics
        self._stats = {
            "executions_started": 0,
            "executions_completed": 0,
            "executions_failed": 0,
            "executions_cancelled": 0,
            "checkpoints_created": 0,
            "recoveries_performed": 0,
        }
        
        logger.info("WorkflowExecutor initialized")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        compiled_workflow: CompiledStateGraph,
        initial_message: Message,
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> WorkflowExecution:
        """Execute a compiled workflow.
        
        Args:
            workflow_id: ID of the workflow
            compiled_workflow: Compiled LangGraph workflow
            initial_message: Initial message to process
            context: Optional context
            timeout: Execution timeout in seconds
            
        Returns:
            WorkflowExecution object
            
        Raises:
            ValueError: If max concurrent executions exceeded
        """
        if len(self._active_executions) >= self.max_concurrent_executions:
            raise ValueError("Maximum concurrent executions limit reached")
        
        # Create execution
        execution_id = str(uuid4())
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
            status=ExecutionStatus.PENDING.value,
        )
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            compiled_workflow=compiled_workflow,
            initial_state=initial_state,
            timeout=timeout,
        )
        
        # Register execution
        self._active_executions[execution_id] = execution
        self._stats["executions_started"] += 1
        
        # Create execution task
        task = asyncio.create_task(self._execute_workflow_task(execution))
        self._execution_tasks[execution_id] = task
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution
    
    async def _execute_workflow_task(self, execution: WorkflowExecution) -> None:
        """Execute workflow in a task.
        
        Args:
            execution: Workflow execution to run
        """
        execution_id = execution.execution_id
        
        try:
            # Update status
            execution.status = ExecutionStatus.RUNNING
            execution.started_at = datetime.utcnow()
            
            # Create initial checkpoint
            if self.auto_checkpoint:
                execution.create_checkpoint(CheckpointType.AUTOMATIC)
                self._stats["checkpoints_created"] += 1
            
            # Execute workflow with timeout
            result = await asyncio.wait_for(
                self._run_workflow_with_checkpoints(execution),
                timeout=execution.timeout
            )
            
            # Update execution with results
            execution.current_state = result
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.result = result
            
            self._stats["executions_completed"] += 1
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except asyncio.TimeoutError:
            execution.status = ExecutionStatus.TIMEOUT
            execution.error = f"Workflow timed out after {execution.timeout} seconds"
            execution.completed_at = datetime.utcnow()
            self._stats["executions_failed"] += 1
            logger.error(f"Workflow execution timed out: {execution_id}")
            
        except asyncio.CancelledError:
            execution.status = ExecutionStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            self._stats["executions_cancelled"] += 1
            logger.info(f"Workflow execution cancelled: {execution_id}")
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            self._stats["executions_failed"] += 1
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            
            # Attempt recovery
            if execution.retry_count < execution.max_retries:
                await self._attempt_recovery(execution, e)
            
        finally:
            # Clean up
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            if execution_id in self._execution_tasks:
                del self._execution_tasks[execution_id]
            
            # Move to history
            self._execution_history.append(execution)
            
            # Save final checkpoint
            if self.auto_checkpoint:
                execution.create_checkpoint(CheckpointType.AUTOMATIC)
                await self._save_checkpoint_to_disk(execution.get_latest_checkpoint())
    
    async def _run_workflow_with_checkpoints(self, execution: WorkflowExecution) -> WorkflowState:
        """Run workflow with automatic checkpointing.
        
        Args:
            execution: Workflow execution
            
        Returns:
            Final workflow state
        """
        current_state = execution.current_state
        
        # Execute the compiled workflow
        async for state in execution.compiled_workflow.astream(current_state):
            # Update current state
            execution.current_state = state
            
            # Create checkpoint if needed
            if self.auto_checkpoint and execution.should_create_checkpoint():
                execution.create_checkpoint(CheckpointType.AUTOMATIC)
                self._stats["checkpoints_created"] += 1
                
                # Save checkpoint to disk periodically
                if len(execution.checkpoints) % 5 == 0:  # Every 5 checkpoints
                    await self._save_checkpoint_to_disk(execution.get_latest_checkpoint())
        
        return execution.current_state
    
    async def _attempt_recovery(self, execution: WorkflowExecution, error: Exception) -> None:
        """Attempt to recover from execution error.
        
        Args:
            execution: Failed execution
            error: The error that occurred
        """
        execution.retry_count += 1
        logger.info(f"Attempting recovery for execution {execution.execution_id} (retry {execution.retry_count})")
        
        # Try to restore from latest checkpoint
        latest_checkpoint = execution.get_latest_checkpoint()
        if latest_checkpoint:
            execution.restore_from_checkpoint(latest_checkpoint.checkpoint_id)
            execution.recovery_strategies.append(f"restored_from_checkpoint_{latest_checkpoint.checkpoint_id}")
            
            # Create recovery checkpoint
            execution.create_checkpoint(
                CheckpointType.ERROR_RECOVERY,
                metadata={"error": str(error), "retry_count": execution.retry_count}
            )
            
            self._stats["recoveries_performed"] += 1
            
            # Restart execution task
            task = asyncio.create_task(self._execute_workflow_task(execution))
            self._execution_tasks[execution.execution_id] = task
    
    async def _save_checkpoint_to_disk(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint to save
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.execution_id}_{checkpoint.checkpoint_id}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            
            logger.debug(f"Saved checkpoint to disk: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint to disk: {e}")
    
    async def load_checkpoint_from_disk(self, execution_id: str, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load checkpoint from disk.
        
        Args:
            execution_id: Execution ID
            checkpoint_id: Checkpoint ID
            
        Returns:
            Loaded checkpoint or None if not found
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{execution_id}_{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            return WorkflowCheckpoint.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from disk: {e}")
            return None
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution.
        
        Args:
            execution_id: ID of execution to pause
            
        Returns:
            True if paused successfully, False otherwise
        """
        if execution_id not in self._active_executions:
            return False
        
        execution = self._active_executions[execution_id]
        
        if execution.status == ExecutionStatus.RUNNING:
            execution.status = ExecutionStatus.PAUSED
            
            # Create pause checkpoint
            execution.create_checkpoint(
                CheckpointType.MANUAL,
                metadata={"action": "paused"}
            )
            
            logger.info(f"Paused execution: {execution_id}")
            return True
        
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution.
        
        Args:
            execution_id: ID of execution to resume
            
        Returns:
            True if resumed successfully, False otherwise
        """
        if execution_id not in self._active_executions:
            return False
        
        execution = self._active_executions[execution_id]
        
        if execution.status == ExecutionStatus.PAUSED:
            execution.status = ExecutionStatus.RUNNING
            logger.info(f"Resumed execution: {execution_id}")
            return True
        
        return False
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if execution_id in self._execution_tasks:
            task = self._execution_tasks[execution_id]
            task.cancel()
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        
        return False
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active executions.
        
        Returns:
            List of active executions
        """
        return list(self._active_executions.values())
    
    def get_execution_history(self, limit: int = 50) -> List[WorkflowExecution]:
        """Get execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of historical executions
        """
        return self._execution_history[-limit:] if limit > 0 else self._execution_history
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """Get execution status.
        
        Args:
            execution_id: ID of execution
            
        Returns:
            Execution status or None if not found
        """
        # Check active executions
        if execution_id in self._active_executions:
            return self._active_executions[execution_id].status
        
        # Check history
        for execution in self._execution_history:
            if execution.execution_id == execution_id:
                return execution.status
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            **self._stats,
            "active_executions": len(self._active_executions),
            "total_executions": len(self._execution_history),
        }
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """Clean up old checkpoint files.
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of files cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                if checkpoint_file.stat().st_mtime < cutoff_date.timestamp():
                    checkpoint_file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old checkpoint files")
            
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")
        
        return cleaned_count
    
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down WorkflowExecutor")
        
        # Cancel all active tasks
        for task in self._execution_tasks.values():
            if not task.done():
                task.cancel()
        
        # Clear data structures
        self._active_executions.clear()
        self._execution_tasks.clear()
        
        logger.info("WorkflowExecutor shutdown complete")