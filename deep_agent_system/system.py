"""Main AgentSystem orchestrator that coordinates all components."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_agent_system.agents.analyst import AnalystAgent
from deep_agent_system.agents.architect import ArchitectAgent
from deep_agent_system.agents.base import BaseAgent
from deep_agent_system.agents.code_reviewer import CodeReviewerAgent
from deep_agent_system.agents.developer import DeveloperAgent
from deep_agent_system.agents.tester import TesterAgent
from deep_agent_system.communication.coordinator import AgentCoordinator
from deep_agent_system.communication.manager import AgentCommunicationManager
from deep_agent_system.config.manager import ConfigManager, ConfigurationError
from deep_agent_system.config.models import AgentType, SystemConfig
from deep_agent_system.models.messages import Message, MessageType, Response
from deep_agent_system.prompts.manager import PromptManager
from deep_agent_system.rag.manager import RAGManager
from deep_agent_system.workflows.executor import WorkflowExecutor
from deep_agent_system.workflows.manager import WorkflowManager


logger = logging.getLogger(__name__)


class AgentSystemError(Exception):
    """Base exception for AgentSystem errors."""
    pass


class AgentSystem:
    """Main orchestrator that coordinates all system components."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the AgentSystem.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self.config: Optional[SystemConfig] = None
        
        # Core components
        self.config_manager: Optional[ConfigManager] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.rag_manager: Optional[RAGManager] = None
        self.communication_manager: Optional[AgentCommunicationManager] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.workflow_manager: Optional[WorkflowManager] = None
        self.workflow_executor: Optional[WorkflowExecutor] = None
        
        # Agents registry
        self.agents: Dict[str, BaseAgent] = {}
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        logger.info("AgentSystem created")
    
    async def initialize(self) -> None:
        """Initialize all system components.
        
        Raises:
            AgentSystemError: If initialization fails
        """
        try:
            logger.info("Initializing AgentSystem...")
            
            # 1. Load configuration
            await self._initialize_configuration()
            
            # 2. Initialize prompt management
            await self._initialize_prompt_manager()
            
            # 3. Initialize RAG system
            await self._initialize_rag_manager()
            
            # 4. Initialize communication system
            await self._initialize_communication()
            
            # 5. Initialize workflow system
            await self._initialize_workflow_system()
            
            # 6. Initialize and register agents
            await self._initialize_agents()
            
            # 7. Setup agent coordination
            await self._setup_agent_coordination()
            
            self.is_initialized = True
            logger.info("AgentSystem initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentSystem: {e}")
            raise AgentSystemError(f"Initialization failed: {e}") from e
    
    async def _initialize_configuration(self) -> None:
        """Initialize configuration management."""
        logger.debug("Initializing configuration...")
        
        self.config_manager = ConfigManager(self.config_file)
        self.config = self.config_manager.load_config()
        
        # Validate configuration
        self.config_manager.validate_config(self.config)
        
        logger.info("Configuration loaded and validated")
    
    async def _initialize_prompt_manager(self) -> None:
        """Initialize prompt management."""
        logger.debug("Initializing prompt manager...")
        
        if not self.config:
            raise AgentSystemError("Configuration not loaded")
        
        self.prompt_manager = PromptManager(self.config.prompt_dir)
        
        # Load and validate prompts
        prompts = self.prompt_manager.load_prompts()
        logger.info(f"Loaded {len(prompts)} prompt templates")
    
    async def _initialize_rag_manager(self) -> None:
        """Initialize RAG system."""
        logger.debug("Initializing RAG manager...")
        
        if not self.config:
            raise AgentSystemError("Configuration not loaded")
        
        # Only initialize if RAG is enabled
        if (self.config.rag_config.vector_store_enabled or 
            self.config.rag_config.graph_store_enabled):
            
            self.rag_manager = RAGManager(
                rag_config=self.config.rag_config,
                database_config=self.config.database_config,
            )
            
            # Initialize RAG components
            await self.rag_manager.initialize()
            logger.info("RAG manager initialized")
        else:
            logger.info("RAG disabled in configuration")
    
    async def _initialize_communication(self) -> None:
        """Initialize communication system."""
        logger.debug("Initializing communication system...")
        
        self.communication_manager = AgentCommunicationManager()
        logger.info("Communication manager initialized")
    
    async def _initialize_workflow_system(self) -> None:
        """Initialize workflow management system."""
        logger.debug("Initializing workflow system...")
        
        if not self.config:
            raise AgentSystemError("Configuration not loaded")
        
        # Initialize workflow executor
        checkpoint_dir = self.config.workflow_config.checkpoint_dir
        self.workflow_executor = WorkflowExecutor(
            checkpoint_dir=checkpoint_dir,
            auto_checkpoint=self.config.workflow_config.enable_checkpoints,
        )
        
        # Workflow manager will be initialized after agents are created
        logger.info("Workflow executor initialized")
    
    async def _initialize_agents(self) -> None:
        """Initialize and register all agents."""
        logger.debug("Initializing agents...")
        
        if not self.config or not self.prompt_manager:
            raise AgentSystemError("Required components not initialized")
        
        # Create agents based on configuration
        for agent_id, agent_config in self.config.agents.items():
            agent = await self._create_agent(agent_config)
            if agent:
                self.agents[agent_id] = agent
                
                # Register with communication manager
                if self.communication_manager:
                    self.communication_manager.register_agent(agent)
                
                logger.info(f"Initialized agent: {agent_id} ({agent_config.agent_type.value})")
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def _create_agent(self, agent_config) -> Optional[BaseAgent]:
        """Create an agent instance based on configuration.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Created agent instance or None if creation failed
        """
        try:
            agent_type = agent_config.agent_type
            
            # Create agent based on type
            if agent_type == AgentType.ANALYST:
                agent = AnalystAgent(
                    config=agent_config,
                    prompt_manager=self.prompt_manager,
                    rag_manager=self.rag_manager,
                    system_config=self.config,
                )
            elif agent_type == AgentType.ARCHITECT:
                agent = ArchitectAgent(
                    config=agent_config,
                    prompt_manager=self.prompt_manager,
                    rag_manager=self.rag_manager,
                    system_config=self.config,
                )
            elif agent_type == AgentType.DEVELOPER:
                agent = DeveloperAgent(
                    config=agent_config,
                    prompt_manager=self.prompt_manager,
                    rag_manager=self.rag_manager,
                    system_config=self.config,
                )
            elif agent_type == AgentType.CODE_REVIEWER:
                agent = CodeReviewerAgent(
                    config=agent_config,
                    prompt_manager=self.prompt_manager,
                    rag_manager=self.rag_manager,
                    system_config=self.config,
                )
            elif agent_type == AgentType.TESTER:
                agent = TesterAgent(
                    config=agent_config,
                    prompt_manager=self.prompt_manager,
                    rag_manager=self.rag_manager,
                    system_config=self.config,
                )
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return None
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_config.agent_id}: {e}")
            return None
    
    async def _setup_agent_coordination(self) -> None:
        """Setup agent coordination system."""
        logger.debug("Setting up agent coordination...")
        
        if not self.communication_manager:
            raise AgentSystemError("Communication manager not initialized")
        
        # Initialize agent coordinator
        self.agent_coordinator = AgentCoordinator(self.communication_manager)
        
        # Initialize workflow manager with agents
        self.workflow_manager = WorkflowManager(self.agents)
        
        logger.info("Agent coordination setup complete")
    
    async def start(self) -> None:
        """Start the agent system.
        
        Raises:
            AgentSystemError: If system is not initialized or start fails
        """
        if not self.is_initialized:
            raise AgentSystemError("System not initialized. Call initialize() first.")
        
        try:
            logger.info("Starting AgentSystem...")
            
            # Start RAG manager if available
            if self.rag_manager:
                await self.rag_manager.start()
            
            # System is now running
            self.is_running = True
            
            logger.info("AgentSystem started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AgentSystem: {e}")
            raise AgentSystemError(f"Start failed: {e}") from e
    
    async def stop(self) -> None:
        """Stop the agent system gracefully."""
        logger.info("Stopping AgentSystem...")
        
        try:
            # Stop RAG manager
            if self.rag_manager:
                await self.rag_manager.stop()
            
            # Shutdown workflow executor
            if self.workflow_executor:
                self.workflow_executor.shutdown()
            
            # Shutdown communication manager
            if self.communication_manager:
                self.communication_manager.shutdown()
            
            # Shutdown all agents
            for agent in self.agents.values():
                agent.shutdown()
            
            self.is_running = False
            logger.info("AgentSystem stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def process_message(
        self,
        content: str,
        sender_id: str = "user",
        message_type: MessageType = MessageType.USER_QUESTION,
        context: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Process a message through the agent system.
        
        Args:
            content: Message content
            sender_id: ID of the sender
            message_type: Type of message
            context: Optional context
            
        Returns:
            Response from the system
            
        Raises:
            AgentSystemError: If system is not running or processing fails
        """
        if not self.is_running:
            raise AgentSystemError("System not running. Call start() first.")
        
        try:
            # Create message
            message = Message(
                sender_id=sender_id,
                recipient_id="system",
                content=content,
                message_type=message_type,
                metadata=context or {},
            )
            
            # Route to analyst agent for initial processing
            analyst_agent = self._get_analyst_agent()
            if not analyst_agent:
                raise AgentSystemError("Analyst agent not available")
            
            # Process message
            response = analyst_agent.process_message(message)
            
            logger.info(f"Processed message from {sender_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise AgentSystemError(f"Message processing failed: {e}") from e
    
    def _get_analyst_agent(self) -> Optional[BaseAgent]:
        """Get the analyst agent for routing.
        
        Returns:
            Analyst agent or None if not found
        """
        for agent in self.agents.values():
            if (hasattr(agent, 'config') and 
                hasattr(agent.config, 'agent_type') and
                agent.config.agent_type == AgentType.ANALYST):
                return agent
        return None
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents of the specified type
        """
        matching_agents = []
        for agent in self.agents.values():
            if (hasattr(agent, 'config') and 
                hasattr(agent.config, 'agent_type') and
                agent.config.agent_type == agent_type):
                matching_agents.append(agent)
        return matching_agents
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        agent_list = []
        for agent_id, agent in self.agents.items():
            agent_info = {
                "agent_id": agent_id,
                "agent_type": agent.agent_type if hasattr(agent, 'agent_type') else "unknown",
                "is_active": getattr(agent, 'state', {}).get('is_active', False),
            }
            agent_list.append(agent_info)
        return agent_list
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status.
        
        Returns:
            Dictionary containing system status information
        """
        status = {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "agents_count": len(self.agents),
            "components": {
                "config_manager": self.config_manager is not None,
                "prompt_manager": self.prompt_manager is not None,
                "rag_manager": self.rag_manager is not None,
                "communication_manager": self.communication_manager is not None,
                "agent_coordinator": self.agent_coordinator is not None,
                "workflow_manager": self.workflow_manager is not None,
                "workflow_executor": self.workflow_executor is not None,
            }
        }
        
        # Add agent status
        if self.agents:
            status["agents"] = self.list_agents()
        
        # Add communication stats if available
        if self.communication_manager:
            status["communication_stats"] = self.communication_manager.get_delivery_statistics()
        
        # Add workflow stats if available
        if self.workflow_executor:
            status["workflow_stats"] = self.workflow_executor.get_statistics()
        
        return status
    
    async def reload_prompts(self) -> bool:
        """Reload prompt templates.
        
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if self.prompt_manager:
                self.prompt_manager.reload_prompts()
                logger.info("Prompts reloaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reload prompts: {e}")
            return False
    
    async def reload_configuration(self) -> bool:
        """Reload system configuration.
        
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if self.config_manager:
                new_config = self.config_manager.load_config()
                self.config_manager.validate_config(new_config)
                self.config = new_config
                logger.info("Configuration reloaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the AgentSystem."""
        return (f"AgentSystem(initialized={self.is_initialized}, "
                f"running={self.is_running}, agents={len(self.agents)})")