"""Agent configuration and type models."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AgentType(str, Enum):
    """Types of agents in the system."""
    ANALYST = "analyst"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    CODE_REVIEWER = "code_reviewer"
    TESTER = "tester"


class Capability(str, Enum):
    """Capabilities that agents can have."""
    QUESTION_ROUTING = "question_routing"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TEST_CREATION = "test_creation"
    ARCHITECTURE_DESIGN = "architecture_design"
    SOLUTION_EXPLANATION = "solution_explanation"
    WORKFLOW_CREATION = "workflow_creation"
    RAG_RETRIEVAL = "rag_retrieval"
    GRAPH_TRAVERSAL = "graph_traversal"


class LLMConfig(BaseModel):
    """Configuration for the LLM model used by an agent."""
    
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    base_url: Optional[str] = Field(None, description="Custom API base URL")
    
    @field_validator('model_name')
    @classmethod
    def model_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Model name cannot be empty')
        return v.strip()


class AgentConfig(BaseModel):
    """Configuration for an individual agent."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: AgentType = Field(..., description="Type of agent")
    llm_config: LLMConfig = Field(..., description="LLM configuration")
    prompt_templates: Dict[str, str] = Field(default_factory=dict)
    capabilities: List[Capability] = Field(default_factory=list)
    rag_enabled: bool = Field(default=True)
    graph_rag_enabled: bool = Field(default=True)
    max_context_length: int = Field(default=4000, gt=0)
    response_timeout: int = Field(default=30, gt=0)
    
    @field_validator('agent_id')
    @classmethod
    def agent_id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Agent ID cannot be empty')
        return v.strip()
    
    @field_validator('capabilities')
    @classmethod
    def validate_capabilities_for_type(cls, v, info):
        """Validate that capabilities match the agent type."""
        if 'agent_type' not in info.data:
            return v
        
        agent_type = info.data['agent_type']
        required_capabilities = {
            AgentType.ANALYST: [Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
            AgentType.ARCHITECT: [Capability.ARCHITECTURE_DESIGN, Capability.SOLUTION_EXPLANATION],
            AgentType.DEVELOPER: [Capability.CODE_GENERATION],
            AgentType.CODE_REVIEWER: [Capability.CODE_REVIEW],
            AgentType.TESTER: [Capability.TEST_CREATION],
        }
        
        if agent_type in required_capabilities:
            missing_capabilities = set(required_capabilities[agent_type]) - set(v)
            if missing_capabilities:
                raise ValueError(
                    f"Agent type {agent_type} requires capabilities: "
                    f"{', '.join(missing_capabilities)}"
                )
        
        return v
    
    def has_capability(self, capability: Capability) -> bool:
        """Check if the agent has a specific capability."""
        return capability in self.capabilities
    
    def get_prompt_template(self, template_name: str) -> Optional[str]:
        """Get a specific prompt template."""
        return self.prompt_templates.get(template_name)


class AgentState(BaseModel):
    """State information for an agent."""
    
    agent_id: str = Field(..., description="Agent identifier")
    is_active: bool = Field(default=True)
    current_task: Optional[str] = Field(None, description="Current task being processed")
    last_activity: Optional[str] = Field(None, description="Timestamp of last activity")
    context_cache: Dict[str, Any] = Field(default_factory=dict)
    conversation_count: int = Field(default=0, ge=0)
    
    def update_activity(self, task: str) -> None:
        """Update the agent's current activity."""
        self.current_task = task
        from datetime import datetime
        self.last_activity = datetime.utcnow().isoformat()
        self.conversation_count += 1
    
    def clear_task(self) -> None:
        """Clear the current task."""
        self.current_task = None
    
    def cache_context(self, key: str, value: Any) -> None:
        """Cache context information."""
        self.context_cache[key] = value
    
    def get_cached_context(self, key: str) -> Optional[Any]:
        """Retrieve cached context information."""
        return self.context_cache.get(key)