"""Architect Agent for solution design and system architecture."""

import logging
from typing import Any, Dict, List, Optional

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


class ArchitecturalAnalysis:
    """Result of architectural analysis."""
    
    def __init__(
        self,
        problem_understanding: str,
        key_challenges: List[str],
        constraints: List[str],
        assumptions: List[str]
    ):
        self.problem_understanding = problem_understanding
        self.key_challenges = key_challenges
        self.constraints = constraints
        self.assumptions = assumptions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "problem_understanding": self.problem_understanding,
            "key_challenges": self.key_challenges,
            "constraints": self.constraints,
            "assumptions": self.assumptions,
        }


class ArchitecturalSolution:
    """Architectural solution design."""
    
    def __init__(
        self,
        overview: str,
        components: List[Dict[str, str]],
        technology_stack: Dict[str, str],
        patterns: List[Dict[str, str]],
        diagrams: Optional[List[str]] = None
    ):
        self.overview = overview
        self.components = components
        self.technology_stack = technology_stack
        self.patterns = patterns
        self.diagrams = diagrams or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overview": self.overview,
            "components": self.components,
            "technology_stack": self.technology_stack,
            "patterns": self.patterns,
            "diagrams": self.diagrams,
        }


class ArchitectAgent(BaseAgent):
    """Architect agent responsible for solution design and system architecture."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Architect agent."""
        super().__init__(*args, **kwargs)
        
        # Validate that this agent has the required capabilities
        required_capabilities = [Capability.ARCHITECTURE_DESIGN, Capability.SOLUTION_EXPLANATION]
        for capability in required_capabilities:
            if not self.has_capability(capability):
                logger.warning(
                    f"ArchitectAgent {self.agent_id} missing required capability: {capability}"
                )
        
        # Architecture domain knowledge
        self._architecture_patterns = {
            "microservices": "Distributed architecture with loosely coupled services",
            "monolithic": "Single deployable unit with all functionality",
            "layered": "Organized into horizontal layers with specific responsibilities",
            "event_driven": "Components communicate through events and message passing",
            "hexagonal": "Ports and adapters pattern for clean architecture",
            "serverless": "Function-as-a-Service with event-driven execution",
            "mvc": "Model-View-Controller separation of concerns",
            "cqrs": "Command Query Responsibility Segregation pattern",
            "saga": "Distributed transaction management pattern",
            "circuit_breaker": "Fault tolerance pattern for service calls"
        }
        
        self._technology_categories = {
            "frontend": ["React", "Vue.js", "Angular", "Svelte", "Next.js"],
            "backend": ["Node.js", "Python/Django", "Java/Spring", "Go", "Rust"],
            "database": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Neo4j"],
            "messaging": ["RabbitMQ", "Apache Kafka", "Redis Pub/Sub", "AWS SQS"],
            "cloud": ["AWS", "Azure", "GCP", "Kubernetes", "Docker"],
            "monitoring": ["Prometheus", "Grafana", "ELK Stack", "Jaeger", "DataDog"]
        }
        
        logger.info(f"ArchitectAgent {self.agent_id} initialized successfully")
    
    def _process_message_impl(self, message: Message) -> Response:
        """Process incoming messages for architectural design."""
        try:
            if message.message_type in [MessageType.USER_QUESTION, MessageType.AGENT_REQUEST]:
                return self._handle_architecture_request(message)
            else:
                return self._create_error_response(
                    message.id,
                    f"Unsupported message type: {message.message_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message in ArchitectAgent: {e}")
            return self._create_error_response(message.id, str(e))
    
    def _handle_architecture_request(self, message: Message) -> Response:
        """Handle architecture and design requests."""
        question = message.content
        
        # Step 1: Retrieve relevant context using graph RAG for relationships
        context = self._retrieve_architectural_context(question)
        
        # Step 2: Analyze the problem from architectural perspective
        analysis = self._analyze_architectural_problem(question, context)
        
        # Step 3: Generate architectural solution
        solution = self._generate_architectural_solution(question, analysis, context)
        
        # Step 4: Create comprehensive response
        response_content = self._format_architectural_response(
            question, analysis, solution, context
        )
        
        # Calculate confidence based on context availability and problem complexity
        confidence = self._calculate_confidence(question, context, analysis)
        
        context_used = []
        if context and context.documents:
            context_used.extend([f"doc_{i}" for i in range(len(context.documents))])
        if context and context.graph_data:
            context_used.append("graph_relationships")
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            confidence_score=confidence,
            context_used=context_used,
            metadata={
                "analysis": analysis.to_dict() if analysis else None,
                "solution": solution.to_dict() if solution else None,
                "architecture_domain": True,
            }
        )
    
    def _retrieve_architectural_context(self, question: str) -> Optional[Context]:
        """Retrieve context with emphasis on graph relationships for architecture."""
        if not self.rag_manager:
            return None
        
        try:
            # Use graph RAG first for architectural relationships
            if self.config.graph_rag_enabled:
                context = self.retrieve_context(question, RetrievalType.GRAPH)
                if context and context.graph_data:
                    return context
            
            # Fall back to vector search for relevant documents
            return self.retrieve_context(question, RetrievalType.VECTOR, k=5)
            
        except Exception as e:
            logger.error(f"Error retrieving architectural context: {e}")
            return None
    
    def _analyze_architectural_problem(
        self,
        question: str,
        context: Optional[Context]
    ) -> Optional[ArchitecturalAnalysis]:
        """Analyze the architectural problem and requirements."""
        try:
            # Extract key architectural concerns from the question
            problem_understanding = self._extract_problem_understanding(question)
            key_challenges = self._identify_architectural_challenges(question, context)
            constraints = self._identify_constraints(question, context)
            assumptions = self._identify_assumptions(question, context)
            
            return ArchitecturalAnalysis(
                problem_understanding=problem_understanding,
                key_challenges=key_challenges,
                constraints=constraints,
                assumptions=assumptions
            )
            
        except Exception as e:
            logger.error(f"Error analyzing architectural problem: {e}")
            return None
    
    def _extract_problem_understanding(self, question: str) -> str:
        """Extract and articulate understanding of the problem."""
        question_lower = question.lower()
        
        # Identify the type of architectural question
        if any(word in question_lower for word in ["design", "architecture", "system"]):
            if "microservices" in question_lower:
                return "Request for microservices architecture design and implementation strategy."
            elif "scalable" in question_lower or "scale" in question_lower:
                return "Request for scalable system architecture with performance considerations."
            elif "api" in question_lower:
                return "Request for API architecture and integration design."
            elif "database" in question_lower or "data" in question_lower:
                return "Request for data architecture and database design strategy."
            else:
                return "Request for comprehensive system architecture and design guidance."
        
        return f"Architectural consultation request: {question[:100]}..."
    
    def _identify_architectural_challenges(
        self,
        question: str,
        context: Optional[Context]
    ) -> List[str]:
        """Identify key architectural challenges from the question."""
        challenges = []
        question_lower = question.lower()
        
        # Common architectural challenges
        if "scale" in question_lower or "scalability" in question_lower:
            challenges.append("Designing for horizontal and vertical scalability")
        
        if "performance" in question_lower:
            challenges.append("Optimizing system performance and response times")
        
        if "distributed" in question_lower or "microservices" in question_lower:
            challenges.append("Managing distributed system complexity and communication")
        
        if "security" in question_lower:
            challenges.append("Implementing comprehensive security architecture")
        
        if "integration" in question_lower or "api" in question_lower:
            challenges.append("Designing effective integration patterns and APIs")
        
        if "data" in question_lower or "database" in question_lower:
            challenges.append("Designing efficient data architecture and storage strategy")
        
        # Add context-based challenges if available
        if context and context.graph_data:
            challenges.append("Leveraging existing system relationships and dependencies")
        
        # Default challenges if none identified
        if not challenges:
            challenges.extend([
                "Balancing system complexity with maintainability",
                "Ensuring scalability and performance requirements",
                "Implementing proper separation of concerns"
            ])
        
        return challenges
    
    def _identify_constraints(
        self,
        question: str,
        context: Optional[Context]
    ) -> List[str]:
        """Identify system constraints from the question and context."""
        constraints = []
        question_lower = question.lower()
        
        # Technology constraints
        if "legacy" in question_lower:
            constraints.append("Integration with existing legacy systems")
        
        if "budget" in question_lower or "cost" in question_lower:
            constraints.append("Budget and cost optimization requirements")
        
        if "time" in question_lower or "deadline" in question_lower:
            constraints.append("Time-to-market and delivery timeline constraints")
        
        # Technical constraints
        if "on-premise" in question_lower or "on premise" in question_lower:
            constraints.append("On-premise deployment requirements")
        
        if "cloud" in question_lower:
            constraints.append("Cloud-native architecture requirements")
        
        # Default constraints
        constraints.extend([
            "Maintainability and code quality standards",
            "Security and compliance requirements",
            "Team expertise and learning curve considerations"
        ])
        
        return constraints
    
    def _identify_assumptions(
        self,
        question: str,
        context: Optional[Context]
    ) -> List[str]:
        """Identify assumptions for the architectural solution."""
        assumptions = [
            "Modern development practices and CI/CD pipeline availability",
            "Team has basic understanding of chosen technology stack",
            "Standard security and monitoring tools will be implemented",
            "Iterative development and feedback cycles are possible"
        ]
        
        question_lower = question.lower()
        
        if "web" in question_lower or "api" in question_lower:
            assumptions.append("HTTP/REST-based communication is acceptable")
        
        if "real-time" in question_lower:
            assumptions.append("Real-time communication requirements are critical")
        
        if "mobile" in question_lower:
            assumptions.append("Mobile-first or mobile-responsive design is required")
        
        return assumptions
    
    def _generate_architectural_solution(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis],
        context: Optional[Context]
    ) -> Optional[ArchitecturalSolution]:
        """Generate comprehensive architectural solution."""
        try:
            # Generate solution overview
            overview = self._generate_solution_overview(question, analysis)
            
            # Identify key components
            components = self._identify_system_components(question, analysis, context)
            
            # Recommend technology stack
            technology_stack = self._recommend_technology_stack(question, analysis)
            
            # Suggest architectural patterns
            patterns = self._suggest_architectural_patterns(question, analysis)
            
            # Generate diagrams if applicable
            diagrams = self._generate_architecture_diagrams(question, components)
            
            return ArchitecturalSolution(
                overview=overview,
                components=components,
                technology_stack=technology_stack,
                patterns=patterns,
                diagrams=diagrams
            )
            
        except Exception as e:
            logger.error(f"Error generating architectural solution: {e}")
            return None
    
    def _generate_solution_overview(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis]
    ) -> str:
        """Generate high-level solution overview."""
        question_lower = question.lower()
        
        if "microservices" in question_lower:
            return (
                "A microservices architecture that decomposes the system into loosely coupled, "
                "independently deployable services. Each service owns its data and communicates "
                "through well-defined APIs, enabling scalability and team autonomy."
            )
        elif "api" in question_lower:
            return (
                "A robust API-first architecture that provides clean separation between "
                "frontend and backend systems, enabling multiple client applications "
                "and third-party integrations through standardized interfaces."
            )
        elif "scalable" in question_lower:
            return (
                "A horizontally scalable architecture designed to handle increasing load "
                "through distributed components, caching strategies, and asynchronous "
                "processing patterns."
            )
        else:
            return (
                "A well-structured system architecture that balances scalability, "
                "maintainability, and performance while following established "
                "architectural principles and patterns."
            )
    
    def _identify_system_components(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis],
        context: Optional[Context]
    ) -> List[Dict[str, str]]:
        """Identify key system components."""
        components = []
        question_lower = question.lower()
        
        # Core components based on question type
        if "web" in question_lower or "api" in question_lower:
            components.extend([
                {
                    "name": "API Gateway",
                    "responsibility": "Request routing, authentication, rate limiting, and API versioning",
                    "technology": "Kong, AWS API Gateway, or Nginx"
                },
                {
                    "name": "Application Services",
                    "responsibility": "Core business logic and domain operations",
                    "technology": "Microservices or modular monolith"
                },
                {
                    "name": "Data Layer",
                    "responsibility": "Data persistence, caching, and retrieval",
                    "technology": "Database + Redis/Memcached"
                }
            ])
        
        if "frontend" in question_lower or "ui" in question_lower:
            components.append({
                "name": "Frontend Application",
                "responsibility": "User interface and client-side logic",
                "technology": "React, Vue.js, or Angular SPA"
            })
        
        if "auth" in question_lower or "security" in question_lower:
            components.append({
                "name": "Authentication Service",
                "responsibility": "User authentication, authorization, and session management",
                "technology": "OAuth2/JWT with identity provider"
            })
        
        if "notification" in question_lower or "messaging" in question_lower:
            components.append({
                "name": "Messaging System",
                "responsibility": "Asynchronous communication and event processing",
                "technology": "RabbitMQ, Apache Kafka, or cloud messaging"
            })
        
        # Always include monitoring and logging
        components.extend([
            {
                "name": "Monitoring & Observability",
                "responsibility": "System health monitoring, logging, and performance metrics",
                "technology": "Prometheus, Grafana, ELK Stack"
            },
            {
                "name": "Load Balancer",
                "responsibility": "Traffic distribution and high availability",
                "technology": "Nginx, HAProxy, or cloud load balancer"
            }
        ])
        
        return components
    
    def _recommend_technology_stack(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis]
    ) -> Dict[str, str]:
        """Recommend appropriate technology stack."""
        question_lower = question.lower()
        stack = {}
        
        # Backend technology
        if "python" in question_lower:
            stack["backend"] = "Python with FastAPI or Django"
        elif "node" in question_lower or "javascript" in question_lower:
            stack["backend"] = "Node.js with Express or NestJS"
        elif "java" in question_lower:
            stack["backend"] = "Java with Spring Boot"
        elif "go" in question_lower:
            stack["backend"] = "Go with Gin or Echo framework"
        else:
            stack["backend"] = "Python/FastAPI (recommended for rapid development)"
        
        # Database
        if "nosql" in question_lower or "mongodb" in question_lower:
            stack["database"] = "MongoDB for document storage"
        elif "graph" in question_lower:
            stack["database"] = "Neo4j for graph relationships"
        elif "time-series" in question_lower:
            stack["database"] = "InfluxDB or TimescaleDB"
        else:
            stack["database"] = "PostgreSQL for relational data"
        
        # Caching
        stack["caching"] = "Redis for session storage and caching"
        
        # Frontend (if mentioned)
        if "react" in question_lower:
            stack["frontend"] = "React with TypeScript"
        elif "vue" in question_lower:
            stack["frontend"] = "Vue.js with Composition API"
        elif "angular" in question_lower:
            stack["frontend"] = "Angular with TypeScript"
        elif "frontend" in question_lower or "ui" in question_lower:
            stack["frontend"] = "React with TypeScript (recommended)"
        
        # Infrastructure
        if "kubernetes" in question_lower or "k8s" in question_lower:
            stack["orchestration"] = "Kubernetes for container orchestration"
        elif "docker" in question_lower:
            stack["containerization"] = "Docker for containerization"
        
        if "aws" in question_lower:
            stack["cloud"] = "AWS (EC2, RDS, S3, Lambda)"
        elif "azure" in question_lower:
            stack["cloud"] = "Microsoft Azure"
        elif "gcp" in question_lower:
            stack["cloud"] = "Google Cloud Platform"
        else:
            stack["cloud"] = "Cloud-agnostic with Docker containers"
        
        return stack
    
    def _suggest_architectural_patterns(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis]
    ) -> List[Dict[str, str]]:
        """Suggest relevant architectural patterns."""
        patterns = []
        question_lower = question.lower()
        
        # Pattern suggestions based on question content
        if "microservices" in question_lower:
            patterns.extend([
                {
                    "pattern": "Microservices",
                    "description": self._architecture_patterns["microservices"],
                    "rationale": "Enables independent scaling and deployment of services"
                },
                {
                    "pattern": "API Gateway",
                    "description": "Centralized entry point for all client requests",
                    "rationale": "Provides cross-cutting concerns like authentication and routing"
                },
                {
                    "pattern": "Circuit Breaker",
                    "description": self._architecture_patterns["circuit_breaker"],
                    "rationale": "Prevents cascade failures in distributed systems"
                }
            ])
        
        if "event" in question_lower or "messaging" in question_lower:
            patterns.append({
                "pattern": "Event-Driven Architecture",
                "description": self._architecture_patterns["event_driven"],
                "rationale": "Enables loose coupling and asynchronous processing"
            })
        
        if "cqrs" in question_lower or "command" in question_lower and "query" in question_lower:
            patterns.append({
                "pattern": "CQRS",
                "description": self._architecture_patterns["cqrs"],
                "rationale": "Separates read and write operations for better performance"
            })
        
        # Default patterns for common scenarios
        if not patterns:
            patterns.extend([
                {
                    "pattern": "Layered Architecture",
                    "description": self._architecture_patterns["layered"],
                    "rationale": "Provides clear separation of concerns and maintainability"
                },
                {
                    "pattern": "Repository Pattern",
                    "description": "Abstracts data access logic from business logic",
                    "rationale": "Enables testability and data source flexibility"
                }
            ])
        
        return patterns
    
    def _generate_architecture_diagrams(
        self,
        question: str,
        components: List[Dict[str, str]]
    ) -> List[str]:
        """Generate Mermaid diagrams for architecture visualization."""
        diagrams = []
        
        # Generate a high-level system diagram
        if components:
            mermaid_diagram = self._create_system_diagram(components)
            diagrams.append(mermaid_diagram)
        
        # Generate sequence diagram for API flows if relevant
        question_lower = question.lower()
        if "api" in question_lower or "flow" in question_lower:
            sequence_diagram = self._create_sequence_diagram(question)
            diagrams.append(sequence_diagram)
        
        return diagrams
    
    def _create_system_diagram(self, components: List[Dict[str, str]]) -> str:
        """Create a Mermaid system architecture diagram."""
        diagram = "```mermaid\ngraph TB\n"
        
        # Add nodes for each component
        for i, component in enumerate(components):
            node_id = f"C{i+1}"
            name = component["name"].replace(" ", "_")
            diagram += f"    {node_id}[{component['name']}]\n"
        
        # Add basic connections (simplified)
        if len(components) > 1:
            diagram += f"    C1 --> C2\n"
            if len(components) > 2:
                diagram += f"    C2 --> C3\n"
            if len(components) > 3:
                diagram += f"    C1 --> C{len(components)}\n"
        
        diagram += "```"
        return diagram
    
    def _create_sequence_diagram(self, question: str) -> str:
        """Create a Mermaid sequence diagram for API flows."""
        diagram = "```mermaid\nsequenceDiagram\n"
        diagram += "    participant Client\n"
        diagram += "    participant API_Gateway\n"
        diagram += "    participant Service\n"
        diagram += "    participant Database\n\n"
        diagram += "    Client->>API_Gateway: Request\n"
        diagram += "    API_Gateway->>Service: Forward Request\n"
        diagram += "    Service->>Database: Query Data\n"
        diagram += "    Database-->>Service: Return Data\n"
        diagram += "    Service-->>API_Gateway: Response\n"
        diagram += "    API_Gateway-->>Client: Final Response\n"
        diagram += "```"
        return diagram
    
    def _format_architectural_response(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis],
        solution: Optional[ArchitecturalSolution],
        context: Optional[Context]
    ) -> str:
        """Format the comprehensive architectural response."""
        response = "# Architectural Design Response\n\n"
        
        # Problem Analysis Section
        response += "## 1. Problem Analysis\n\n"
        if analysis:
            response += f"**Understanding:** {analysis.problem_understanding}\n\n"
            
            if analysis.key_challenges:
                response += "**Key Architectural Challenges:**\n"
                for challenge in analysis.key_challenges:
                    response += f"- {challenge}\n"
                response += "\n"
            
            if analysis.constraints:
                response += "**Constraints:**\n"
                for constraint in analysis.constraints:
                    response += f"- {constraint}\n"
                response += "\n"
            
            if analysis.assumptions:
                response += "**Assumptions:**\n"
                for assumption in analysis.assumptions:
                    response += f"- {assumption}\n"
                response += "\n"
        
        # Proposed Solution Section
        if solution:
            response += "## 2. Proposed Solution\n\n"
            response += f"**Overview:** {solution.overview}\n\n"
            
            if solution.components:
                response += "**Key Components:**\n"
                for component in solution.components:
                    response += f"- **{component['name']}**: {component['responsibility']}\n"
                    response += f"  - *Technology*: {component['technology']}\n"
                response += "\n"
            
            if solution.technology_stack:
                response += "**Technology Stack:**\n"
                for category, tech in solution.technology_stack.items():
                    response += f"- **{category.title()}**: {tech}\n"
                response += "\n"
        
        # Design Patterns Section
        if solution and solution.patterns:
            response += "## 3. Design Patterns\n\n"
            for pattern in solution.patterns:
                response += f"**{pattern['pattern']}**\n"
                response += f"- *Description*: {pattern['description']}\n"
                response += f"- *Rationale*: {pattern['rationale']}\n\n"
        
        # Architecture Diagrams
        if solution and solution.diagrams:
            response += "## 4. Architecture Diagrams\n\n"
            for i, diagram in enumerate(solution.diagrams, 1):
                response += f"### Diagram {i}\n\n{diagram}\n\n"
        
        # Implementation Recommendations
        response += "## 5. Implementation Roadmap\n\n"
        response += "**Phase 1: Foundation**\n"
        response += "- Set up core infrastructure and development environment\n"
        response += "- Implement basic authentication and security measures\n"
        response += "- Create initial data models and API structure\n\n"
        
        response += "**Phase 2: Core Features**\n"
        response += "- Implement primary business logic and services\n"
        response += "- Set up monitoring and logging infrastructure\n"
        response += "- Add comprehensive testing suite\n\n"
        
        response += "**Phase 3: Optimization**\n"
        response += "- Performance optimization and caching strategies\n"
        response += "- Security hardening and compliance measures\n"
        response += "- Scalability improvements and load testing\n\n"
        
        # Context Information
        if context:
            response += "## 6. Additional Context\n\n"
            if context.documents:
                response += f"*This response incorporates insights from {len(context.documents)} relevant documents.*\n"
            if context.graph_data:
                response += "*Graph relationships were analyzed to understand system dependencies.*\n"
        
        return response
    
    def _calculate_confidence(
        self,
        question: str,
        context: Optional[Context],
        analysis: Optional[ArchitecturalAnalysis]
    ) -> float:
        """Calculate confidence score for the architectural response."""
        base_confidence = 0.7
        
        # Increase confidence based on available context
        if context:
            if context.documents:
                base_confidence += 0.1
            if context.graph_data:
                base_confidence += 0.1
        
        # Increase confidence based on question specificity
        question_lower = question.lower()
        specific_terms = [
            "microservices", "api", "database", "scalable", "architecture",
            "design", "pattern", "security", "performance"
        ]
        
        specificity_score = sum(1 for term in specific_terms if term in question_lower)
        base_confidence += min(0.1, specificity_score * 0.02)
        
        # Ensure confidence is within valid range
        return min(0.95, max(0.5, base_confidence))
    
    def _create_error_response(self, message_id: str, error_message: str) -> Response:
        """Create an error response."""
        return Response(
            message_id=message_id,
            agent_id=self.agent_id,
            content=f"I encountered an error while analyzing the architecture: {error_message}",
            confidence_score=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def analyze_architecture(self, question: str) -> Optional[ArchitecturalAnalysis]:
        """Public method to analyze architectural requirements."""
        context = self._retrieve_architectural_context(question)
        return self._analyze_architectural_problem(question, context)
    
    def generate_solution(
        self,
        question: str,
        analysis: Optional[ArchitecturalAnalysis] = None
    ) -> Optional[ArchitecturalSolution]:
        """Public method to generate architectural solution."""
        if analysis is None:
            context = self._retrieve_architectural_context(question)
            analysis = self._analyze_architectural_problem(question, context)
        
        context = self._retrieve_architectural_context(question)
        return self._generate_architectural_solution(question, analysis, context)