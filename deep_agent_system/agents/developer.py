"""Developer Agent for code generation and implementation."""

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


class CodeAnalysis:
    """Result of code analysis and requirements extraction."""
    
    def __init__(
        self,
        problem_understanding: str,
        programming_language: str,
        framework_requirements: List[str],
        complexity_level: str,
        key_requirements: List[str]
    ):
        self.problem_understanding = problem_understanding
        self.programming_language = programming_language
        self.framework_requirements = framework_requirements
        self.complexity_level = complexity_level
        self.key_requirements = key_requirements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "problem_understanding": self.problem_understanding,
            "programming_language": self.programming_language,
            "framework_requirements": self.framework_requirements,
            "complexity_level": self.complexity_level,
            "key_requirements": self.key_requirements,
        }


class CodeImplementation:
    """Code implementation result."""
    
    def __init__(
        self,
        strategy: str,
        code_blocks: List[Dict[str, str]],
        dependencies: List[str],
        setup_instructions: List[str],
        usage_examples: List[str],
        test_cases: List[Dict[str, str]]
    ):
        self.strategy = strategy
        self.code_blocks = code_blocks
        self.dependencies = dependencies
        self.setup_instructions = setup_instructions
        self.usage_examples = usage_examples
        self.test_cases = test_cases
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "strategy": self.strategy,
            "code_blocks": self.code_blocks,
            "dependencies": self.dependencies,
            "setup_instructions": self.setup_instructions,
            "usage_examples": self.usage_examples,
            "test_cases": self.test_cases,
        }


class DeveloperAgent(BaseAgent):
    """Developer agent responsible for code generation and implementation."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Developer agent."""
        super().__init__(*args, **kwargs)
        
        # Validate that this agent has the required capabilities
        required_capabilities = [Capability.CODE_GENERATION]
        for capability in required_capabilities:
            if not self.has_capability(capability):
                logger.warning(
                    f"DeveloperAgent {self.agent_id} missing required capability: {capability}"
                )
        
        # Programming language patterns and frameworks
        self._language_patterns = {
            "python": {
                "extensions": [".py"],
                "frameworks": ["django", "flask", "fastapi", "pandas", "numpy", "pytest"],
                "keywords": ["def", "class", "import", "from", "python", "pip"]
            },
            "javascript": {
                "extensions": [".js", ".ts", ".jsx", ".tsx"],
                "frameworks": ["react", "vue", "angular", "node", "express", "next"],
                "keywords": ["function", "const", "let", "var", "npm", "yarn", "javascript", "typescript"]
            },
            "java": {
                "extensions": [".java"],
                "frameworks": ["spring", "hibernate", "maven", "gradle"],
                "keywords": ["public", "class", "interface", "package", "java", "maven"]
            },
            "go": {
                "extensions": [".go"],
                "frameworks": ["gin", "echo", "fiber", "gorm"],
                "keywords": ["func", "package", "import", "go", "golang"]
            },
            "rust": {
                "extensions": [".rs"],
                "frameworks": ["actix", "rocket", "tokio", "serde"],
                "keywords": ["fn", "struct", "impl", "use", "cargo", "rust"]
            }
        }
        
        # Code complexity indicators
        self._complexity_indicators = {
            "simple": ["hello world", "basic", "simple", "tutorial", "example"],
            "medium": ["api", "database", "authentication", "crud", "service"],
            "complex": ["distributed", "microservice", "scalable", "enterprise", "architecture"]
        }
        
        logger.info(f"DeveloperAgent {self.agent_id} initialized successfully")
    
    def _process_message_impl(self, message: Message) -> Response:
        """Process incoming messages for code generation."""
        try:
            if message.message_type in [MessageType.USER_QUESTION, MessageType.AGENT_REQUEST]:
                return self._handle_development_request(message)
            else:
                return self._create_error_response(
                    message.id,
                    f"Unsupported message type: {message.message_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message in DeveloperAgent: {e}")
            return self._create_error_response(message.id, str(e))
    
    def _handle_development_request(self, message: Message) -> Response:
        """Handle development and code generation requests."""
        question = message.content
        
        # Step 1: Retrieve relevant code examples and patterns using vector RAG
        context = self._retrieve_development_context(question)
        
        # Step 2: Analyze the development requirements
        analysis = self._analyze_development_requirements(question, context)
        
        # Step 3: Generate code implementation
        implementation = self._generate_code_implementation(question, analysis, context)
        
        # Step 4: Create comprehensive response
        response_content = self._format_development_response(
            question, analysis, implementation, context
        )
        
        # Calculate confidence based on context availability and complexity
        confidence = self._calculate_confidence(question, context, analysis)
        
        context_used = []
        if context and context.documents:
            context_used.extend([f"code_example_{i}" for i in range(len(context.documents))])
        
        return Response(
            message_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            confidence_score=confidence,
            context_used=context_used,
            metadata={
                "analysis": analysis.to_dict() if analysis else None,
                "implementation": implementation.to_dict() if implementation else None,
                "development_domain": True,
            }
        )
    
    def _retrieve_development_context(self, question: str) -> Optional[Context]:
        """Retrieve context with emphasis on code examples and patterns."""
        if not self.rag_manager:
            return None
        
        try:
            # Use vector search for code examples and documentation
            return self.retrieve_context(question, RetrievalType.VECTOR, k=5)
            
        except Exception as e:
            logger.error(f"Error retrieving development context: {e}")
            return None
    
    def _analyze_development_requirements(
        self,
        question: str,
        context: Optional[Context]
    ) -> Optional[CodeAnalysis]:
        """Analyze the development requirements from the question."""
        try:
            # Extract problem understanding
            problem_understanding = self._extract_problem_understanding(question)
            
            # Detect programming language
            programming_language = self._detect_programming_language(question, context)
            
            # Identify framework requirements
            framework_requirements = self._identify_framework_requirements(question, programming_language)
            
            # Assess complexity level
            complexity_level = self._assess_complexity_level(question)
            
            # Extract key requirements
            key_requirements = self._extract_key_requirements(question)
            
            return CodeAnalysis(
                problem_understanding=problem_understanding,
                programming_language=programming_language,
                framework_requirements=framework_requirements,
                complexity_level=complexity_level,
                key_requirements=key_requirements
            )
            
        except Exception as e:
            logger.error(f"Error analyzing development requirements: {e}")
            return None
    
    def _extract_problem_understanding(self, question: str) -> str:
        """Extract and articulate understanding of the development problem."""
        question_lower = question.lower()
        
        if "api" in question_lower:
            return "Request for API development and implementation."
        elif "database" in question_lower or "crud" in question_lower:
            return "Request for database operations and data management implementation."
        elif "authentication" in question_lower or "auth" in question_lower:
            return "Request for authentication and authorization system implementation."
        elif "web" in question_lower or "frontend" in question_lower:
            return "Request for web application or frontend development."
        elif "algorithm" in question_lower or "function" in question_lower:
            return "Request for algorithm implementation or function development."
        elif "test" in question_lower:
            return "Request for test implementation and testing strategies."
        else:
            return f"Development implementation request: {question[:100]}..."
    
    def _detect_programming_language(self, question: str, context: Optional[Context]) -> str:
        """Detect the intended programming language from the question."""
        question_lower = question.lower()
        
        # Check for explicit language mentions
        for language, patterns in self._language_patterns.items():
            if language in question_lower:
                return language
            
            # Check for framework mentions
            for framework in patterns["frameworks"]:
                if framework in question_lower:
                    return language
            
            # Check for language-specific keywords
            for keyword in patterns["keywords"]:
                if keyword in question_lower:
                    return language
        
        # Check context for language clues
        if context and context.documents:
            for doc in context.documents:
                content_lower = doc.content.lower()
                for language, patterns in self._language_patterns.items():
                    for keyword in patterns["keywords"]:
                        if keyword in content_lower:
                            return language
        
        # Default to Python for general development questions
        return "python"
    
    def _identify_framework_requirements(self, question: str, language: str) -> List[str]:
        """Identify framework requirements based on question and language."""
        frameworks = []
        question_lower = question.lower()
        
        if language in self._language_patterns:
            available_frameworks = self._language_patterns[language]["frameworks"]
            
            for framework in available_frameworks:
                if framework in question_lower:
                    frameworks.append(framework)
        
        # Add default frameworks based on question type
        if "api" in question_lower or "rest" in question_lower:
            if language == "python" and not frameworks:
                frameworks.append("fastapi")
            elif language == "javascript" and not frameworks:
                frameworks.append("express")
            elif language == "java" and not frameworks:
                frameworks.append("spring")
        
        if "web" in question_lower and language == "javascript" and not frameworks:
            frameworks.append("react")
        
        if "test" in question_lower:
            if language == "python":
                frameworks.append("pytest")
            elif language == "javascript":
                frameworks.append("jest")
        
        return frameworks
    
    def _assess_complexity_level(self, question: str) -> str:
        """Assess the complexity level of the development request."""
        question_lower = question.lower()
        
        # Count complexity indicators
        simple_count = sum(1 for indicator in self._complexity_indicators["simple"] 
                          if indicator in question_lower)
        medium_count = sum(1 for indicator in self._complexity_indicators["medium"] 
                          if indicator in question_lower)
        complex_count = sum(1 for indicator in self._complexity_indicators["complex"] 
                           if indicator in question_lower)
        
        if complex_count > 0:
            return "complex"
        elif medium_count > 0:
            return "medium"
        elif simple_count > 0:
            return "simple"
        else:
            # Default based on question length and technical terms
            technical_terms = ["class", "function", "method", "interface", "database", "api"]
            tech_count = sum(1 for term in technical_terms if term in question_lower)
            
            if len(question.split()) > 20 or tech_count > 3:
                return "medium"
            else:
                return "simple"
    
    def _extract_key_requirements(self, question: str) -> List[str]:
        """Extract key functional requirements from the question."""
        requirements = []
        question_lower = question.lower()
        
        # Common development requirements
        if "crud" in question_lower or "create" in question_lower and "read" in question_lower:
            requirements.append("CRUD operations implementation")
        
        if "api" in question_lower:
            requirements.append("RESTful API endpoints")
        
        if "database" in question_lower or "data" in question_lower:
            requirements.append("Database integration and operations")
        
        if "auth" in question_lower or "login" in question_lower:
            requirements.append("Authentication and authorization")
        
        if "validation" in question_lower:
            requirements.append("Input validation and error handling")
        
        if "test" in question_lower:
            requirements.append("Unit tests and test coverage")
        
        if "security" in question_lower:
            requirements.append("Security best practices implementation")
        
        if "performance" in question_lower:
            requirements.append("Performance optimization")
        
        # Default requirements if none identified
        if not requirements:
            requirements.extend([
                "Clean, readable code implementation",
                "Proper error handling",
                "Documentation and comments"
            ])
        
        return requirements
    
    def _generate_code_implementation(
        self,
        question: str,
        analysis: Optional[CodeAnalysis],
        context: Optional[Context]
    ) -> Optional[CodeImplementation]:
        """Generate comprehensive code implementation."""
        try:
            if not analysis:
                return None
            
            # Generate implementation strategy
            strategy = self._generate_implementation_strategy(question, analysis)
            
            # Generate code blocks
            code_blocks = self._generate_code_blocks(question, analysis, context)
            
            # Identify dependencies
            dependencies = self._identify_dependencies(analysis)
            
            # Generate setup instructions
            setup_instructions = self._generate_setup_instructions(analysis)
            
            # Generate usage examples
            usage_examples = self._generate_usage_examples(question, analysis)
            
            # Generate test cases
            test_cases = self._generate_test_cases(question, analysis)
            
            return CodeImplementation(
                strategy=strategy,
                code_blocks=code_blocks,
                dependencies=dependencies,
                setup_instructions=setup_instructions,
                usage_examples=usage_examples,
                test_cases=test_cases
            )
            
        except Exception as e:
            logger.error(f"Error generating code implementation: {e}")
            return None
    
    def _generate_implementation_strategy(self, question: str, analysis: CodeAnalysis) -> str:
        """Generate implementation strategy description."""
        strategy = f"Implementation approach for {analysis.programming_language} development:\n\n"
        
        if analysis.complexity_level == "simple":
            strategy += "- Direct implementation with clear, straightforward code\n"
            strategy += "- Focus on readability and basic error handling\n"
        elif analysis.complexity_level == "medium":
            strategy += "- Modular design with separation of concerns\n"
            strategy += "- Comprehensive error handling and validation\n"
            strategy += "- Integration with external libraries/frameworks\n"
        else:  # complex
            strategy += "- Scalable architecture with design patterns\n"
            strategy += "- Comprehensive testing and documentation\n"
            strategy += "- Performance optimization and security considerations\n"
        
        if analysis.framework_requirements:
            strategy += f"- Utilize {', '.join(analysis.framework_requirements)} for enhanced functionality\n"
        
        return strategy
    
    def _generate_code_blocks(
        self,
        question: str,
        analysis: CodeAnalysis,
        context: Optional[Context]
    ) -> List[Dict[str, str]]:
        """Generate actual code implementation blocks."""
        code_blocks = []
        
        if analysis.programming_language == "python":
            code_blocks.extend(self._generate_python_code(question, analysis, context))
        elif analysis.programming_language == "javascript":
            code_blocks.extend(self._generate_javascript_code(question, analysis, context))
        elif analysis.programming_language == "java":
            code_blocks.extend(self._generate_java_code(question, analysis, context))
        else:
            # Generic code template
            code_blocks.append({
                "filename": f"main.{self._get_file_extension(analysis.programming_language)}",
                "description": "Main implementation file",
                "code": self._generate_generic_code_template(question, analysis)
            })
        
        return code_blocks
    
    def _generate_python_code(
        self,
        question: str,
        analysis: CodeAnalysis,
        context: Optional[Context]
    ) -> List[Dict[str, str]]:
        """Generate Python-specific code implementations."""
        code_blocks = []
        question_lower = question.lower()
        
        if "api" in question_lower or "fastapi" in analysis.framework_requirements:
            # FastAPI implementation
            code_blocks.append({
                "filename": "main.py",
                "description": "FastAPI application with endpoints",
                "code": '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="API Implementation", version="1.0.0")

class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

# In-memory storage (replace with database in production)
items_db = []

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {"message": "API is running", "version": "1.0.0"}

@app.get("/items", response_model=List[Item])
async def get_items():
    """Get all items."""
    return items_db

@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get a specific item by ID."""
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    """Create a new item."""
    item.id = len(items_db) + 1
    items_db.append(item)
    return item

@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    """Update an existing item."""
    for i, existing_item in enumerate(items_db):
        if existing_item.id == item_id:
            item.id = item_id
            items_db[i] = item
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item."""
    for i, item in enumerate(items_db):
        if item.id == item_id:
            del items_db[i]
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
            })
        
        elif "class" in question_lower or "function" in question_lower:
            # General Python class/function implementation
            code_blocks.append({
                "filename": "implementation.py",
                "description": "Core implementation with classes and functions",
                "code": '''"""
Core implementation module.
"""
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class Implementation:
    """Main implementation class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the implementation.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._data = []
        logger.info("Implementation initialized")
    
    def process(self, input_data: Any) -> Any:
        """Process input data and return result.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processed result
            
        Raises:
            ValueError: If input data is invalid
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        try:
            # Process the data
            result = self._process_impl(input_data)
            logger.debug(f"Processed data successfully: {type(result)}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    def _process_impl(self, data: Any) -> Any:
        """Internal processing implementation.
        
        Args:
            data: Data to process
            
        Returns:
            Processed result
        """
        # Implement your specific logic here
        return data
    
    def add_item(self, item: Any) -> None:
        """Add an item to the internal storage.
        
        Args:
            item: Item to add
        """
        self._data.append(item)
        logger.debug(f"Added item: {item}")
    
    def get_items(self) -> List[Any]:
        """Get all stored items.
        
        Returns:
            List of all items
        """
        return self._data.copy()
    
    def clear(self) -> None:
        """Clear all stored items."""
        self._data.clear()
        logger.debug("Cleared all items")

def utility_function(param1: str, param2: int = 0) -> str:
    """Utility function for common operations.
    
    Args:
        param1: String parameter
        param2: Integer parameter with default value
        
    Returns:
        Processed string result
    """
    if not param1:
        return ""
    
    return f"{param1}_{param2}"'''
            })
        
        return code_blocks
    
    def _generate_javascript_code(
        self,
        question: str,
        analysis: CodeAnalysis,
        context: Optional[Context]
    ) -> List[Dict[str, str]]:
        """Generate JavaScript-specific code implementations."""
        code_blocks = []
        question_lower = question.lower()
        
        if "api" in question_lower or "express" in analysis.framework_requirements:
            # Express.js API implementation
            code_blocks.append({
                "filename": "server.js",
                "description": "Express.js server with REST API endpoints",
                "code": '''const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// In-memory storage (replace with database in production)
let items = [];
let nextId = 1;

// Error handling middleware
const asyncHandler = (fn) => (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
};

// Routes
app.get('/', (req, res) => {
    res.json({ 
        message: 'API is running', 
        version: '1.0.0',
        endpoints: ['/items', '/items/:id']
    });
});

app.get('/items', asyncHandler(async (req, res) => {
    res.json({
        success: true,
        data: items,
        count: items.length
    });
}));

app.get('/items/:id', asyncHandler(async (req, res) => {
    const id = parseInt(req.params.id);
    const item = items.find(item => item.id === id);
    
    if (!item) {
        return res.status(404).json({
            success: false,
            error: 'Item not found'
        });
    }
    
    res.json({
        success: true,
        data: item
    });
}));

app.post('/items', asyncHandler(async (req, res) => {
    const { name, description } = req.body;
    
    if (!name) {
        return res.status(400).json({
            success: false,
            error: 'Name is required'
        });
    }
    
    const newItem = {
        id: nextId++,
        name,
        description: description || '',
        createdAt: new Date().toISOString()
    };
    
    items.push(newItem);
    
    res.status(201).json({
        success: true,
        data: newItem
    });
}));

app.put('/items/:id', asyncHandler(async (req, res) => {
    const id = parseInt(req.params.id);
    const { name, description } = req.body;
    
    const itemIndex = items.findIndex(item => item.id === id);
    
    if (itemIndex === -1) {
        return res.status(404).json({
            success: false,
            error: 'Item not found'
        });
    }
    
    items[itemIndex] = {
        ...items[itemIndex],
        name: name || items[itemIndex].name,
        description: description !== undefined ? description : items[itemIndex].description,
        updatedAt: new Date().toISOString()
    };
    
    res.json({
        success: true,
        data: items[itemIndex]
    });
}));

app.delete('/items/:id', asyncHandler(async (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);
    
    if (itemIndex === -1) {
        return res.status(404).json({
            success: false,
            error: 'Item not found'
        });
    }
    
    items.splice(itemIndex, 1);
    
    res.json({
        success: true,
        message: 'Item deleted successfully'
    });
}));

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({
        success: false,
        error: 'Internal server error'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found'
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;'''
            })
        
        elif "react" in analysis.framework_requirements:
            # React component implementation
            code_blocks.append({
                "filename": "Component.jsx",
                "description": "React component with hooks and state management",
                "code": '''import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';

const ItemManager = ({ initialItems = [] }) => {
    const [items, setItems] = useState(initialItems);
    const [newItem, setNewItem] = useState({ name: '', description: '' });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Load items on component mount
    useEffect(() => {
        loadItems();
    }, []);

    const loadItems = useCallback(async () => {
        setLoading(true);
        setError(null);
        
        try {
            // Replace with actual API call
            const response = await fetch('/api/items');
            if (!response.ok) {
                throw new Error('Failed to load items');
            }
            const data = await response.json();
            setItems(data.data || []);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    const handleAddItem = async (e) => {
        e.preventDefault();
        
        if (!newItem.name.trim()) {
            setError('Name is required');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/items', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newItem),
            });

            if (!response.ok) {
                throw new Error('Failed to add item');
            }

            const data = await response.json();
            setItems(prev => [...prev, data.data]);
            setNewItem({ name: '', description: '' });
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteItem = async (id) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/items/${id}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete item');
            }

            setItems(prev => prev.filter(item => item.id !== id));
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewItem(prev => ({
            ...prev,
            [name]: value
        }));
    };

    if (loading && items.length === 0) {
        return <div className="loading">Loading...</div>;
    }

    return (
        <div className="item-manager">
            <h2>Item Manager</h2>
            
            {error && (
                <div className="error" role="alert">
                    {error}
                </div>
            )}

            <form onSubmit={handleAddItem} className="add-item-form">
                <div className="form-group">
                    <label htmlFor="name">Name:</label>
                    <input
                        type="text"
                        id="name"
                        name="name"
                        value={newItem.name}
                        onChange={handleInputChange}
                        required
                        disabled={loading}
                    />
                </div>
                
                <div className="form-group">
                    <label htmlFor="description">Description:</label>
                    <textarea
                        id="description"
                        name="description"
                        value={newItem.description}
                        onChange={handleInputChange}
                        disabled={loading}
                    />
                </div>
                
                <button type="submit" disabled={loading}>
                    {loading ? 'Adding...' : 'Add Item'}
                </button>
            </form>

            <div className="items-list">
                <h3>Items ({items.length})</h3>
                {items.length === 0 ? (
                    <p>No items found.</p>
                ) : (
                    <ul>
                        {items.map(item => (
                            <li key={item.id} className="item">
                                <div className="item-content">
                                    <h4>{item.name}</h4>
                                    {item.description && <p>{item.description}</p>}
                                </div>
                                <button
                                    onClick={() => handleDeleteItem(item.id)}
                                    disabled={loading}
                                    className="delete-btn"
                                >
                                    Delete
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
};

ItemManager.propTypes = {
    initialItems: PropTypes.array,
};

export default ItemManager;'''
            })
        
        return code_blocks
    
    def _generate_java_code(
        self,
        question: str,
        analysis: CodeAnalysis,
        context: Optional[Context]
    ) -> List[Dict[str, str]]:
        """Generate Java-specific code implementations."""
        code_blocks = []
        question_lower = question.lower()
        
        if "spring" in analysis.framework_requirements or "api" in question_lower:
            # Spring Boot REST API implementation
            code_blocks.append({
                "filename": "ItemController.java",
                "description": "Spring Boot REST controller with CRUD operations",
                "code": '''package com.example.api.controller;

import com.example.api.model.Item;
import com.example.api.service.ItemService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/items")
@CrossOrigin(origins = "*")
public class ItemController {

    @Autowired
    private ItemService itemService;

    @GetMapping
    public ResponseEntity<List<Item>> getAllItems() {
        List<Item> items = itemService.getAllItems();
        return ResponseEntity.ok(items);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Item> getItemById(@PathVariable Long id) {
        Optional<Item> item = itemService.getItemById(id);
        return item.map(ResponseEntity::ok)
                  .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<Item> createItem(@Valid @RequestBody Item item) {
        try {
            Item createdItem = itemService.createItem(item);
            return ResponseEntity.status(HttpStatus.CREATED).body(createdItem);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PutMapping("/{id}")
    public ResponseEntity<Item> updateItem(@PathVariable Long id, 
                                          @Valid @RequestBody Item item) {
        try {
            Item updatedItem = itemService.updateItem(id, item);
            return ResponseEntity.ok(updatedItem);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteItem(@PathVariable Long id) {
        try {
            itemService.deleteItem(id);
            return ResponseEntity.noContent().build();
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
}'''
            })
            
            code_blocks.append({
                "filename": "Item.java",
                "description": "JPA Entity model class",
                "code": '''package com.example.api.model;

import javax.persistence.*;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;
import java.time.LocalDateTime;

@Entity
@Table(name = "items")
public class Item {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Name is required")
    @Size(max = 100, message = "Name must not exceed 100 characters")
    @Column(nullable = false, length = 100)
    private String name;

    @Size(max = 500, message = "Description must not exceed 500 characters")
    @Column(length = 500)
    private String description;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    // Constructors
    public Item() {}

    public Item(String name, String description) {
        this.name = name;
        this.description = description;
    }

    // JPA lifecycle callbacks
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public String toString() {
        return "Item{" +
                "id=" + id +
                ", name='" + name + '\\'' +
                ", description='" + description + '\\'' +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                '}';
    }
}'''
            })
        
        return code_blocks
    
    def _generate_generic_code_template(self, question: str, analysis: CodeAnalysis) -> str:
        """Generate a generic code template for unsupported languages."""
        return f'''// {analysis.programming_language.title()} Implementation
// Generated for: {question[:50]}...

// Main implementation structure
public class Implementation {{
    
    // Constructor
    public Implementation() {{
        // Initialize implementation
    }}
    
    // Main processing method
    public Object process(Object input) {{
        // Implement your logic here
        return input;
    }}
    
    // Utility methods
    public void setup() {{
        // Setup code
    }}
    
    public void cleanup() {{
        // Cleanup code
    }}
}}'''
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "go": "go",
            "rust": "rs",
            "cpp": "cpp",
            "c": "c"
        }
        return extensions.get(language, "txt")
    
    def _identify_dependencies(self, analysis: CodeAnalysis) -> List[str]:
        """Identify required dependencies based on analysis."""
        dependencies = []
        
        if analysis.programming_language == "python":
            if "fastapi" in analysis.framework_requirements:
                dependencies.extend(["fastapi", "uvicorn", "pydantic"])
            if "django" in analysis.framework_requirements:
                dependencies.extend(["django", "djangorestframework"])
            if "pytest" in analysis.framework_requirements:
                dependencies.append("pytest")
            
            # Common Python dependencies
            dependencies.extend(["requests", "python-dotenv"])
        
        elif analysis.programming_language == "javascript":
            if "express" in analysis.framework_requirements:
                dependencies.extend(["express", "cors", "helmet"])
            if "react" in analysis.framework_requirements:
                dependencies.extend(["react", "react-dom", "prop-types"])
            if "jest" in analysis.framework_requirements:
                dependencies.append("jest")
        
        elif analysis.programming_language == "java":
            if "spring" in analysis.framework_requirements:
                dependencies.extend([
                    "spring-boot-starter-web",
                    "spring-boot-starter-data-jpa",
                    "spring-boot-starter-validation"
                ])
        
        return dependencies
    
    def _generate_setup_instructions(self, analysis: CodeAnalysis) -> List[str]:
        """Generate setup instructions based on analysis."""
        instructions = []
        
        if analysis.programming_language == "python":
            instructions.extend([
                "Create a virtual environment: python -m venv venv",
                "Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)",
                "Install dependencies: pip install -r requirements.txt"
            ])
            
            if "fastapi" in analysis.framework_requirements:
                instructions.append("Run the application: uvicorn main:app --reload")
        
        elif analysis.programming_language == "javascript":
            instructions.extend([
                "Initialize npm project: npm init -y",
                "Install dependencies: npm install",
                "Create .env file for environment variables"
            ])
            
            if "express" in analysis.framework_requirements:
                instructions.append("Start the server: npm start or node server.js")
            if "react" in analysis.framework_requirements:
                instructions.append("Start development server: npm start")
        
        elif analysis.programming_language == "java":
            instructions.extend([
                "Ensure Java 11+ is installed",
                "Build the project: ./mvnw clean compile",
                "Run tests: ./mvnw test"
            ])
            
            if "spring" in analysis.framework_requirements:
                instructions.append("Run the application: ./mvnw spring-boot:run")
        
        return instructions
    
    def _generate_usage_examples(self, question: str, analysis: CodeAnalysis) -> List[str]:
        """Generate usage examples based on the implementation."""
        examples = []
        question_lower = question.lower()
        
        if "api" in question_lower:
            examples.extend([
                "# Get all items\ncurl -X GET http://localhost:8000/items",
                "# Create a new item\ncurl -X POST http://localhost:8000/items -H 'Content-Type: application/json' -d '{\"name\": \"Test Item\", \"description\": \"Test description\"}'",
                "# Get specific item\ncurl -X GET http://localhost:8000/items/1",
                "# Update item\ncurl -X PUT http://localhost:8000/items/1 -H 'Content-Type: application/json' -d '{\"name\": \"Updated Item\"}'",
                "# Delete item\ncurl -X DELETE http://localhost:8000/items/1"
            ])
        
        elif analysis.programming_language == "python":
            examples.extend([
                "# Basic usage example\nfrom implementation import Implementation\n\n# Create instance\nimpl = Implementation()\n\n# Process data\nresult = impl.process('input data')\nprint(result)",
                "# Add and retrieve items\nimpl.add_item('item1')\nimpl.add_item('item2')\nitems = impl.get_items()\nprint(f'Items: {items}')"
            ])
        
        elif analysis.programming_language == "javascript":
            examples.extend([
                "// Basic usage example\nconst implementation = new Implementation();\n\n// Process data\nconst result = implementation.process('input data');\nconsole.log(result);",
                "// Async/await usage\nasync function example() {\n  try {\n    const result = await processData('input');\n    console.log('Result:', result);\n  } catch (error) {\n    console.error('Error:', error);\n  }\n}"
            ])
        
        return examples
    
    def _generate_test_cases(self, question: str, analysis: CodeAnalysis) -> List[Dict[str, str]]:
        """Generate test cases based on the implementation."""
        test_cases = []
        
        if analysis.programming_language == "python":
            test_cases.append({
                "filename": "test_implementation.py",
                "description": "Unit tests for the implementation",
                "code": '''import pytest
from implementation import Implementation, utility_function

class TestImplementation:
    """Test cases for Implementation class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.impl = Implementation()
    
    def test_initialization(self):
        """Test implementation initialization."""
        assert self.impl is not None
        assert self.impl.get_items() == []
    
    def test_process_valid_input(self):
        """Test processing with valid input."""
        result = self.impl.process("test input")
        assert result == "test input"
    
    def test_process_none_input(self):
        """Test processing with None input raises ValueError."""
        with pytest.raises(ValueError, match="Input data cannot be None"):
            self.impl.process(None)
    
    def test_add_and_get_items(self):
        """Test adding and retrieving items."""
        self.impl.add_item("item1")
        self.impl.add_item("item2")
        
        items = self.impl.get_items()
        assert len(items) == 2
        assert "item1" in items
        assert "item2" in items
    
    def test_clear_items(self):
        """Test clearing all items."""
        self.impl.add_item("item1")
        self.impl.add_item("item2")
        
        self.impl.clear()
        assert self.impl.get_items() == []

class TestUtilityFunction:
    """Test cases for utility functions."""
    
    def test_utility_function_basic(self):
        """Test utility function with basic inputs."""
        result = utility_function("test", 5)
        assert result == "test_5"
    
    def test_utility_function_default_param(self):
        """Test utility function with default parameter."""
        result = utility_function("test")
        assert result == "test_0"
    
    def test_utility_function_empty_string(self):
        """Test utility function with empty string."""
        result = utility_function("")
        assert result == ""

# Integration tests
class TestIntegration:
    """Integration test cases."""
    
    def test_full_workflow(self):
        """Test complete workflow integration."""
        impl = Implementation({"setting": "value"})
        
        # Add items
        impl.add_item("item1")
        impl.add_item("item2")
        
        # Process items
        items = impl.get_items()
        for item in items:
            result = impl.process(item)
            assert result == item
        
        # Clear and verify
        impl.clear()
        assert len(impl.get_items()) == 0

if __name__ == "__main__":
    pytest.main([__file__])'''
            })
        
        elif analysis.programming_language == "javascript":
            test_cases.append({
                "filename": "test.js",
                "description": "Jest test suite for JavaScript implementation",
                "code": '''const request = require('supertest');
const app = require('./server');

describe('API Endpoints', () => {
    let itemId;

    beforeEach(() => {
        // Reset items array before each test
        // In a real app, you'd reset the database
    });

    describe('GET /', () => {
        it('should return API information', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);

            expect(response.body).toHaveProperty('message');
            expect(response.body).toHaveProperty('version');
        });
    });

    describe('POST /items', () => {
        it('should create a new item', async () => {
            const newItem = {
                name: 'Test Item',
                description: 'Test description'
            };

            const response = await request(app)
                .post('/items')
                .send(newItem)
                .expect(201);

            expect(response.body.success).toBe(true);
            expect(response.body.data).toHaveProperty('id');
            expect(response.body.data.name).toBe(newItem.name);
            
            itemId = response.body.data.id;
        });

        it('should return error for missing name', async () => {
            const invalidItem = {
                description: 'Test description'
            };

            const response = await request(app)
                .post('/items')
                .send(invalidItem)
                .expect(400);

            expect(response.body.success).toBe(false);
            expect(response.body.error).toContain('required');
        });
    });

    describe('GET /items', () => {
        it('should return all items', async () => {
            const response = await request(app)
                .get('/items')
                .expect(200);

            expect(response.body.success).toBe(true);
            expect(response.body).toHaveProperty('data');
            expect(Array.isArray(response.body.data)).toBe(true);
        });
    });

    describe('GET /items/:id', () => {
        beforeEach(async () => {
            // Create an item for testing
            const newItem = { name: 'Test Item', description: 'Test' };
            const response = await request(app)
                .post('/items')
                .send(newItem);
            itemId = response.body.data.id;
        });

        it('should return specific item', async () => {
            const response = await request(app)
                .get(`/items/${itemId}`)
                .expect(200);

            expect(response.body.success).toBe(true);
            expect(response.body.data.id).toBe(itemId);
        });

        it('should return 404 for non-existent item', async () => {
            const response = await request(app)
                .get('/items/99999')
                .expect(404);

            expect(response.body.success).toBe(false);
        });
    });

    describe('PUT /items/:id', () => {
        beforeEach(async () => {
            const newItem = { name: 'Test Item', description: 'Test' };
            const response = await request(app)
                .post('/items')
                .send(newItem);
            itemId = response.body.data.id;
        });

        it('should update existing item', async () => {
            const updatedData = {
                name: 'Updated Item',
                description: 'Updated description'
            };

            const response = await request(app)
                .put(`/items/${itemId}`)
                .send(updatedData)
                .expect(200);

            expect(response.body.success).toBe(true);
            expect(response.body.data.name).toBe(updatedData.name);
        });
    });

    describe('DELETE /items/:id', () => {
        beforeEach(async () => {
            const newItem = { name: 'Test Item', description: 'Test' };
            const response = await request(app)
                .post('/items')
                .send(newItem);
            itemId = response.body.data.id;
        });

        it('should delete existing item', async () => {
            const response = await request(app)
                .delete(`/items/${itemId}`)
                .expect(200);

            expect(response.body.success).toBe(true);

            // Verify item is deleted
            await request(app)
                .get(`/items/${itemId}`)
                .expect(404);
        });
    });
});'''
            })
        
        return test_cases
    
    def _format_development_response(
        self,
        question: str,
        analysis: Optional[CodeAnalysis],
        implementation: Optional[CodeImplementation],
        context: Optional[Context]
    ) -> str:
        """Format the comprehensive development response."""
        response = "# Development Implementation Response\n\n"
        
        # Problem Analysis Section
        response += "## 1. Problem Analysis\n\n"
        if analysis:
            response += f"**Understanding:** {analysis.problem_understanding}\n\n"
            response += f"**Programming Language:** {analysis.programming_language.title()}\n\n"
            
            if analysis.framework_requirements:
                response += "**Framework Requirements:**\n"
                for framework in analysis.framework_requirements:
                    response += f"- {framework}\n"
                response += "\n"
            
            response += f"**Complexity Level:** {analysis.complexity_level.title()}\n\n"
            
            if analysis.key_requirements:
                response += "**Key Requirements:**\n"
                for requirement in analysis.key_requirements:
                    response += f"- {requirement}\n"
                response += "\n"
        
        # Implementation Strategy Section
        if implementation:
            response += "## 2. Implementation Strategy\n\n"
            response += f"{implementation.strategy}\n\n"
            
            # Code Implementation Section
            if implementation.code_blocks:
                response += "## 3. Code Implementation\n\n"
                for i, code_block in enumerate(implementation.code_blocks, 1):
                    response += f"### {code_block['filename']}\n\n"
                    response += f"*{code_block['description']}*\n\n"
                    response += f"```{analysis.programming_language if analysis else 'text'}\n"
                    response += f"{code_block['code']}\n"
                    response += "```\n\n"
            
            # Dependencies Section
            if implementation.dependencies:
                response += "## 4. Dependencies & Setup\n\n"
                response += "**Required Dependencies:**\n"
                for dep in implementation.dependencies:
                    response += f"- {dep}\n"
                response += "\n"
                
                if implementation.setup_instructions:
                    response += "**Setup Instructions:**\n"
                    for i, instruction in enumerate(implementation.setup_instructions, 1):
                        response += f"{i}. {instruction}\n"
                    response += "\n"
            
            # Usage Examples Section
            if implementation.usage_examples:
                response += "## 5. Usage Examples\n\n"
                for i, example in enumerate(implementation.usage_examples, 1):
                    response += f"### Example {i}\n\n"
                    response += f"```{analysis.programming_language if analysis else 'bash'}\n"
                    response += f"{example}\n"
                    response += "```\n\n"
            
            # Test Cases Section
            if implementation.test_cases:
                response += "## 6. Testing\n\n"
                for test_case in implementation.test_cases:
                    response += f"### {test_case['filename']}\n\n"
                    response += f"*{test_case['description']}*\n\n"
                    response += f"```{analysis.programming_language if analysis else 'text'}\n"
                    response += f"{test_case['code']}\n"
                    response += "```\n\n"
        
        # Best Practices Section
        response += "## 7. Best Practices & Recommendations\n\n"
        
        if analysis and analysis.programming_language == "python":
            response += "**Python Best Practices:**\n"
            response += "- Follow PEP 8 style guidelines\n"
            response += "- Use type hints for better code documentation\n"
            response += "- Implement proper error handling with try/except blocks\n"
            response += "- Write comprehensive docstrings for functions and classes\n"
            response += "- Use virtual environments for dependency management\n\n"
        
        elif analysis and analysis.programming_language == "javascript":
            response += "**JavaScript Best Practices:**\n"
            response += "- Use ES6+ features (const/let, arrow functions, async/await)\n"
            response += "- Implement proper error handling with try/catch blocks\n"
            response += "- Use ESLint and Prettier for code formatting\n"
            response += "- Write comprehensive tests with Jest or similar frameworks\n"
            response += "- Use environment variables for configuration\n\n"
        
        elif analysis and analysis.programming_language == "java":
            response += "**Java Best Practices:**\n"
            response += "- Follow Java naming conventions\n"
            response += "- Use proper exception handling\n"
            response += "- Implement proper logging with SLF4J\n"
            response += "- Write unit tests with JUnit\n"
            response += "- Use dependency injection with Spring\n\n"
        
        response += "**General Development Best Practices:**\n"
        response += "- Write clean, readable, and maintainable code\n"
        response += "- Implement comprehensive error handling\n"
        response += "- Add proper logging and monitoring\n"
        response += "- Write unit and integration tests\n"
        response += "- Use version control (Git) effectively\n"
        response += "- Document your code and APIs\n"
        response += "- Follow security best practices\n"
        response += "- Optimize for performance when necessary\n\n"
        
        # Next Steps Section
        response += "## 8. Next Steps\n\n"
        response += "1. **Setup Development Environment**: Follow the setup instructions above\n"
        response += "2. **Implement Core Functionality**: Start with the main implementation files\n"
        response += "3. **Add Tests**: Implement the provided test cases and add more as needed\n"
        response += "4. **Error Handling**: Add comprehensive error handling and validation\n"
        response += "5. **Documentation**: Add detailed documentation and API specs\n"
        response += "6. **Performance Testing**: Test performance under expected load\n"
        response += "7. **Security Review**: Implement security best practices\n"
        response += "8. **Deployment**: Prepare for production deployment\n\n"
        
        return response
    
    def _calculate_confidence(
        self,
        question: str,
        context: Optional[Context],
        analysis: Optional[CodeAnalysis]
    ) -> float:
        """Calculate confidence score for the development response."""
        confidence = 0.7  # Base confidence
        
        # Adjust based on context availability
        if context and context.documents:
            confidence += 0.1 * min(len(context.documents), 3) / 3
        
        # Adjust based on analysis quality
        if analysis:
            if analysis.programming_language in self._language_patterns:
                confidence += 0.1
            
            if analysis.framework_requirements:
                confidence += 0.05
            
            if analysis.complexity_level == "simple":
                confidence += 0.1
            elif analysis.complexity_level == "medium":
                confidence += 0.05
        
        # Adjust based on question clarity
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ["implement", "code", "create", "build"]):
            confidence += 0.05
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _create_error_response(self, message_id: str, error_message: str) -> Response:
        """Create an error response."""
        return Response(
            message_id=message_id,
            agent_id=self.agent_id,
            content=f"I encountered an error while processing your development request: {error_message}",
            confidence_score=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def generate_code(
        self,
        requirements: str,
        language: str = "python",
        framework: Optional[str] = None
    ) -> Optional[CodeImplementation]:
        """Public method to generate code based on requirements."""
        # Create a mock analysis for the public API
        analysis = CodeAnalysis(
            problem_understanding=f"Code generation request for {language}",
            programming_language=language,
            framework_requirements=[framework] if framework else [],
            complexity_level="medium",
            key_requirements=["Clean implementation", "Error handling", "Documentation"]
        )
        
        return self._generate_code_implementation(requirements, analysis, None)
    
    def analyze_requirements(self, requirements: str) -> Optional[CodeAnalysis]:
        """Public method to analyze development requirements."""
        return self._analyze_development_requirements(requirements, None)