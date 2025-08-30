"""Tests for prompt management."""

import os
import tempfile
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from deep_agent_system.prompts.manager import PromptManager, PromptError


class TestPromptManager:
    """Test PromptManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompt_dir = Path(self.temp_dir)
        
        # Create test prompt structure
        self._create_test_prompts()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_prompts(self):
        """Create test prompt files."""
        # Create agent directories
        (self.prompt_dir / "analyst").mkdir(parents=True)
        (self.prompt_dir / "architect").mkdir(parents=True)
        (self.prompt_dir / "developer").mkdir(parents=True)
        
        # Create test prompt files
        test_prompts = {
            "analyst/classification.txt": "You are an analyst. Classify this: {user_question}",
            "analyst/routing.txt": "Route this question: {user_question} to {target_agent}",
            "architect/design.txt": "Design a solution for: {user_question}",
            "developer/implementation.txt": "Implement: {user_question} using {technology}",
        }
        
        for file_path, content in test_prompts.items():
            full_path = self.prompt_dir / file_path
            full_path.write_text(content, encoding='utf-8')
    
    def test_init_success(self):
        """Test successful PromptManager initialization."""
        manager = PromptManager(str(self.prompt_dir))
        
        assert manager.prompt_dir == self.prompt_dir
        assert not manager.enable_hot_reload
        assert len(manager._prompts) > 0
    
    def test_init_with_hot_reload(self):
        """Test PromptManager initialization with hot reload enabled."""
        manager = PromptManager(str(self.prompt_dir), enable_hot_reload=True)
        
        assert manager.enable_hot_reload is True
    
    def test_init_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(PromptError, match="Prompt directory does not exist"):
            PromptManager("/nonexistent/directory")
    
    def test_init_file_instead_of_directory(self):
        """Test initialization with file path instead of directory."""
        # Create a file instead of directory
        file_path = self.prompt_dir / "not_a_directory.txt"
        file_path.write_text("test")
        
        with pytest.raises(PromptError, match="Prompt path is not a directory"):
            PromptManager(str(file_path))
    
    def test_get_prompt_key(self):
        """Test prompt key generation."""
        manager = PromptManager(str(self.prompt_dir))
        
        key = manager._get_prompt_key("analyst", "classification")
        assert key == "analyst.classification"
    
    def test_get_prompt_file_path(self):
        """Test prompt file path generation."""
        manager = PromptManager(str(self.prompt_dir))
        
        path = manager._get_prompt_file_path("analyst", "classification")
        expected = self.prompt_dir / "analyst" / "classification.txt"
        assert path == expected
    
    def test_load_prompt_file_success(self):
        """Test successful prompt file loading."""
        manager = PromptManager(str(self.prompt_dir))
        
        file_path = self.prompt_dir / "analyst" / "classification.txt"
        content = manager._load_prompt_file(file_path)
        
        assert "You are an analyst" in content
        assert "{user_question}" in content
    
    def test_load_prompt_file_not_found(self):
        """Test loading non-existent prompt file."""
        manager = PromptManager(str(self.prompt_dir))
        
        file_path = self.prompt_dir / "nonexistent.txt"
        with pytest.raises(PromptError, match="Prompt file not found"):
            manager._load_prompt_file(file_path)
    
    def test_load_prompt_file_empty(self):
        """Test loading empty prompt file."""
        manager = PromptManager(str(self.prompt_dir))
        
        # Create empty file
        empty_file = self.prompt_dir / "empty.txt"
        empty_file.write_text("")
        
        with patch('deep_agent_system.prompts.manager.logger') as mock_logger:
            content = manager._load_prompt_file(empty_file)
            assert content == ""
            mock_logger.warning.assert_called_once()
    
    def test_scan_prompt_files(self):
        """Test scanning prompt directory for files."""
        manager = PromptManager(str(self.prompt_dir))
        
        prompt_files = manager._scan_prompt_files()
        
        assert "analyst.classification" in prompt_files
        assert "analyst.routing" in prompt_files
        assert "architect.design" in prompt_files
        assert "developer.implementation" in prompt_files
    
    def test_load_prompts(self):
        """Test loading all prompts."""
        manager = PromptManager(str(self.prompt_dir))
        
        prompts = manager.load_prompts()
        
        assert len(prompts) >= 4
        assert "analyst.classification" in prompts
        assert "You are an analyst" in prompts["analyst.classification"]
    
    def test_get_prompt_success(self):
        """Test successful prompt retrieval."""
        manager = PromptManager(str(self.prompt_dir))
        
        prompt = manager.get_prompt("analyst", "classification")
        assert "You are an analyst" in prompt
        assert "{user_question}" in prompt
    
    def test_get_prompt_not_found(self):
        """Test retrieving non-existent prompt."""
        manager = PromptManager(str(self.prompt_dir))
        
        with pytest.raises(PromptError, match="Prompt not found"):
            manager.get_prompt("nonexistent", "prompt")
    
    def test_get_prompt_hot_reload(self):
        """Test prompt hot reloading."""
        manager = PromptManager(str(self.prompt_dir), enable_hot_reload=True)
        
        # Get initial prompt
        original_prompt = manager.get_prompt("analyst", "classification")
        
        # Modify the file
        file_path = self.prompt_dir / "analyst" / "classification.txt"
        time.sleep(0.1)  # Ensure different timestamp
        file_path.write_text("Modified prompt: {user_question}")
        
        # Get prompt again - should be reloaded
        modified_prompt = manager.get_prompt("analyst", "classification")
        assert modified_prompt != original_prompt
        assert "Modified prompt" in modified_prompt
    
    def test_reload_prompts(self):
        """Test reloading all prompts."""
        manager = PromptManager(str(self.prompt_dir))
        
        # Modify a file
        file_path = self.prompt_dir / "analyst" / "classification.txt"
        file_path.write_text("Reloaded prompt: {user_question}")
        
        # Reload prompts
        manager.reload_prompts()
        
        # Check that prompt was reloaded
        prompt = manager.get_prompt("analyst", "classification")
        assert "Reloaded prompt" in prompt
    
    def test_validate_prompts_success(self):
        """Test successful prompt validation."""
        # Create all required prompts
        required_files = [
            "analyst/aggregation.txt",
            "code_reviewer/review.txt",
            "tester/test_creation.txt",
        ]
        
        for file_path in required_files:
            full_path = self.prompt_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Test prompt for {file_path}")
        
        manager = PromptManager(str(self.prompt_dir))
        assert manager.validate_prompts() is True
    
    def test_validate_prompts_missing_files(self):
        """Test prompt validation with missing files."""
        manager = PromptManager(str(self.prompt_dir))
        
        with pytest.raises(PromptError, match="Missing required prompt files"):
            manager.validate_prompts()
    
    def test_validate_prompts_empty_content(self):
        """Test prompt validation with empty content."""
        # Create all required files but make one empty
        required_files = [
            "analyst/aggregation.txt",
            "code_reviewer/review.txt",
            "tester/test_creation.txt",
        ]
        
        for file_path in required_files:
            full_path = self.prompt_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"Test prompt for {file_path}" if "aggregation" not in file_path else ""
            full_path.write_text(content)
        
        manager = PromptManager(str(self.prompt_dir))
        
        with pytest.raises(PromptError, match="Invalid or empty prompt templates"):
            manager.validate_prompts()
    
    def test_list_available_prompts(self):
        """Test listing available prompts."""
        manager = PromptManager(str(self.prompt_dir))
        
        prompts_by_agent = manager.list_available_prompts()
        
        assert "analyst" in prompts_by_agent
        assert "classification" in prompts_by_agent["analyst"]
        assert "routing" in prompts_by_agent["analyst"]
        assert "architect" in prompts_by_agent
        assert "design" in prompts_by_agent["architect"]
    
    def test_get_prompt_info(self):
        """Test getting prompt information."""
        manager = PromptManager(str(self.prompt_dir))
        
        info = manager.get_prompt_info()
        
        assert "analyst.classification" in info
        prompt_info = info["analyst.classification"]
        assert prompt_info["agent_type"] == "analyst"
        assert prompt_info["prompt_name"] == "classification"
        assert prompt_info["content_length"] > 0
        assert prompt_info["word_count"] > 0
        assert prompt_info["exists"] is True
    
    def test_format_prompt_success(self):
        """Test successful prompt formatting."""
        manager = PromptManager(str(self.prompt_dir))
        
        formatted = manager.format_prompt(
            "analyst", 
            "classification", 
            user_question="What is the best architecture?"
        )
        
        assert "What is the best architecture?" in formatted
        assert "{user_question}" not in formatted
    
    def test_format_prompt_missing_variable(self):
        """Test prompt formatting with missing variable."""
        manager = PromptManager(str(self.prompt_dir))
        
        with pytest.raises(PromptError, match="Missing template variable"):
            manager.format_prompt("analyst", "classification")  # Missing user_question
    
    def test_format_prompt_not_found(self):
        """Test formatting non-existent prompt."""
        manager = PromptManager(str(self.prompt_dir))
        
        with pytest.raises(PromptError, match="Prompt not found"):
            manager.format_prompt("nonexistent", "prompt", user_question="test")
    
    def test_check_file_modified(self):
        """Test file modification checking."""
        manager = PromptManager(str(self.prompt_dir))
        
        file_path = self.prompt_dir / "analyst" / "classification.txt"
        
        # File should already be in timestamps after initialization
        # Should return False for unmodified file
        assert manager._check_file_modified(file_path) is False
        
        # Modify file
        time.sleep(0.1)  # Ensure different timestamp
        file_path.write_text("Modified content")
        
        # Should return True now
        assert manager._check_file_modified(file_path) is True
        
        # Test with new file not in timestamps
        new_file = self.prompt_dir / "new_file.txt"
        new_file.write_text("New content")
        
        # Should return True for new file
        assert manager._check_file_modified(new_file) is True
    
    def test_concurrent_access(self):
        """Test thread-safe access to prompts."""
        import threading
        
        manager = PromptManager(str(self.prompt_dir))
        results = []
        errors = []
        
        def get_prompt_worker():
            try:
                prompt = manager.get_prompt("analyst", "classification")
                results.append(len(prompt))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_prompt_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 10
        assert all(r > 0 for r in results)  # All should have content


class TestPromptManagerIntegration:
    """Integration tests for PromptManager."""
    
    def test_real_prompt_directory(self):
        """Test with the actual prompt directory structure."""
        # Use the real prompts directory
        prompt_dir = Path("prompts")
        
        if not prompt_dir.exists():
            pytest.skip("Real prompts directory not found")
        
        manager = PromptManager(str(prompt_dir))
        
        # Test that we can load real prompts
        assert len(manager._prompts) > 0
        
        # Test specific prompts exist
        classification_prompt = manager.get_prompt("analyst", "classification")
        assert len(classification_prompt) > 0
        assert "analyst" in classification_prompt.lower()
        
        design_prompt = manager.get_prompt("architect", "design")
        assert len(design_prompt) > 0
        assert "architect" in design_prompt.lower()
    
    def test_prompt_formatting_with_real_prompts(self):
        """Test prompt formatting with real prompt templates."""
        prompt_dir = Path("prompts")
        
        if not prompt_dir.exists():
            pytest.skip("Real prompts directory not found")
        
        manager = PromptManager(str(prompt_dir))
        
        # Test formatting analyst classification prompt
        formatted = manager.format_prompt(
            "analyst",
            "classification",
            user_question="How do I implement a REST API?"
        )
        
        assert "How do I implement a REST API?" in formatted
        assert "{user_question}" not in formatted
        
        # Test formatting architect design prompt
        formatted = manager.format_prompt(
            "architect",
            "design",
            user_question="Design a microservices architecture",
            context="For an e-commerce platform"
        )
        
        assert "Design a microservices architecture" in formatted
        assert "For an e-commerce platform" in formatted