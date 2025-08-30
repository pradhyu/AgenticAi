"""Prompt manager for loading and managing external prompt templates."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Set
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class PromptError(Exception):
    """Exception raised for prompt-related errors."""
    pass


class PromptManager:
    """Manager for loading and managing external prompt templates."""
    
    def __init__(self, prompt_dir: str, enable_hot_reload: bool = False):
        """
        Initialize the PromptManager.
        
        Args:
            prompt_dir: Directory containing prompt template files
            enable_hot_reload: Whether to enable hot-reloading of prompt files
        """
        self.prompt_dir = Path(prompt_dir)
        self.enable_hot_reload = enable_hot_reload
        self._prompts: Dict[str, str] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._lock = Lock()
        
        # Validate prompt directory
        if not self.prompt_dir.exists():
            raise PromptError(f"Prompt directory does not exist: {prompt_dir}")
        
        if not self.prompt_dir.is_dir():
            raise PromptError(f"Prompt path is not a directory: {prompt_dir}")
        
        # Load initial prompts
        self.load_prompts()
        
        logger.info(f"PromptManager initialized with {len(self._prompts)} prompts from {prompt_dir}")
    
    def _get_prompt_key(self, agent_type: str, prompt_name: str) -> str:
        """
        Generate a unique key for a prompt.
        
        Args:
            agent_type: Type of agent (e.g., 'analyst', 'architect')
            prompt_name: Name of the prompt (e.g., 'classification', 'design')
            
        Returns:
            Unique prompt key
        """
        return f"{agent_type}.{prompt_name}"
    
    def _get_prompt_file_path(self, agent_type: str, prompt_name: str) -> Path:
        """
        Get the file path for a prompt template.
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
            
        Returns:
            Path to the prompt file
        """
        return self.prompt_dir / agent_type / f"{prompt_name}.txt"
    
    def _load_prompt_file(self, file_path: Path) -> str:
        """
        Load a single prompt file.
        
        Args:
            file_path: Path to the prompt file
            
        Returns:
            Prompt content
            
        Raises:
            PromptError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"Prompt file is empty: {file_path}")
            
            return content
            
        except FileNotFoundError:
            raise PromptError(f"Prompt file not found: {file_path}")
        except PermissionError:
            raise PromptError(f"Permission denied reading prompt file: {file_path}")
        except UnicodeDecodeError:
            raise PromptError(f"Invalid encoding in prompt file: {file_path}")
        except Exception as e:
            raise PromptError(f"Error reading prompt file {file_path}: {e}")
    
    def _scan_prompt_files(self) -> Dict[str, Path]:
        """
        Scan the prompt directory for all prompt files.
        
        Returns:
            Dictionary mapping prompt keys to file paths
        """
        prompt_files = {}
        
        try:
            for agent_dir in self.prompt_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                
                agent_type = agent_dir.name
                
                for prompt_file in agent_dir.glob("*.txt"):
                    prompt_name = prompt_file.stem
                    prompt_key = self._get_prompt_key(agent_type, prompt_name)
                    prompt_files[prompt_key] = prompt_file
            
            return prompt_files
            
        except Exception as e:
            raise PromptError(f"Error scanning prompt directory {self.prompt_dir}: {e}")
    
    def _check_file_modified(self, file_path: Path) -> bool:
        """
        Check if a file has been modified since last load.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file has been modified
        """
        try:
            current_mtime = file_path.stat().st_mtime
            file_key = str(file_path)
            
            if file_key not in self._file_timestamps:
                return True
            
            return current_mtime > self._file_timestamps[file_key]
            
        except Exception:
            # If we can't check the timestamp, assume it's modified
            return True
    
    def load_prompts(self) -> Dict[str, str]:
        """
        Load all prompt templates from the prompt directory.
        
        Returns:
            Dictionary of loaded prompts
            
        Raises:
            PromptError: If prompts cannot be loaded
        """
        with self._lock:
            try:
                prompt_files = self._scan_prompt_files()
                loaded_count = 0
                
                for prompt_key, file_path in prompt_files.items():
                    # Check if we need to reload this file
                    if not self.enable_hot_reload or self._check_file_modified(file_path):
                        content = self._load_prompt_file(file_path)
                        self._prompts[prompt_key] = content
                        self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
                        loaded_count += 1
                
                logger.info(f"Loaded {loaded_count} prompt templates")
                return self._prompts.copy()
                
            except Exception as e:
                raise PromptError(f"Failed to load prompts: {e}")
    
    def get_prompt(self, agent_type: str, prompt_name: str) -> str:
        """
        Get a specific prompt template.
        
        Args:
            agent_type: Type of agent (e.g., 'analyst', 'architect')
            prompt_name: Name of the prompt (e.g., 'classification', 'design')
            
        Returns:
            Prompt template content
            
        Raises:
            PromptError: If prompt is not found
        """
        prompt_key = self._get_prompt_key(agent_type, prompt_name)
        
        # Check for hot reload if enabled
        if self.enable_hot_reload:
            file_path = self._get_prompt_file_path(agent_type, prompt_name)
            if file_path.exists() and self._check_file_modified(file_path):
                with self._lock:
                    try:
                        content = self._load_prompt_file(file_path)
                        self._prompts[prompt_key] = content
                        self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
                        logger.debug(f"Hot-reloaded prompt: {prompt_key}")
                    except Exception as e:
                        logger.error(f"Failed to hot-reload prompt {prompt_key}: {e}")
        
        if prompt_key not in self._prompts:
            raise PromptError(f"Prompt not found: {agent_type}.{prompt_name}")
        
        return self._prompts[prompt_key]
    
    def reload_prompts(self) -> None:
        """
        Reload all prompt templates from disk.
        
        Raises:
            PromptError: If prompts cannot be reloaded
        """
        logger.info("Reloading all prompt templates")
        
        # Clear existing prompts and timestamps
        with self._lock:
            self._prompts.clear()
            self._file_timestamps.clear()
        
        # Load fresh prompts
        self.load_prompts()
    
    def validate_prompts(self) -> bool:
        """
        Validate that all required prompt templates exist and are readable.
        
        Returns:
            True if all prompts are valid
            
        Raises:
            PromptError: If validation fails
        """
        required_prompts = {
            'analyst': ['classification', 'routing', 'aggregation'],
            'architect': ['design'],
            'developer': ['implementation'],
            'code_reviewer': ['review'],
            'tester': ['test_creation'],
        }
        
        missing_prompts = []
        invalid_prompts = []
        
        for agent_type, prompt_names in required_prompts.items():
            for prompt_name in prompt_names:
                prompt_key = self._get_prompt_key(agent_type, prompt_name)
                file_path = self._get_prompt_file_path(agent_type, prompt_name)
                
                # Check if file exists
                if not file_path.exists():
                    missing_prompts.append(f"{agent_type}/{prompt_name}.txt")
                    continue
                
                # Check if prompt is loaded and valid
                try:
                    content = self.get_prompt(agent_type, prompt_name)
                    if not content or len(content.strip()) == 0:
                        invalid_prompts.append(prompt_key)
                except PromptError:
                    invalid_prompts.append(prompt_key)
        
        # Report validation results
        if missing_prompts:
            raise PromptError(f"Missing required prompt files: {missing_prompts}")
        
        if invalid_prompts:
            raise PromptError(f"Invalid or empty prompt templates: {invalid_prompts}")
        
        logger.info("All prompt templates validated successfully")
        return True
    
    def list_available_prompts(self) -> Dict[str, Set[str]]:
        """
        List all available prompts organized by agent type.
        
        Returns:
            Dictionary mapping agent types to sets of available prompt names
        """
        prompts_by_agent = {}
        
        for prompt_key in self._prompts.keys():
            agent_type, prompt_name = prompt_key.split('.', 1)
            
            if agent_type not in prompts_by_agent:
                prompts_by_agent[agent_type] = set()
            
            prompts_by_agent[agent_type].add(prompt_name)
        
        return prompts_by_agent
    
    def get_prompt_info(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about all loaded prompts.
        
        Returns:
            Dictionary with prompt information including size and modification time
        """
        info = {}
        
        for prompt_key, content in self._prompts.items():
            agent_type, prompt_name = prompt_key.split('.', 1)
            file_path = self._get_prompt_file_path(agent_type, prompt_name)
            
            info[prompt_key] = {
                'agent_type': agent_type,
                'prompt_name': prompt_name,
                'file_path': str(file_path),
                'content_length': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.splitlines()),
                'exists': file_path.exists(),
                'last_modified': self._file_timestamps.get(str(file_path), 0),
            }
        
        return info
    
    def format_prompt(self, agent_type: str, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt template and format it with provided variables.
        
        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted prompt content
            
        Raises:
            PromptError: If prompt is not found or formatting fails
        """
        try:
            template = self.get_prompt(agent_type, prompt_name)
            return template.format(**kwargs)
        except KeyError as e:
            raise PromptError(f"Missing template variable in prompt {agent_type}.{prompt_name}: {e}")
        except Exception as e:
            raise PromptError(f"Error formatting prompt {agent_type}.{prompt_name}: {e}")